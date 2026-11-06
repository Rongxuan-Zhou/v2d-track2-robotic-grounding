"""第 3 步配方试验的判定（计划 agent-doc/plan-step3-recipe-trial-2026-10-04.md 第 2 版第 4 节）。只用标准库，Mac 与训练机宿主都能跑。

输入是试验清单（stages/evaluate/trials/r3_t<档>_<机器>.json，stages/policy/make_trial_queue.py 生成）：每条试验 episode 的四个 run
（基线 seed 42、变体 A seed 42、基线 seed 43、变体 A seed 43）与对照 episode 的两个 run（基线、变体 A，都是 seed 42）。
评测结果一律用 8c778f9 评测器的数：
  基线 seed 42：fc_export.sh（V2D_EXPORT_WORLDS=4096 V2D_EXPORT_TRAJ=0）写的 runs/flash_chord/<run>/export_8c778f9/s<步数>_w4096/episode_<序号>.json
  其余：评测链 fc_eval.sh 写的 runs/flash_chord/<run>/eval_<步数>_w4096.json
来历：JSON 的 checkpoint_sha256 等于 policy 文件的 sha256，步数与世界数对得上，评测日志里最后一行 in-flash-chord 挂的源码是 8c778f905c 副本。
训练健康（S5）：run 目录有 DONE、没有 NONFINITE，训练日志里没有 NONFINITE_GUARD_STOP。chord_sr 只报，不进判据（与终止阈值有关）。

用法（仓库根）：
  python3 stages/evaluate/trial_verdict.py <清单> --midcheck   段间门（中途检查，只许判停）：退出码 0 = CONTINUE，1 = STOP，2 = 读不全或来历不对
  python3 stages/evaluate/trial_verdict.py <清单>              全部评完后的判定：S1–S6 与只报项；最后一行 TRIAL_VERDICT PASS / FAIL /
                                                               INCOMPLETE（退出码 3）/ STOPPED_AT_MIDCHECK（BC 段没开跑，退出码 4）
  --root <目录>  换仓库根（测试用）
Tier 2 的清单在读任何结果之前先调 forbid_tier1()。测试：stages/evaluate/test_trial_verdict.py。
"""
import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_TAG = "8c778f905c"
MID_STOP_MAX_GAIN = 0.10   # 中途检查：平均 ≤ 0 且没有一条 ≥ +0.10 才停
S1_MIN = 0.15
S2_MARGIN, S2_MIN_COUNT = 0.05, 3
S3_MP_DROP, S3_AUC_DROP = 0.20, 0.05
S4_AUC, S4_SP, S4_MPPE = 0.02, 0.10, 0.5
S6_DROP = 0.10
KEYS = {"mp": "maniptrans_sr", "sp": "spider_sr_uncentered", "auc": "add_auc", "mppe": "mppe_cm"}


class Bad(Exception):
    """读不全、来历不对或指标非有限。"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(root: Path, run: str, ep: int, steps: int, worlds: int, kind: str) -> dict:
    """读一个 run 的评测结果并核来历。kind：export（基线 seed 42 的重评）或 eval（评测链）。"""
    o = root / "runs" / "flash_chord" / run
    if kind == "export":
        j = o / "export_8c778f9" / f"s{steps}_w{worlds}" / f"episode_{ep:06d}.json"
        log = root / "runs" / "logs" / f"{run}.export_8c778f9_s{steps}_w{worlds}.log"
    else:
        j = o / f"eval_{steps}_w{worlds}.json"
        log = root / "runs" / "logs" / f"{run}.eval_{steps}_w{worlds}.log"
    if not j.is_file():
        raise Bad(f"{run}：没有评测结果 {j.relative_to(root)}")
    d = json.loads(j.read_text())
    pol = o / f"policy_{steps}.safetensors"
    if not pol.is_file():
        raise Bad(f"{run}：没有 {pol.relative_to(root)}")
    if d.get("checkpoint_sha256") != sha256(pol):
        raise Bad(f"{run}：评测记的 checkpoint_sha256 与 {pol.name} 不符")
    if d.get("environment_steps") != steps or d.get("world_count") != worlds:
        raise Bad(f"{run}：评测的步数 / 世界数是 {d.get('environment_steps')} / {d.get('world_count')}，要 {steps} / {worlds}")
    mounts = [x for x in (log.read_text(errors="replace").splitlines() if log.is_file() else []) if "in-flash-chord: /workspace ← " in x]
    if not mounts or SRC_TAG not in mounts[-1]:
        raise Bad(f"{run}：评测日志 {log.name} 里挂的源码不是 8c778f9 副本（{mounts[-1].strip() if mounts else '没有 in-flash-chord 行'}）")
    m = d.get("metrics") or {}
    out = {}
    for short, key in KEYS.items():
        v = m.get(key)
        if not isinstance(v, (int, float)) or v != v or v in (float("inf"), float("-inf")):
            raise Bad(f"{run}：{key} 缺失或非有限（{v!r}）")
        out[short] = float(v)
    out["chord"] = (d.get("reference_tracking") or {}).get("success_rate")
    out["nonfinite"] = list(d.get("non_finite_worlds") or [])
    return out


def health(root: Path, run: str) -> list:
    """S5：训练健康。返回问题列表，空表示健康。"""
    o, logs = root / "runs" / "flash_chord" / run, root / "runs" / "logs"
    problems = []
    if not (o / "DONE").exists():
        problems.append("没有 DONE")
    if (o / "NONFINITE").exists():
        problems.append("有 NONFINITE")
    for lg in [logs / f"{run}.log", *sorted(logs.glob(f"{run}.s*.log"))]:
        if lg.is_file() and "NONFINITE_GUARD_STOP" in lg.read_text(errors="replace"):
            problems.append(f"{lg.name} 有 NONFINITE_GUARD_STOP")
    return problems


def guard(spec: dict, root: Path) -> None:
    if int(spec["tier"]) == 2:
        sys.path.insert(0, str(root / "libs" / "v2d_infra"))
        from tier_guard import forbid_tier1
        forbid_tier1()
        print("净室：Tier 2 清单，已调 forbid_tier1()")


def midcheck(spec: dict, root: Path) -> int:
    steps, worlds = int(spec["steps"]), int(spec["worlds"])
    rows = []
    try:
        for e in spec["episodes"]:
            b = load(root, e["base42"], e["ep"], steps, worlds, "export")
            a = load(root, e["var42"], e["ep"], steps, worlds, "eval")
            rows.append((e["ep"], a["mp"], b["mp"], a["mp"] - b["mp"]))
    except Bad as err:
        print(f"MIDCHECK ERROR {err}")
        return 2
    print(f"中途检查（档位 {spec['tier']}，{spec['machine']}）：变体 A seed 42 对基线 seed 42 的 4096 世界 MP-SR")
    for ep, a, b, d in rows:
        print(f"  ep{ep:<3} 变体 {a:.4f}  基线 {b:.4f}  差 {d:+.4f}")
    mean = statistics.mean(r[3] for r in rows)
    best = max(r[3] for r in rows)
    stop = mean <= 0 and best < MID_STOP_MAX_GAIN
    print(f"  平均差 {mean:+.4f}，最大差 {best:+.4f}；规则：平均 ≤ 0 且没有一条 ≥ +{MID_STOP_MAX_GAIN:.2f} 就停")
    print("MIDCHECK " + ("STOP" if stop else "CONTINUE"))
    return 1 if stop else 0


def verdict(spec: dict, root: Path) -> int:
    steps, worlds = int(spec["steps"]), int(spec["worlds"])
    eps, can = spec["episodes"], spec["canary"]
    variant_runs = [e[k] for e in eps for k in ("var42", "var43")] + [can["var42"]]
    bc_runs = [e[k] for e in eps for k in ("base43", "var43")]
    if not any((root / "runs" / "flash_chord" / r).exists() for r in bc_runs):
        print(f"第 3 步配方试验判定（档位 {spec['tier']}，{spec['machine']}）：BC 段的 {len(bc_runs)} 条一条都没开跑")
        print("TRIAL_VERDICT STOPPED_AT_MIDCHECK（结论以中途检查为准，见段间门写在 BC 段训练日志里的输出）")
        return 4
    sick = {}
    for r in variant_runs:
        problems = health(root, r)
        if problems:
            sick[r] = problems
    res, missing, notes = {}, [], []
    plan = [(e["ep"], e["base42"], "export") for e in eps] + [(can["ep"], can["base42"], "export")]
    plan += [(e["ep"], e[k], "eval") for e in eps for k in ("var42", "var43", "base43")] + [(can["ep"], can["var42"], "eval")]
    for ep, run, kind in plan:
        try:
            res[run] = load(root, run, ep, steps, worlds, kind)
        except Bad as err:
            missing.append(str(err))
    print(f"第 3 步配方试验判定（档位 {spec['tier']}，{spec['machine']}，8c778f9 评测器、{worlds} 世界）")
    for r, p in sick.items():
        print(f"  S5 不健康：{r}：{'；'.join(p)}")
    for msg in missing:
        print(f"  读不到：{msg}")
    var_missing = [r for r in variant_runs if r not in res]
    base42_missing = [r for r in [e["base42"] for e in eps] + [can["base42"]] if r not in res]
    if base42_missing or (var_missing and not sick):
        print("TRIAL_VERDICT INCOMPLETE（基线 seed 42 的重评或变体的评测缺失，且不是训练原因）")
        return 3

    def arm(e, which):
        if which == "A":
            vals = [res.get(e["var42"]), res.get(e["var43"])]
        else:
            vals = [res[e["base42"]], res.get(e["base43"])]
            if vals[1] is None:
                notes.append(f"ep{e['ep']} 基线 seed 43 缺失，B 只用 seed 42")
        vals = [v for v in vals if v is not None]
        return {k: statistics.mean(v[k] for v in vals) for k in KEYS} if vals else None

    A = {e["ep"]: arm(e, "A") for e in eps}
    B = {e["ep"]: arm(e, "B") for e in eps}
    print(f"  {'ep':<5}{'MP_A':>8}{'MP_B':>8}{'差':>8}{'AUC_A':>8}{'AUC_B':>8}{'SP_A':>7}{'SP_B':>7}{'MPPE_A':>8}{'MPPE_B':>8}")
    for e in eps:
        a, b = A[e["ep"]], B[e["ep"]]
        if a is None:
            print(f"  ep{e['ep']:<3} 变体 A 没有可用的评测")
            continue
        print(f"  ep{e['ep']:<3}{a['mp']:>8.4f}{b['mp']:>8.4f}{a['mp'] - b['mp']:>+8.4f}{a['auc']:>8.4f}{b['auc']:>8.4f}"
              f"{a['sp']:>7.3f}{b['sp']:>7.3f}{a['mppe']:>8.2f}{b['mppe']:>8.2f}")
    ok = {}
    full = all(A[e["ep"]] is not None for e in eps)
    if full:
        mean = lambda d, k: statistics.mean(d[e["ep"]][k] for e in eps)
        gain = mean(A, "mp") - mean(B, "mp")
        ok["S1"] = (gain >= S1_MIN, f"平均 MP_A − 平均 MP_B = {gain:+.4f}（要 ≥ +{S1_MIN}）")
        n2 = sum(A[e["ep"]]["mp"] >= B[e["ep"]]["mp"] + S2_MARGIN for e in eps)
        ok["S2"] = (n2 >= S2_MIN_COUNT, f"MP_A ≥ MP_B + {S2_MARGIN} 的 episode {n2} 条（要 ≥ {S2_MIN_COUNT}）")
        drops = [f"ep{e['ep']} MP {A[e['ep']]['mp'] - B[e['ep']]['mp']:+.3f}" for e in eps if A[e["ep"]]["mp"] < B[e["ep"]]["mp"] - S3_MP_DROP]
        drops += [f"ep{e['ep']} ADD-AUC {A[e['ep']]['auc'] - B[e['ep']]['auc']:+.3f}" for e in eps if A[e["ep"]]["auc"] < B[e["ep"]]["auc"] - S3_AUC_DROP]
        ok["S3"] = (not drops, "没有大退步" if not drops else "大退步：" + "，".join(drops))
        s4 = [mean(A, "auc") >= mean(B, "auc") - S4_AUC, mean(A, "sp") >= mean(B, "sp") - S4_SP, mean(A, "mppe") <= mean(B, "mppe") + S4_MPPE]
        ok["S4"] = (all(s4), f"ADD-AUC {mean(A, 'auc'):.4f} 对 {mean(B, 'auc'):.4f}；SP-SR {mean(A, 'sp'):.4f} 对 {mean(B, 'sp'):.4f}；"
                             f"MPPE {mean(A, 'mppe'):.2f} 对 {mean(B, 'mppe'):.2f} cm")
    else:
        for s in ("S1", "S2", "S3", "S4"):
            ok[s] = (False, "变体 A 有 episode 没有可用的评测，按不通过算")
    ok["S5"] = (not sick and not var_missing, "变体 A 的 run 全部健康、评完" if not sick and not var_missing
                else f"不健康 {sorted(sick)}，缺评测 {var_missing}")
    ca, cb = res.get(can["var42"]), res[can["base42"]]
    ok["S6"] = (ca is not None and ca["mp"] >= cb["mp"] - S6_DROP,
                f"对照 ep{can['ep']} 变体 {ca['mp'] if ca else float('nan'):.4f}，基线 {cb['mp']:.4f}（要 ≥ 基线 − {S6_DROP}）")
    for s, (passed, why) in ok.items():
        print(f"  {s} {'过' if passed else '不过'}：{why}")
    for n in notes:
        print(f"  注：{n}")
    # 只报
    if full:
        b43 = [e for e in eps if e["base43"] in res]
        if b43:
            sens = statistics.mean(A[e["ep"]]["mp"] - res[e["base43"]]["mp"] for e in b43)
            print(f"  只报 敏感性：变体两种子均值 − 基线 seed 43 的 MP-SR 平均 {sens:+.4f}（{len(b43)} 条）")
            print("  只报 基线两个种子之差（seed 43 − seed 42）MP-SR：" + "，".join(
                f"ep{e['ep']} {res[e['base43']]['mp'] - res[e['base42']]['mp']:+.3f}" for e in b43))
    print("  只报 chord_sr（与终止阈值有关，变体与基线不可比）：" + "，".join(
        f"{r} {v['chord']:.3f}" for r, v in res.items() if isinstance(v.get("chord"), (int, float))))
    nf = {r: v["nonfinite"] for r, v in res.items() if v["nonfinite"]}
    print(f"  只报 评测里的非有限世界：{nf if nf else '无'}")
    failed = [s for s, (p, _) in ok.items() if not p]
    print("TRIAL_VERDICT " + ("PASS" if not failed else "FAIL（" + "、".join(failed) + " 不过）"))
    return 0 if not failed else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("spec")
    ap.add_argument("--midcheck", action="store_true")
    ap.add_argument("--root", default=str(ROOT))
    args = ap.parse_args(argv)
    root = Path(args.root)
    spec = json.loads((Path(args.spec) if Path(args.spec).is_absolute() else root / args.spec).read_text())
    guard(spec, root)
    return midcheck(spec, root) if args.midcheck else verdict(spec, root)


if __name__ == "__main__":
    sys.exit(main())
