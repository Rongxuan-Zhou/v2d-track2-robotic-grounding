"""失败世界的越线起点与参考接触（容器里跑，要 numpy、pyarrow）：读 fc_eval_dump.sh 的 npz，调 fall_order.run_one（定义不改），
每条 run 输出每帧的越线计数与参考的接触，供逐帧对照。计划 agent-doc/plan-frame-onset-2026-10-14.md 第 3 节第 0 步、第 4 节。

- 复现门：--repro fixed 用 fall_order 的 REPRO_MP（0.01，Tier 1 丙，登记的不改）；--repro noise 用 max(0.01, 3·√(2p(1−p)/N))，
  p 取原评测的 MP-SR，N 取新评的世界数（Tier 3；两次评测是两次抽样）。
- 参考：policy 内嵌配置的 task.parquet（与评测器重建环境用的是同一份，取法同 diag_rollout.py），读 hand_sides 与 hand_contact_active（2×T）。
  “接触中” = 任一侧为真。
- 门（不过就记“门不过”，JSON 里 gate_ok 为 false，不出计数）：fall_order.run_one 列出的问题；新评 JSON 的 start_frame 不是 0 或
  reference_tracking.reference_end_fraction 不是 1.0（帧号 = 步号的前提）；参考帧数不等于评测的 step_count；policy 或参考读不了；
  G4：这一次给的 run 之间骨盆、手掌、物体位置阈值与 active_after_step 不同，或同一变体（名字带不带 -r3a- 或 -r3o-，fall_order.is_variant）的物体朝向阈值不同：与严格多数不同的那条
  记门不过，没有严格多数时整组记门不过；门不过的删去计数。参考的 hand_contact_active 只许 0、1。坏文件只算这一条门不过；档位守卫拦下的读取让整次调用失败。
  复现门另打 z = 差 / √(2p(1−p)/N)。
- 起点：失败世界里 t_on = min(t_pel, t_obj)（存在的那个）；两者都没有的失败世界计入 no_cross。帧号 = 步号。
- 输出：<outdir>/onset_<run>.json（n_on、n_pel、n_obj 每帧计数，contact，门与汇总）与同名 .txt（第 4 节“只报”各项）。
- 档位（G6）：一次只收同一档的 run（名字前缀 t1-、t2- 或 t3-），混档拒绝；t2-、t3- 的读文件之前调 tier_guard.forbid_tier1()。
退出码：全部 run 处理完为 0（门不过的写进输出）；参数、混档或输出目录有问题为 2。结尾一行 FAIL_ONSET_DONE（门过 x、门不过 y）。
用法：python stages/evaluate/fail_onset.py <outdir> <run> [<run> ...] [--steps 349999104] [--worlds 4096] [--root .] [--repro fixed|noise]
"""

import argparse
import json
import math
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import fall_order as fo  # noqa: E402
from libs.v2d_infra import tier_guard  # noqa: E402

WIN = 15            # 起点最集中的窗口（帧）
COUNT_KEYS = ("n_on", "n_pel", "n_obj", "n_palm", "contact", "q_on", "q_pel", "q_obj", "classes", "pel_phase", "mode_window")
KS = (10, 25, 50)   # 三类的 k，主值 25


def repro_tol(mode: str, p: float, n: int) -> float:
    if mode == "fixed":
        return fo.REPRO_MP
    return max(0.01, 3 * math.sqrt(2 * p * (1 - p) / n))


def ref_parquet(root: Path, run: str, steps: int) -> Path:
    with open(root / "runs/flash_chord" / run / f"policy_{steps}.safetensors", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        cfg = json.loads(json.loads(f.read(n))["__metadata__"]["user.resolved_config_json"])
    p = str(cfg["task"]["parquet"]).strip("'\"")
    if p.startswith("/repo/") and str(root.resolve()) != "/repo":
        return root / p[len("/repo/"):]
    return Path(p) if p.startswith("/") else root / p


def ref_contact(path: Path):
    import pyarrow.parquet as pq

    t = pq.read_table(path, columns=["hand_sides", "hand_contact_active"]).to_pydict()
    sides = [str(s) for s in t["hand_sides"][0]]
    c = np.asarray(t["hand_contact_active"][0], dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != len(sides):
        raise ValueError(f"hand_contact_active 形状 {c.shape} 与 hand_sides {sides} 对不上")
    bad = sorted(set(np.unique(c).tolist()) - {0.0, 1.0})
    if bad:   # 只许 0、1：不论按什么门槛读，接触帧都一样（代码审查第 5 条）
        raise ValueError(f"hand_contact_active 有 0、1 以外的值：{bad[:5]}")
    return sides, c > 0.5


def quantiles(x: np.ndarray):
    """np.percentile（线性插值）的 10/25/50/75/90 分位，保留一位小数；t3_onset_evidence.py 用同一个定义。"""
    return [round(float(v), 1) for v in np.percentile(x, [10, 25, 50, 75, 90])] if len(x) else []


def onset(t_pel: np.ndarray, t_obj: np.ndarray) -> np.ndarray:
    both = (t_pel >= 0) & (t_obj >= 0)
    return np.where(both, np.minimum(t_pel, t_obj), np.where(t_pel >= 0, t_pel, t_obj))


def contact_phase(t: np.ndarray, contact_any: np.ndarray):
    """t 中每个步号相对接触的位置：before（第一次接触之前或从不接触）、during、gap（第一次接触之后的不接触段）。"""
    first = int(np.argmax(contact_any)) if contact_any.any() else len(contact_any)
    before = t < first
    during = ~before & contact_any[np.clip(t, 0, len(contact_any) - 1)]
    return first, before, during, ~before & ~during


def one(root: Path, run: str, steps: int, worlds: int, mode: str) -> dict:
    o = root / "runs/flash_chord" / run
    out = {"run": run, "steps": steps, "repro_mode": mode, "gate_ok": False, "problems": []}
    try:
        p_old = float(json.loads((o / f"eval_{steps}_w{worlds}.json").read_text())["metrics"]["maniptrans_sr"])
        n_new = int(json.loads((o / f"eval_dump/eval_{steps}_w{worlds}.json").read_text())["world_count"])
        tol = repro_tol(mode, p_old, n_new)
        r = fo.run_one(str(root), run, steps, worlds, fo.is_variant(run), repro_mp=tol)
    except tier_guard.Tier1AccessError:
        raise   # 档位守卫拦下的读取必须让整次调用失败，不能记成门不过
    except Exception as e:  # noqa: BLE001  坏文件（如截断的 npz）只算这一条门不过，其余 run 照常（代码审查第 14 条）
        out["problems"].append(f"读不了评测文件或缺键（{type(e).__name__}: {e}）")
        return out
    T = int(r["te"].shape[0])
    sd = math.sqrt(2 * p_old * (1 - p_old) / n_new) if 0 < p_old < 1 else 0.0
    out.update(tol=tol, p_old=p_old, d_mp=r["d_mp"], z=(r["d_mp"] / sd if sd else None), mp_new=r["mp_new"], mppe_new=r["mppe_new"],
               d_mppe=r["d_mppe"], mppe_tol=r["tol"], step_count=T, act=[int(a) for a in r["act"]], thr=[float(x) for x in r["thr"]], fps=r["fps"])
    out["problems"] += list(r["problems"])
    newj = json.loads((o / f"eval_dump/eval_{steps}_w{worlds}.json").read_text())
    sf, ref_end = newj.get("start_frame"), (newj.get("reference_tracking") or {}).get("reference_end_fraction")
    if sf != 0 or ref_end != 1.0:
        out["problems"].append(f"新评 JSON 的 start_frame {sf}、reference_end_fraction {ref_end}，要求 0 与 1.0")
    try:
        pq_path = ref_parquet(root, run, steps)
        sides, contact = ref_contact(pq_path)
        out["parquet"], out["sides"] = str(pq_path), sides
        if contact.shape[1] != T:
            out["problems"].append(f"参考帧数 {contact.shape[1]} 与评测 step_count {T} 不符")
    except tier_guard.Tier1AccessError:
        raise
    except Exception as e:  # noqa: BLE001
        out["problems"].append(f"读不了 policy 或参考（{type(e).__name__}: {e}）")
    if out["problems"]:
        return out
    failed, fallen = r["failed"], r["fallen"]
    t_pel, t_obj, t_palm = r["t_pel"], r["t_obj"], r["t_palm"]
    t_on = onset(t_pel, t_obj)
    has = failed & (t_on >= 0)
    contact_any = contact.any(axis=0)
    first, before, during, gap = contact_phase(t_pel[fallen], contact_any)
    out.update(
        gate_ok=True, n_valid=int(r["valid"].sum()), n_failed=int(failed.sum()), n_onset=int(has.sum()),
        n_no_cross=int((failed & (t_on < 0)).sum()), n_fallen=int(fallen.sum()),
        n_on=np.bincount(t_on[has], minlength=T).tolist(),
        n_pel=np.bincount(t_pel[failed & (t_pel >= 0)], minlength=T).tolist(),
        n_obj=np.bincount(t_obj[failed & (t_obj >= 0)], minlength=T).tolist(),
        n_palm=np.bincount(t_palm[failed & (t_palm >= 0)], minlength=T).tolist(),
        contact=contact.astype(int).tolist(), first_contact=first,
        q_on=quantiles(t_on[has]), q_pel=quantiles(t_pel[failed & (t_pel >= 0)]), q_obj=quantiles(t_obj[failed & (t_obj >= 0)]),
        classes={str(k): [float(x.mean()) if fallen.any() else 0.0 for x in fo.classify(t_obj[fallen], t_pel[fallen], k)] for k in KS},
        pel_phase={"before": float(before.mean()) if fallen.any() else 0.0, "during": float(during.mean()) if fallen.any() else 0.0,
                   "gap": float(gap.mean()) if fallen.any() else 0.0},
    )
    c = np.convolve(np.asarray(out["n_on"]), np.ones(WIN + 1, dtype=int))[: T]   # c[t] = [t − WIN, t] 内的起点数
    t_best = int(np.argmax(c)) if has.any() else -1
    out["mode_window"] = [max(0, t_best - WIN), t_best, int(c[t_best]) if has.any() else 0]
    return out


def g4(ds: list) -> None:
    """G4：骨盆、手掌、物体位置阈值与 active_after_step 全体相同，物体朝向阈值同一变体内相同；与多数不同的那条记门不过（只看读到了阈值的 run）。"""
    from collections import Counter

    have = [d for d in ds if "thr" in d]

    def mark(key, group):
        if not group:
            return
        (common, n), = Counter(key(d) for d in group).most_common(1)
        strict = n * 2 > len(group)   # 没有严格多数（如两条各一种）时整组都记门不过，不靠参数顺序定谁对（代码审查第 2 条）
        for d in group:
            if not strict or key(d) != common:
                d["problems"].append(f"G4：阈值或 active_after_step 与这一次的其他 run 不同或没有多数（{d['thr']}、{d['act']}）")
                d["gate_ok"] = False
                for k in COUNT_KEYS:   # 门不过不出计数（代码审查第 10 条）
                    d.pop(k, None)

    mark(lambda d: (tuple(d["thr"][:5]), tuple(d["act"])), have)
    for var in (True, False):
        mark(lambda d: d["thr"][5], [d for d in have if fo.is_variant(d["run"]) == var])


def summary(d: dict) -> str:
    head = f"== {d['run']}（复现门 {d['repro_mode']}"
    if "tol" in d:
        z = "—" if d.get("z") is None else f"{d['z']:+.2f}"
        head += f" {d['tol']:.4f}；原评 MP-SR {d['p_old']:.4f}，新评 {d['mp_new']:.4f}，差 {d['d_mp']:+.4f}（z {z}）；MPPE 差 {d['d_mppe']:+.2f} cm，只报"
    head += "）"
    if not d["gate_ok"]:
        return head + "\n  门不过，不出计数：" + "；".join(d["problems"])
    fps, s = d["fps"], d["step_count"]
    lines = [head,
             f"  step_count {s}，阈值 {d['thr']}，active_after_step {d['act']}，有效世界 {d['n_valid']}，失败 {d['n_failed']}（有起点 {d['n_onset']}、两者都没越线 {d['n_no_cross']}），"
             f"摔倒 {d['n_fallen']}",
             f"  起点 t_on 分位 10/25/50/75/90（帧）：{d['q_on']}；t_pel：{d['q_pel']}；t_obj：{d['q_obj']}（{fps:g} 帧每秒）"]
    w0, w1, n = d["mode_window"]
    if d["n_onset"]:
        lines.append(f"  起点最集中的 {WIN} 帧窗口：第 {w0}–{w1} 帧，{n} 个，占有起点的 {n / d['n_onset']:.1%}")
    for k in KS:
        of, pf, si = d["classes"][str(k)]
        lines.append(f"  摔倒世界的三类（k={k}）：物体先 {of:.1%}，骨盆先 {pf:.1%}，同时 {si:.1%}")
    ph = d["pel_phase"]
    lines.append(f"  参考第一次接触在第 {d['first_contact']} 帧（{'、'.join(d['sides'])} 任一侧）；摔倒世界的 t_pel：接触之前 {ph['before']:.1%}，"
                 f"接触中 {ph['during']:.1%}，接触之后的不接触段 {ph['gap']:.1%}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("outdir")
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--steps", type=int, default=349999104)
    ap.add_argument("--worlds", type=int, default=4096)
    ap.add_argument("--root", default=".")
    ap.add_argument("--repro", choices=("fixed", "noise"), required=True)
    a = ap.parse_args(argv)
    tiers = {r.split("-", 1)[0] for r in a.runs}
    if len(tiers) != 1 or not tiers <= {"t1", "t2", "t3"}:
        print(f"fail_onset: 一次只收同一档的 run（t1-、t2- 或 t3- 开头），这次是 {sorted(tiers)}", file=sys.stderr)
        return 2
    if tiers != {"t1"}:
        tier_guard.forbid_tier1()
    outdir = Path(a.outdir)
    try:
        outdir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"fail_onset: 建不了输出目录 {outdir}（{e}）", file=sys.stderr)
        return 2
    ds = [one(Path(a.root), run, a.steps, a.worlds, a.repro) for run in a.runs]
    g4(ds)
    good = bad = 0
    for d in ds:
        run = d["run"]
        (outdir / f"onset_{run}.json").write_text(json.dumps(d, ensure_ascii=False))
        text = summary(d)
        (outdir / f"onset_{run}.txt").write_text(text + "\n")
        print(text)
        good, bad = (good + 1, bad) if d["gate_ok"] else (good, bad + 1)
    print(f"FAIL_ONSET_DONE（门过 {good}、门不过 {bad}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
