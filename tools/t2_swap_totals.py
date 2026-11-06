#!/usr/bin/env python3
"""Tier 2 清单换掉几集后的合计：导出那 1 局（对自己的 v5.4 参考）与 4096 世界评测并列，并按 agent-doc/plan-t2mx-2026-10-19.md 的 Q3 逐项列出
（Mac 上、仓库根执行，只用标准库，只读记录）。只用 Tier 2 的数据。
原清单各集：导出取 <导出目录>/episode_<集>.json，核对 checkpoint 是清单里那条 run 的那一步。换上的 run：导出取导出链的记录
results/[ml40/]submission/exports/cand_t2_*_<run>_s<步数>.txt 的 summary_json，同名 _check.txt 要有 EXPORT_OK。
4096 世界都取评测记录 results/[ml40/]day2/eval_<run>_s<步数>.txt 的 summary_json，核对 checkpoint、步数与 world_count。
Q3 以给定的清单为基准（现在是 b1014）。MP 分支：导出 MP 成功 ≥ 21，新增的成功要导出位置误差 ≤ 0.025 m 才算；不够就要 ≥ 22；
4096 世界 MP-SR 合计多出 ≥ 3；导出 SP 成功 ≥ 22。AUC 分支：导出 AUC 平均 ≥ 0.530，4096 世界 AUC 平均 ≥ 0.515，导出 MP 成功 ≥ 19，SP 成功 ≥ 22。
另按 agent-doc/plan-t2st-2026-10-19.md 第 2 节 S3 列 MPPE 分支：导出 MPPE 平均 ≤ 12.97，4096 世界 MPPE 平均 ≤ 16.0，导出 SP、MP 成功与两种 AUC 平均都不低于
给定的清单，换上的每一集导出 MPPE ≤ 它 4096 世界 MPPE + 10 cm。第二次提交的门槛（要比上一次交的强）不在这里算。
--write-tsv <新清单>：在给定清单上只改换掉的那几行（run、步数与机器，机器取 run 名最后一段），其余行逐字照抄；目标已存在就拒绝。
--tier 3（agent-doc/plan-t3auc1-2026-11-03.md 第 2 节）：读清单的 Tier 3 行，换上的 run 找 cand_t3_* 的导出记录；4096 世界记录另在 results/[ml40/]t3static/ 下找
（清单里 ep9、11、28 的站立保底 run）；只打合计与两种 AUC 平均的变化，不打 Tier 2 的 Q3、S3 两行。不给时是 Tier 2，一切照旧。
用法：python3 tools/t2_swap_totals.py <清单.tsv> <导出目录> [ep<N>=<run>[@步数] …] [--tier 2|3] [--write-tsv <新清单>]   最后一行 T2_SWAP_TOTALS_DONE
"""

import glob
import json
import sys
from pathlib import Path

TIER = "2"
KEYS = ("spider_sr_uncentered", "maniptrans_sr", "add_auc", "mppe_cm", "object_position_error_m")
POS_NEW = 0.025


def summary_of(path: Path) -> dict:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("summary_json "):
            return json.loads(line[len("summary_json "):])
    sys.exit(f"{path} 里没有 summary_json 行")


def check_ckpt(j: dict, run: str, step: str, src) -> None:
    ck = Path(str(j.get("checkpoint", "")))
    if ck.parent.name != run or ck.name != f"policy_{step}.safetensors":
        sys.exit(f"{src} 的 checkpoint 是 {ck}，不是 {run} 第 {step} 步")


def eval_4096(run: str, step: str) -> dict:
    for d in ("results/day2", "results/ml40/day2", "results/l40n/day2") + (("results/t3static", "results/ml40/t3static") if TIER == "3" else ()):
        p = Path(d) / f"eval_{run}_s{step}.txt"
        if p.is_file():
            j = summary_of(p)
            check_ckpt(j, run, step, p)
            if j.get("world_count") != 4096 or str(j.get("environment_steps")) != step:
                sys.exit(f"{p}：world_count {j.get('world_count')}，步数 {j.get('environment_steps')}")
            m = j if all(k in j for k in KEYS) else j.get("metrics", {})   # t3static 的站立保底评测把指标放在 metrics 下
            return {k: float(m[k]) for k in KEYS}
    sys.exit(f"找不到 {run} 第 {step} 步的 4096 世界评测记录")


def base_export(exp: Path, ep: int, run: str, step: str) -> dict:
    p = exp / f"episode_{ep:06d}.json"
    j = json.loads(p.read_text())
    check_ckpt(j, run, step, p)
    return {k: float(j["metrics"][k]) for k in KEYS}


def swap_export(run: str, step: str) -> dict:
    hits = sorted(h for d in ("results/submission/exports", "results/ml40/submission/exports", "results/l40n/submission/exports")
                  for h in glob.glob(f"{d}/cand_t{TIER}_*_{run}_s{step}.txt"))
    if len(hits) != 1:
        sys.exit(f"{run} 第 {step} 步的导出记录有 {len(hits)} 份：{hits}")
    p = Path(hits[0])
    chk = p.with_name(p.name[:-len(".txt")] + "_check.txt")
    if not chk.is_file() or "EXPORT_OK" not in chk.read_text(encoding="utf-8"):
        sys.exit(f"{chk} 不存在或没有 EXPORT_OK")
    j = summary_of(p)
    check_ckpt(j, run, step, p)
    return {k: float(j[k]) for k in KEYS}


def fmt_x(x: dict) -> str:
    return (f"{x['spider_sr_uncentered']:.0f} / {x['maniptrans_sr']:.0f} / {x['add_auc']:.3f} / {x['mppe_cm']:.2f} / "
            f"{100 * x['object_position_error_m']:.2f}")


def fmt_w(w: dict) -> str:
    return f"{w['spider_sr_uncentered']:.3f} / {w['maniptrans_sr']:.3f} / {w['add_auc']:.3f} / {w['mppe_cm']:.2f}"


def total(sel: dict, world: str, key: str) -> float:
    return sum(v[world][key] for v in sel.values())


def ok(flag: bool) -> str:
    return "过" if flag else "不过"


def main():
    global TIER
    args = sys.argv[1:]
    out_tsv = None
    if "--tier" in args:
        i = args.index("--tier")
        if i + 1 >= len(args) or args[i + 1] not in ("2", "3"):
            sys.exit("--tier 后面只能是 2 或 3")
        TIER = args[i + 1]
        del args[i:i + 2]
    if "--write-tsv" in args:
        i = args.index("--write-tsv")
        if i + 1 >= len(args):
            sys.exit("--write-tsv 后面要给新清单的路径")
        out_tsv = Path(args[i + 1])
        del args[i:i + 2]
        if out_tsv.exists():
            sys.exit(f"{out_tsv} 已存在")
    if len(args) < 2:
        sys.exit(__doc__)
    sel_path, exp = Path(args[0]), Path(args[1])
    rows = [l.split("\t") for l in sel_path.read_text().splitlines() if l.startswith(f"{TIER}\t")]
    if len(rows) != 30:
        sys.exit(f"清单里 Tier {TIER} 有 {len(rows)} 行")
    base = {}
    for _, ep, run, step, *_ in rows:
        e = int(ep)
        base[e] = {"run": run, "step": step, "x": base_export(exp, e, run, step), "w": eval_4096(run, step)}
    new = dict(base)
    for a in args[2:]:
        k, _, v = a.partition("=")
        if not k.startswith("ep") or not k[2:].isdigit() or not v:
            sys.exit(f"参数 {a} 不是 ep<N>=<run>[@步数]")
        e = int(k[2:])
        run, _, step = v.partition("@")
        step = step or "349999104"
        if e not in base or new[e] is not base[e]:
            sys.exit(f"ep{e} 不在清单里或换了两次")
        new[e] = {"run": run, "step": step, "x": swap_export(run, step), "w": eval_4096(run, step)}

    print("集\t原 run → 换上的 run\t导出 SP / MP / AUC / MPPE cm / 位置 cm（原 → 换后）\t4096 世界 SP-SR / MP-SR / AUC / MPPE cm（原 → 换后）")
    strict_mp = 0
    for e in sorted(new):
        b, n = base[e], new[e]
        if n is b:
            strict_mp += round(b["x"]["maniptrans_sr"])
            continue
        gained = b["x"]["maniptrans_sr"] < 0.5 <= n["x"]["maniptrans_sr"]
        if n["x"]["maniptrans_sr"] >= 0.5 and (not gained or n["x"]["object_position_error_m"] <= POS_NEW):
            strict_mp += 1
        print(f"ep{e}\t{b['run']} → {n['run']}\t{fmt_x(b['x'])} → {fmt_x(n['x'])}\t{fmt_w(b['w'])} → {fmt_w(n['w'])}")

    sp0, sp1 = round(total(base, "x", "spider_sr_uncentered")), round(total(new, "x", "spider_sr_uncentered"))
    mp0, mp1 = round(total(base, "x", "maniptrans_sr")), round(total(new, "x", "maniptrans_sr"))
    xa0, xa1 = total(base, "x", "add_auc") / 30, total(new, "x", "add_auc") / 30
    xm0, xm1 = total(base, "x", "mppe_cm") / 30, total(new, "x", "mppe_cm") / 30
    wsp0, wsp1 = total(base, "w", "spider_sr_uncentered"), total(new, "w", "spider_sr_uncentered")
    wmp0, wmp1 = total(base, "w", "maniptrans_sr"), total(new, "w", "maniptrans_sr")
    wa0, wa1 = total(base, "w", "add_auc") / 30, total(new, "w", "add_auc") / 30
    wm0, wm1 = total(base, "w", "mppe_cm") / 30, total(new, "w", "mppe_cm") / 30
    print(f"导出（对自己参考）：SP 成功 {sp0} → {sp1} / 30，MP 成功 {mp0} → {mp1} / 30（新增的成功只算位置 ≤ {POS_NEW} m 的：{strict_mp}），"
          f"AUC 平均 {xa0:.5f} → {xa1:.5f}，MPPE 平均 {xm0:.3f} → {xm1:.3f} cm")
    print(f"4096 世界：SP-SR 合计 {wsp0:.3f} → {wsp1:.3f}，MP-SR 合计 {wmp0:.3f} → {wmp1:.3f}（多出 {wmp1 - wmp0:.3f}），"
          f"AUC 平均 {wa0:.5f} → {wa1:.5f}，MPPE 平均 {wm0:.3f} → {wm1:.3f} cm")
    if TIER == "3":
        print(f"Tier 3 两种 AUC 平均的变化：导出 {xa1 - xa0:+.5f}，4096 世界 {wa1 - wa0:+.5f}，小的那个 {min(xa1 - xa0, wa1 - wa0):+.5f}")
        write_tsv(out_tsv, sel_path, new, base)
        print("T2_SWAP_TOTALS_DONE")
        return
    mp_count = strict_mp >= 21 or mp1 >= 22
    mp_branch = mp_count and wmp1 - wmp0 >= 3 and sp1 >= 22
    print(f"Q3 MP 分支：导出 MP（严格计 {strict_mp} ≥ 21，或全计 {mp1} ≥ 22）{ok(mp_count)}；4096 世界 MP-SR 多出 {wmp1 - wmp0:.3f} ≥ 3 "
          f"{ok(wmp1 - wmp0 >= 3)}；导出 SP {sp1} ≥ 22 {ok(sp1 >= 22)}；分支{ok(mp_branch)}")
    auc_branch = xa1 >= 0.530 and wa1 >= 0.515 and mp1 >= 19 and sp1 >= 22
    print(f"Q3 AUC 分支：导出 AUC 平均 {xa1:.5f} ≥ 0.530 {ok(xa1 >= 0.530)}；4096 世界 AUC 平均 {wa1:.5f} ≥ 0.515 {ok(wa1 >= 0.515)}；"
          f"导出 MP {mp1} ≥ 19 {ok(mp1 >= 19)}；导出 SP {sp1} ≥ 22 {ok(sp1 >= 22)}；分支{ok(auc_branch)}")
    print(f"Q3：{'满足（' + ('MP' if mp_branch else 'AUC') + ' 分支）' if mp_branch or auc_branch else '两条分支都不满足，不交'}")
    per_ok = all(n["x"]["mppe_cm"] <= n["w"]["mppe_cm"] + 10 for e, n in new.items() if n is not base[e])
    s3 = [(f"导出 MPPE 平均 {xm1:.3f} ≤ 12.97", xm1 <= 12.97), (f"4096 世界 MPPE 平均 {wm1:.3f} ≤ 16.0", wm1 <= 16.0),
          (f"导出 SP {sp1} ≥ {sp0}", sp1 >= sp0), (f"导出 MP {mp1} ≥ {mp0}", mp1 >= mp0),
          (f"导出 AUC 平均 {xa1:.5f} ≥ {xa0:.5f}", xa1 >= xa0), (f"4096 世界 AUC 平均 {wa1:.5f} ≥ {wa0:.5f}", wa1 >= wa0),
          ("换上的每一集导出 MPPE ≤ 4096 世界 MPPE + 10", per_ok)]
    print("S3 MPPE 分支（plan-t2st）：" + "；".join(f"{k} {ok(v)}" for k, v in s3) + f"；分支{ok(all(v for _, v in s3))}")
    write_tsv(out_tsv, sel_path, new, base)
    print("T2_SWAP_TOTALS_DONE")


def write_tsv(out_tsv, sel_path, new, base):
    if out_tsv is not None:
        lines = []
        for line in sel_path.read_text().splitlines():
            c = line.split("\t")
            if c[0] == TIER and new[int(c[1])] is not base[int(c[1])]:
                n = new[int(c[1])]
                line = "\t".join([TIER, c[1], n["run"], n["step"], n["run"].rsplit("-", 1)[1]])
            lines.append(line)
        out_tsv.write_text("\n".join(lines) + "\n")
        print(f"写 {out_tsv}（{len(lines)} 行，改了 {sum(1 for e in new if new[e] is not base[e])} 行）")


if __name__ == "__main__":
    main()
