#!/usr/bin/env python3
"""Tier 2 跟丢的集：每条候选导出的那 1 局对照自己参考的 MPPE 与 ADD-AUC，以及按“这几集改选 MPPE 低的那条”换掉后合计能少多少（只用 Tier 2 的数据）。

计划 agent-doc/plan-first-place-2026-10-14.md 第 1.6 节：榜上 Tier 2 MPPE 与 Xueyi Wang (FF) 差 1.198 cm，30 集真值合计要减 35.9 cm 以上。
读已拉回 Mac 的导出记录（fc_export.sh 的输出：results/submission/exports/、results/ml40/submission/exports/ 下 *_<run>_s<步数>.txt，
不含 _check），取 “== …episode_XXXXXX.json” 之后的 add_auc、mppe_cm、spider_sr_uncentered、maniptrans_sr；记录的 world_count 不是 1 就拒绝；
同一条 run 有几份记录时核对几份的数一致。一集有几条合格的替代时只换 MPPE 最低的那一条。
现在交的 run 取 results/select_t2/select_b1013.txt 的 “→” 行。不读 Tier 1 的任何东西。
用法：python3 tools/t2_lost_swap.py --select results/select_t2/select_b1013.txt --episodes 4 25 28 9 1 17 2 11 [--step 349999104]
      [--current-tsv stages/evaluate/exports/t2_select_b1013.tsv --write-tsv stages/evaluate/exports/t2_select_b1014.tsv]
--write-tsv：在现在交的选择清单上只改标“换”的那几行（run 与机器，机器取 run 名最后一段），其余行逐字照抄；目标已存在就拒绝。
  核对：现在交的清单里这几集的 run 必须与 select 记录的 “→” 行相同，清单恰好 30 行、档位 2。
"""

import argparse
import re
from pathlib import Path

DIRS = [Path("results/submission/exports"), Path("results/ml40/submission/exports")]


def exported(run: str, step: str):
    vals = []
    for d in DIRS:
        for f in sorted(d.glob(f"*_{run}_s{step}.txt")):
            t = f.read_text(encoding="utf-8", errors="replace")
            if "== " not in t:
                continue
            body = t.split("\n== ", 1)[1]
            auc = re.search(r"^\s*add_auc\s+([0-9.]+)\s*$", body, re.M)
            mppe = re.search(r"^\s*mppe_cm\s+([0-9.]+)\s*$", body, re.M)
            sp = re.search(r"^\s*spider_sr_uncentered\s+([0-9.]+)\s*$", body, re.M)
            mp = re.search(r"^\s*maniptrans_sr\s+([0-9.]+)\s*$", body, re.M)
            wc = re.search(r"world_count (\d+)", body)
            if not wc or wc.group(1) != "1":
                raise SystemExit(f"t2_lost_swap: {f} 不是 1 个世界的导出（world_count {wc.group(1) if wc else '缺'}），不能当导出那 1 局读")
            if auc and mppe and sp and mp:
                vals.append((f.stat().st_mtime, str(f), float(auc.group(1)), float(mppe.group(1)), float(sp.group(1)), float(mp.group(1))))
    if not vals:
        return None
    if len({v[2:] for v in vals}) != 1:
        raise SystemExit(f"t2_lost_swap: {run} 的几份导出记录数值不一致：{vals}")
    return max(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", required=True)
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--step", default="349999104")
    ap.add_argument("--current-tsv")
    ap.add_argument("--write-tsv")
    a = ap.parse_args()
    if bool(a.current_tsv) != bool(a.write_tsv):
        raise SystemExit("t2_lost_swap: --current-tsv 与 --write-tsv 要一起给")
    swaps = {}
    cands, chosen = {}, {}
    for line in Path(a.select).read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"^\s+ep(\d+)\s+(→)?\s*(t2-\S+?)：", line)
        if m and int(m.group(1)) in a.episodes:
            ep = int(m.group(1))
            cands.setdefault(ep, []).append(m.group(3))
            if m.group(2):
                chosen[ep] = m.group(3)
    total = 0.0
    print("集\t现在交的 run\t导出那 1 局 SP / MP / AUC / MPPE\t另一条 run\t导出那 1 局 SP / MP / AUC / MPPE\t换了 MPPE 合计变化（cm）\tAUC 合计变化")
    for ep in a.episodes:
        if ep not in chosen or len(cands.get(ep, [])) < 2:
            print(f"{ep}\t（候选不足两条：{cands.get(ep)}）")
            continue
        cur = exported(chosen[ep], a.step)
        best = None   # 合格的替代里 MPPE 最低的一条，一集只换一次
        for other in cands[ep]:
            if other == chosen[ep]:
                continue
            alt = exported(other, a.step)
            if cur is None or alt is None:
                print(f"{ep}\t{chosen[ep]}\t{'缺记录' if cur is None else ''}\t{other}\t{'缺记录' if alt is None else ''}")
                continue
            d_mppe, d_auc = alt[3] - cur[3], alt[2] - cur[2]
            take = d_mppe < 0 and alt[4] >= cur[4] and alt[5] >= cur[5]
            if take and (best is None or d_mppe < best[0]):
                best = (d_mppe, other)
            fmt = lambda v: f"{v[4]:.0f} / {v[5]:.0f} / {v[2]:.3f} / {v[3]:.2f}"  # noqa: E731
            print(f"{ep}\t{chosen[ep]}\t{fmt(cur)}\t{other}\t{fmt(alt)}\t{d_mppe:+.2f}{'（合格）' if take else ''}\t{d_auc:+.3f}")
        if best:
            total += best[0]
            swaps[ep] = best[1]
            print(f"  ep{ep} 换成 {best[1]}（合格的替代里 MPPE 最低）")
    print(f"只换 MPPE 更低、SP 与 MP 不变差的集：导出那 1 局对照自己参考的 MPPE 合计变化 {total:+.2f} cm（目标：真值合计减 35.9 cm 以上；这里是代理量，不是真值）")
    if a.write_tsv:
        out = Path(a.write_tsv)
        if out.exists():
            raise SystemExit(f"t2_lost_swap: {out} 已存在，不覆盖")
        rows = [line.split("\t") for line in Path(a.current_tsv).read_text().splitlines() if line.strip()]
        if len(rows) != 30 or any(r[0] != "2" or len(r) != 5 for r in rows) or sorted(int(r[1]) for r in rows) != list(range(30)):
            raise SystemExit(f"t2_lost_swap: {a.current_tsv} 不是 30 行、档位 2、5 列、集号 0–29 的选择清单")
        new = []
        for r in rows:
            ep = int(r[1])
            if ep in chosen and r[2] != chosen[ep]:
                raise SystemExit(f"t2_lost_swap: 清单里 ep{ep} 是 {r[2]}，select 记录里现在交的是 {chosen[ep]}")
            if ep in swaps:
                r = [r[0], r[1], swaps[ep], r[3], swaps[ep].rsplit("-", 1)[1]]
                print(f"写：ep{ep} {chosen[ep]} → {swaps[ep]}")
            new.append("\t".join(r))
        out.write_text("\n".join(new) + "\n")
        print(f"写 {out}（{len(new)} 行，改了 {len(swaps)} 行）")
    print("T2_LOST_SWAP_DONE")


if __name__ == "__main__":
    main()
