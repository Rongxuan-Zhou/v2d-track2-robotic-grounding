#!/usr/bin/env python3
"""Tier 2 交上去的每集 run 在 4096 世界评测里对照自己参考（降噪后的 Tier 2 轨迹）的 MPPE 与成功率，按 MPPE 从大到小排
（Mac 上跑，只读已落盘的结果，只用标准库）。

档位隔离：只读 Tier 2 的选择清单与 Tier 2 run 的评测，不读 Tier 1 的任何数据。这里的 MPPE 是“跟自己的参考差多少”，不是榜上对照
干净真值的 MPPE；跟丢的集（跟踪误差几十厘米）在真值上也一定差得多，所以只拿它找哪些集拉高了总和，不拿它估真值上的分数。
用法：python3 tools/t2_mppe_proxy.py <选择清单 tsv，列：档 集 run 步数 机器> [--board-mppe 16.54028] [--target 15.34253]
计划 agent-doc/plan-first-place-2026-10-14.md 第 1 节。
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stages" / "evaluate"))
from eval_table import row  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("select_tsv")
    ap.add_argument("--board-mppe", type=float)
    ap.add_argument("--target", type=float)
    a = ap.parse_args()
    rows = []
    for line in open(a.select_tsv, encoding="utf-8"):
        f = line.rstrip("\n").split("\t")
        if not line.strip() or line.startswith("#") or f[0] != "2":
            if f and f[0] not in ("2", "") and not line.startswith("#"):
                raise SystemExit(f"不是 Tier 2 的行：{line.strip()}")
            continue
        ep, run, step = int(f[1]), f[2], f[3]
        p = None
        for d in (ROOT / "results" / "ml40" / "day2", ROOT / "results" / "day2"):
            q = d / f"eval_{run}_s{step}.txt"
            if q.exists():
                p = q
                break
        if p is None:
            raise SystemExit(f"找不到 {run} 的 4096 世界评测")
        r = row(p)
        rows.append((ep, run, float(r[4]), float(r[5]), float(r[6]), float(r[7]), r[11], r[12], r[13]))
    if len(rows) != 30:
        raise SystemExit(f"选择清单里 Tier 2 的行有 {len(rows)} 条，要 30 条")
    print("集\trun\tMP-SR\tSP-SR\tADD-AUC\tMPPE_cm（对照自己的参考）\t失败_骨盆\t失败_物体\t失败_手掌")
    for r in sorted(rows, key=lambda x: -x[5]):
        print(f"{r[0]}\t{r[1]}\t{r[2]:.3f}\t{r[3]:.3f}\t{r[4]:.3f}\t{r[5]:.2f}\t{r[6]}\t{r[7]}\t{r[8]}")
    tot = sum(r[5] for r in rows)
    big = [r for r in rows if r[5] > 15.0]
    print(f"\n30 集对照自己参考的 MPPE 平均 {tot / 30:.3f} cm、合计 {tot:.1f} cm；超过 15 cm 的 {len(big)} 集合计 "
          f"{sum(r[5] for r in big):.1f} cm（{sum(r[5] for r in big) / tot:.1%}）：" + "、".join(f"ep{r[0]} {r[5]:.1f}" for r in sorted(big, key=lambda x: -x[5])))
    if a.board_mppe is not None and a.target is not None:
        need = (a.board_mppe - a.target) * 30
        print(f"榜上 MPPE {a.board_mppe:.5f}，目标 {a.target:.5f}：30 集真值 MPPE 合计要减 {need:.1f} cm 以上（每集平均 {need / 30:.3f} cm）")
    print("T2_MPPE_PROXY_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
