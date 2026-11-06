#!/usr/bin/env python3
"""某几集试过的全部 run：4096 世界评测的成功率、误差与失败原因，按集并列（Mac 上跑，只读已落盘的结果，只用标准库）。

找 results/day2/ 与 results/ml40/day2/ 下的 eval_t<档>-ep<集>-*_s<步数>.txt（不含 .failed），同一个 run 只取步数最大的那份，
用 stages/evaluate/eval_table.py 的 row() 解析。给了 --select（select_t1.py 的记录），再把其中“  ep<集>  <run>: …”行里
1 个世界的四项（交上去的那一局）并到同名 run 后面；CHOSEN 行选中的 run 标“选中”。
档位隔离：Tier 1 与 Tier 2 的行只是并列打印，不拿一档的数去算另一档的任何东西。
用法：python3 tools/episode_runs_table.py --eps 25,4,28,11,9,1,2 --tiers 1,2 [--select <select 记录>]
计划 agent-doc/plan-first-place-2026-10-14.md 第 1 节。
"""
import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stages" / "evaluate"))
from eval_table import row  # noqa: E402

FILE = re.compile(r"^eval_(t\d-ep\d+-.+)_s(\d+)\.txt$")
CAND = re.compile(r"^  ep(\d+)\s+(\S+): add_auc=(\S+) spider_sr=(\S+) maniptrans_sr=(\S+) mppe_cm=(\S+)")
CHOSEN = re.compile(r"^CHOSEN ep(\d+): (\S+)（")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", required=True)
    ap.add_argument("--tiers", default="1,2")
    ap.add_argument("--select")
    a = ap.parse_args()
    eps = [int(x) for x in a.eps.split(",")]
    tiers = [int(x) for x in a.tiers.split(",")]
    one, chosen = {}, set()
    if a.select:
        for line in open(a.select, encoding="utf-8", errors="replace"):
            m = CAND.match(line)
            if m:
                one[m.group(2)] = m.groups()[2:]
            m = CHOSEN.match(line)
            if m:
                chosen.add(m.group(2))
    head = ["集", "档", "run", "步数", "MP-SR", "SP-SR", "ADD-AUC", "MPPE_cm", "失败_骨盆", "失败_物体", "失败_手掌",
            "1世界_AUC", "1世界_SP", "1世界_MP", "1世界_MPPE", "选中"]
    print("\t".join(head))
    for ep in eps:
        for t in tiers:
            best = {}
            for d in (ROOT / "results" / "day2", ROOT / "results" / "ml40" / "day2"):
                for p in d.glob(f"eval_t{t}-ep{ep}-*_s*.txt"):
                    m = FILE.match(p.name)
                    if not m:
                        continue
                    run, step = m.group(1), int(m.group(2))
                    if run not in best or step > best[run][0]:
                        best[run] = (step, p)
            for run in sorted(best):
                r = row(best[run][1])
                extra = list(one.get(run, ("—",) * 4))
                print("\t".join([str(ep), str(t), r[0], r[1], r[4], r[5], r[6], r[7], r[11], r[12], r[13], *extra,
                                 "选中" if run in chosen else ""]))
    print("EPISODE_RUNS_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
