"""Tier 1 逐条候选的“单项最优”组合：只用 select_t1.py 已算好的逐候选导出分数（--out-json），不重新打分、不打包。

Kaggle 上每个指标是一个单独的比赛，榜上留每队在这一项上最好的一次（`agent-doc/plan-first-place-2026-10-14.md` 第 268 行附近记的
Tier 2 AUC 榜）。这里回答：手上已有的候选，按某一项逐条挑最好的，这一项的合计最多能到多少；同时报另外三项会变成多少、
和现在的清单比换了哪些集。只报数，不改清单；用哪一种、交不交由 rxzhou 定。

用法：python3 stages/evaluate/t1_metric_optimal.py <scores.json> <现在的清单.tsv>
"""

import json
import sys

METRICS = ("add_auc", "spider_sr", "maniptrans_sr", "mppe_cm")
LOWER_BETTER = {"mppe_cm"}


def totals(pick):
    return {m: sum(c[m] for c in pick.values()) / len(pick) for m in METRICS}


def fmt(t):
    return "  ".join(f"{m} {t[m]:.6f}" for m in METRICS)


def main():
    scores = json.load(open(sys.argv[1]))
    current = {}
    for line in open(sys.argv[2]):
        f = line.rstrip("\n").split("\t")
        if len(f) >= 3 and f[0] == "1":
            current[int(f[1])] = f[2]
    by_ep = {}
    for c in scores.values():
        by_ep.setdefault(int(c["episode"]), []).append(c)
    missing = sorted(set(current) ^ set(by_ep))
    if missing or len(current) != 30:
        raise SystemExit(f"清单与候选的集对不上：{missing}，清单 {len(current)} 集")
    for ep, run in current.items():
        if run not in scores:
            raise SystemExit(f"ep{ep} 的现选 {run} 不在分数表里")
    cur = {ep: scores[run] for ep, run in current.items()}
    print(f"候选 {len(scores)} 个，30 集；现在的清单：{fmt(totals(cur))}")
    for m in METRICS:
        sign = 1.0 if m in LOWER_BETTER else -1.0
        pick = {}
        for ep, cands in by_ep.items():
            # 这一项最好；平手时留现选，再平手按 AUC 高、MPPE 低
            pick[ep] = min(cands, key=lambda c: (sign * c[m], c["run"] != current[ep], -c["add_auc"], c["mppe_cm"]))
        changed = sorted(ep for ep in pick if pick[ep]["run"] != current[ep])
        print(f"== 按 {m} 逐条挑最好：{fmt(totals(pick))}")
        for ep in changed:
            a, b = cur[ep], pick[ep]
            print(f"   ep{ep}: {a['run']} → {b['run']}  " + "  ".join(f"{k} {a[k]:.4f}→{b[k]:.4f}" for k in METRICS)
                  + f"  4096 世界 MP-SR {a.get('mp_sr_4096')}→{b.get('mp_sr_4096')}")
    print("T1_METRIC_OPTIMAL_DONE")


if __name__ == "__main__":
    main()
