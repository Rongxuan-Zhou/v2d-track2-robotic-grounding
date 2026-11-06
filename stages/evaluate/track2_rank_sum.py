"""按官方 FAQ 的“各指标名次相加、和最小者第一”算 Track 2 的总名次（Mac 上跑，只用标准库；只读一份榜单快照记录）。
官方没有写明三个档位与四个指标怎么合（`results/ops/challenge_faq_1017.txt`；官网“Each tier is weighted equally”），这里两种读法都算：
  读法 B：12 个榜各自排名，名次直接相加（三档等权即简单相加）；
  读法 A：每个指标先把三档的分数取平均，再排名，4 个名次相加（缺某档的队伍不参加读法 A）。
排名规则：基线（队名含 Baseline）不计名次（同官网 leaderboard.js）；同分怎么排有两种，--tie 选：
  kaggle（默认）：先提交者在前（Kaggle 规则 7.b；提交时间取快照里的 submissionDate，即这队最近一次提交的时间）；
  site：队名小写按字母序（官网 leaderboard.js 的 compare 函数，`results/ops/site_rank_code_1017.txt`）。
MPPE 越小越好，其余越大越好。输入：scripts/dev/record 落盘的 kaggle competitions leaderboard --show 输出（每榜以“== <比赛名>”开头）。
用法：python3 stages/evaluate/track2_rank_sum.py results/submission/leaderboard_1017_1815.txt [--teams "maniphys.ai,Xueyi Wang (FF)"] [--tie kaggle|site]
最后一行 RANK_SUM_DONE。
"""
import argparse
import re
import sys
from collections import defaultdict

ROW = re.compile(r"^\s*\d+\s+(.+?)\s+(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s*$")
METRICS = ["auc", "sp-sr", "mp-sr", "mppe"]


def parse(path: str) -> dict:
    boards, cur = {}, None
    for line in open(path, encoding="utf-8"):
        m = re.match(r"^== v2d-challenge-track2-tier(\d)-([a-z-]+)\s*$", line)
        if m:
            cur = (int(m.group(1)), m.group(2))
            boards[cur] = []
            continue
        r = ROW.match(line)
        if cur and r:
            boards[cur].append((r.group(1).strip(), r.group(2), float(r.group(3))))
    return boards


def tie_key(team: str, date: str, tie: str):
    return date if tie == "kaggle" else team.lower()


def rank(rows: list, metric: str, tie: str = "kaggle") -> dict:
    rows = [x for x in rows if "Baseline" not in x[0]]
    sign = 1 if metric == "mppe" else -1
    rows.sort(key=lambda x: (sign * x[2], tie_key(x[0], x[1], tie)))
    return {team: i + 1 for i, (team, _, _) in enumerate(rows)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("--teams", default="maniphys.ai,Xueyi Wang (FF)")
    ap.add_argument("--tie", choices=("kaggle", "site"), default="kaggle")
    a = ap.parse_args()
    boards = parse(a.snapshot)
    teams = [t.strip() for t in a.teams.split(",")]
    assert len(boards) == 12, f"快照里应有 12 个 Track 2 榜，读到 {len(boards)} 个"
    tie_text = "同分按先提交者在前（Kaggle）" if a.tie == "kaggle" else "同分按队名字母序（官网 leaderboard.js）"
    print(f"快照 {a.snapshot}；基线不计名次，{tie_text}")
    total_b = defaultdict(int)
    for tier in (1, 2, 3):
        parts = []
        for metric in METRICS:
            rk = rank(boards[(tier, metric)], metric, a.tie)
            for t in teams:
                total_b[(t, tier)] += rk.get(t, 99)
            parts.append(f"{metric} " + "、".join(f"{t} {rk.get(t, '—')}" for t in teams))
        print(f"Tier {tier}：" + "；".join(parts) + "；名次和 " + "、".join(f"{t} {total_b[(t, tier)]}" for t in teams))
    print("读法 B（12 个榜名次直接相加）：" + "、".join(f"{t} {sum(total_b[(t, k)] for k in (1, 2, 3))}" for t in teams))
    avg = {}
    for metric in METRICS:
        per = defaultdict(list)
        first = {}
        for tier in (1, 2, 3):
            for team, date, score in boards[(tier, metric)]:
                if "Baseline" in team:
                    continue
                per[team].append(score)
                first[team] = min(first.get(team, date), date)
        avg[metric] = {t: (sum(v) / 3, first[t]) for t, v in per.items() if len(v) == 3}
    total_a = defaultdict(int)
    for metric in METRICS:
        sign = 1 if metric == "mppe" else -1
        order = sorted(avg[metric].items(), key=lambda kv: (sign * round(kv[1][0], 9), tie_key(kv[0], kv[1][1], a.tie)))
        rk = {t: i + 1 for i, (t, _) in enumerate(order)}
        ties = len({round(v[0], 9) for v in avg[metric].values()}) < len(avg[metric])
        print(f"读法 A {metric}：" + "、".join(f"{t} {avg[metric][t][0]:.5f}（第 {rk[t]}）" for t in teams if t in avg[metric])
              + (("；有同分，按各队在这一指标上最早的提交时间排" if a.tie == "kaggle" else "；有同分，按队名字母序排") if ties else ""))
        for t in teams:
            total_a[t] += rk.get(t, 99)
    print("读法 A（三档平均后 4 个名次相加，只算三档都交了的队伍）：" + "、".join(f"{t} {total_a[t]}" for t in teams))
    print("RANK_SUM_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
