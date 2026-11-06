"""两份 Kaggle 榜单快照逐榜比较（Mac 上跑，只用标准库，只读快照记录；tools/board_watch.sh 调用）。
列出哪些榜变了（分数变化、重新提交、新进来或不见了的队），我们在 12 个榜各自的名次与差距，以及两种读法、两种同分规则下的总名次。
读法、名次与同分规则照 track2_rank_sum.py：基线不计名次；同分 kaggle = 先提交者在前（快照里的 submissionDate），site = 队名小写的字母序。
读法 B 把 12 个榜的名次相加，缺榜的队按那个榜的最后一名再加 1 计（只为排总名次；官方没写缺榜怎么算）。读法 A 只算三档都交了的队。
有下列情况就打印 ALERT 行：我们在某个榜的名次变差（任一同分规则）；任一读法、任一同分规则下我们的总名次变差。
--exclude 队名（可重复）：两份快照里都去掉这些队再比（看“不计某队”时的名次；默认不去掉任何队）。
用法：python3 stages/evaluate/board_diff.py <上一份快照> <这一份快照> [--us "Maniphys Lab"] [--exclude 队名 ...]   最后一行 BOARD_DIFF_DONE
"""
import argparse
import sys
from collections import defaultdict

from track2_rank_sum import METRICS, parse, rank, tie_key

KEYS = [(t, m) for t in (1, 2, 3) for m in METRICS]
TIES = ("kaggle", "site")
# 11/04 14:00 MST 起 Kaggle 上的队名从 maniphys.ai 改成 Maniphys Lab（team ID 不变）；旧快照读进来时换成新名，前后两份才对得上。
RENAMED = {"maniphys.ai": "Maniphys Lab"}


def load(path: str) -> dict:
    boards = {k: [(RENAMED.get(team, team), date, score) for team, date, score in rows] for k, rows in parse(path).items()}
    if sorted(boards) != sorted(KEYS) or any(not boards[k] for k in KEYS):
        sys.exit(f"{path}：应有 12 个 Track 2 榜且每个榜都有队伍，读到 {len(boards)} 个")
    return boards


def table(boards: dict) -> dict:
    return {k: {team: (score, date) for team, date, score in rows} for k, rows in boards.items()}


def totals_b(boards: dict, tie: str) -> dict:
    teams = {team for rows in boards.values() for team, _, _ in rows if "Baseline" not in team}
    tot = defaultdict(int)
    for k in KEYS:
        rk = rank(boards[k], k[1], tie)
        for t in teams:
            tot[t] += rk.get(t, len(rk) + 1)
    return dict(tot)


def totals_a(boards: dict, tie: str) -> dict:
    tot, seen = defaultdict(int), defaultdict(int)
    for metric in METRICS:
        per, first = defaultdict(list), {}
        for tier in (1, 2, 3):
            for team, date, score in boards[(tier, metric)]:
                if "Baseline" not in team:
                    per[team].append(score)
                    first[team] = min(first.get(team, date), date)
        avg = {t: sum(v) / 3 for t, v in per.items() if len(v) == 3}
        sign = 1 if metric == "mppe" else -1
        for i, t in enumerate(sorted(avg, key=lambda t: (sign * round(avg[t], 9), tie_key(t, first[t], tie)))):
            tot[t] += i + 1
            seen[t] += 1
    return {t: v for t, v in tot.items() if seen[t] == len(METRICS)}


def position(tot: dict, team: str):
    """（名次, 与我们总和相同的队数）；不在表里返回 (None, 0)。"""
    if team not in tot:
        return None, 0
    return 1 + sum(1 for v in tot.values() if v < tot[team]), sum(1 for t, v in tot.items() if v == tot[team] and t != team)


def standing(boards: dict, k, us: str, tie: str) -> str:
    sign = 1 if k[1] == "mppe" else -1
    order = sorted((x for x in boards[k] if "Baseline" not in x[0]), key=lambda x: (sign * x[2], tie_key(x[0], x[1], tie)))
    names = [x[0] for x in order]
    if us not in names:
        return "未上榜"
    i = names.index(us)
    s = order[i][2]
    if i == 0:
        if len(order) == 1:
            return f"第 1（{s:.5f}，只有我们）"
        return f"第 1（{s:.5f}，第 2 名 {order[1][0]} {order[1][2]:.5f}，差 {abs(s - order[1][2]):.5f}）"
    return f"第 {i + 1}（{s:.5f}，第 1 名 {order[0][0]} {order[0][2]:.5f}，差 {abs(order[0][2] - s):.5f}）"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--us", default="Maniphys Lab")
    ap.add_argument("--exclude", action="append", default=[])
    a = ap.parse_args()
    old, new = load(a.old), load(a.new)
    for b in (old, new):
        for k in KEYS:
            b[k] = [r for r in b[k] if r[0] not in a.exclude]
    if a.exclude:
        print(f"不计：{'、'.join(a.exclude)}")
    ot, nt = table(old), table(new)
    print(f"上一份 {a.old}；这一份 {a.new}")
    alerts, changed = [], 0
    print("## 变了的榜")
    for k in KEYS:
        o, n = ot[k], nt[k]
        items = []
        for team in sorted(set(o) | set(n)):
            if team not in o:
                items.append(f"新进 {team} {n[team][0]:.5f}（{n[team][1][:16]}）")
            elif team not in n:
                items.append(f"不见了 {team}（原 {o[team][0]:.5f}）")
            elif o[team][0] != n[team][0]:
                items.append(f"{team} {o[team][0]:.5f} → {n[team][0]:.5f}（{n[team][1][:16]}）")
            elif o[team][1] != n[team][1]:
                items.append(f"{team} 重新提交，分数没变（{n[team][0]:.5f}，{n[team][1][:16]}）")
        if items:
            changed += 1
            print(f"tier{k[0]}-{k[1]}：" + "；".join(items))
    if not changed:
        print("（没有）")
    print(f"## {a.us} 在各榜（同分按 Kaggle 规则；括号里是差距）")
    for tier in (1, 2, 3):
        print(f"Tier {tier}：" + "；".join(f"{m} {standing(new, (tier, m), a.us, 'kaggle')}" for m in METRICS))
    for tie in TIES:
        firsts = [sum(1 for k in KEYS if rank(b[k], k[1], tie).get(a.us) == 1) for b in (old, new)]
        print(f"第一的榜（同分 {tie}）：{firsts[0]} → {firsts[1]} 个")
        for k in KEYS:
            r0, r1 = rank(old[k], k[1], tie).get(a.us), rank(new[k], k[1], tie).get(a.us)
            if r0 is not None and (r1 is None or r1 > r0):
                alerts.append(f"tier{k[0]}-{k[1]} 名次 {r0} → {r1}（同分 {tie}）")
    print("## 总名次（名次和小者在前）")
    for name, fn in (("读法 B（12 个榜名次相加）", totals_b), ("读法 A（三档平均后 4 个名次相加）", totals_a)):
        for tie in TIES:
            t0, t1 = fn(old, tie), fn(new, tie)
            p0, _ = position(t0, a.us)
            p1, same = position(t1, a.us)
            top = sorted(t1.items(), key=lambda kv: (kv[1], kv[0]))[:4]
            us_now = f"{t1[a.us]}（第 {p1}{'，与 ' + str(same) + ' 队同分' if same else ''}）" if a.us in t1 else "不在表里"
            us_was = f"{t0[a.us]}（第 {p0}）" if a.us in t0 else "不在表里"
            print(f"{name}，同分 {tie}：{a.us} {us_was} → {us_now}；前 4：" + "、".join(f"{t} {v}" for t, v in top))
            if p0 is not None and (p1 is None or p1 > p0 or (p1 == p0 and same and not position(t0, a.us)[1])):
                alerts.append(f"{name}（同分 {tie}）总名次 {p0} → {p1}{'，并列' if same else ''}")
    for x in alerts:
        print("ALERT " + x)
    print(f"BOARD_CHANGED {changed} 个榜" if changed else "BOARD_NO_CHANGE")
    print("BOARD_DIFF_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
