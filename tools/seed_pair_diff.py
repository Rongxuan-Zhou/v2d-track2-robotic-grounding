#!/usr/bin/env python3
"""同配置换种子的成对 run：逐对列出某一列（默认 MP-SR）的两个值与差，再给出差的最大值与中位数。

输入是 stages/evaluate/eval_table.py 的输出（record 文件也行，# 开头的行跳过）。run 名去掉 -s<种子> 后相同的两行算一对，
只有一行的跳过。可以用 --only 只看名字含某段的 run（例如 -r3a-）。

用法：python3 tools/seed_pair_diff.py <eval_table 输出> [--col MP-SR] [--only <子串>]
"""
import argparse
import re
import statistics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("table")
    ap.add_argument("--col", default="MP-SR")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    rows = [l.rstrip("\n").split("\t") for l in open(a.table, encoding="utf-8") if l.strip() and not l.startswith("#")]
    head, body = rows[0], rows[1:]
    ci = head.index(a.col)
    groups = {}
    for r in body:
        if a.only and a.only not in r[0]:
            continue
        base = re.sub(r"-s\d+(?=-)", "", r[0])
        groups.setdefault(base, []).append((r[0], float(r[ci])))
    diffs = []
    for base, runs in groups.items():
        if len(runs) != 2:
            print(f"跳过 {base}：{len(runs)} 行")
            continue
        (n1, v1), (n2, v2) = runs
        d = abs(v1 - v2)
        diffs.append((d, base))
        print(f"{base}  {a.col} {v1:.3f}（{n1}）与 {v2:.3f}（{n2}），差 {d:.3f}")
    if not diffs:
        print("没有成对的 run")
        return
    ds = sorted(d for d, _ in diffs)
    mx = max(diffs)
    print(f"共 {len(ds)} 对；{a.col} 差的中位数 {statistics.median(ds):.3f}，最大 {mx[0]:.3f}（{mx[1]}）；"
          f"差超过 0.2 的 {sum(d > 0.2 for d in ds)} 对，超过 0.4 的 {sum(d > 0.4 for d in ds)} 对")


if __name__ == "__main__":
    main()
