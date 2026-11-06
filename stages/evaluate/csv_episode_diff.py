"""两份打包好的提交 CSV 逐行比，按 episode 列出内容不同的行（Mac 上、仓库根执行，只用标准库）。计划 agent-doc/plan-t3-gb3-switch-2026-10-17.md 第 4 节第 4 条。

行号（第一列 row_id，track_2/e<集>/w<世界>/f<帧>/o<物体>）必须逐行相同；code_commit_url 一列两次打包本来就不同（各自打包时的提交），单独报、不算内容不同；
其余各列（位姿与各个 sha256）逐字比。给了期望的集，不同的集恰好是这几集才算通过。
用法：python3 stages/evaluate/csv_episode_diff.py <旧 CSV> <新 CSV> [期望不同的集,集,…]   最后一行 CSV_EPISODE_DIFF_OK 或 CSV_EPISODE_DIFF_FAIL
"""

import csv
import re
import sys
from collections import Counter

ROW = re.compile(r"^track_2/e(\d{6})/w\d+/f\d+/o\d+$")


def main():
    if len(sys.argv) not in (3, 4):
        sys.exit("用法：csv_episode_diff.py <旧 CSV> <新 CSV> [期望不同的集,集,…]")
    with open(sys.argv[1], newline="") as a, open(sys.argv[2], newline="") as b:
        ra, rb = list(csv.reader(a)), list(csv.reader(b))
    if ra[0] != rb[0]:
        sys.exit(f"表头不同：{ra[0]} 对 {rb[0]}")
    head = ra[0]
    ci = head.index("code_commit_url")
    print(f"行数：旧 {len(ra) - 1}，新 {len(rb) - 1}")
    same_ids = len(ra) == len(rb) and all(x[0] == y[0] for x, y in zip(ra[1:], rb[1:]))
    print(f"逐行的行号都相同：{same_ids}")
    if not same_ids:
        print("CSV_EPISODE_DIFF_FAIL 行号不同")
        sys.exit(1)
    total, diff, urls = Counter(), Counter(), set()
    for x, y in zip(ra[1:], rb[1:]):
        m = ROW.match(x[0])
        if not m:
            sys.exit(f"行号读不了：{x[0]}")
        ep = int(m.group(1))
        total[ep] += 1
        urls.add((x[ci], y[ci]))
        if [v for i, v in enumerate(x) if i != ci] != [v for i, v in enumerate(y) if i != ci]:
            diff[ep] += 1
    print(f"code_commit_url：{sorted(urls)}")
    print("内容不同的行（按 episode：不同行数/该集行数）：" + "，".join(f"ep{e} {diff[e]}/{total[e]}" for e in sorted(diff)))
    ok = True
    if len(sys.argv) == 4:
        want = sorted(int(v) for v in sys.argv[3].split(",") if v.strip())
        ok = sorted(diff) == want
        print(f"不同的 episode 恰好是 {want}：{ok}")
    print("CSV_EPISODE_DIFF_OK" if ok else "CSV_EPISODE_DIFF_FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
