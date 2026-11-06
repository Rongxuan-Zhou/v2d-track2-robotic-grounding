"""假如 maniphys.ai 在某一档重交：把榜单快照里这一队这一档的四项分数换成给定值，另存一份快照，供 track2_rank_sum.py 算总名次。

只读快照记录，只写 --out。提交时间换成 --date（重交后 Kaggle 记的是新时间，同分按先提交者在前时会排到后面）。
Mac 上跑，只用标准库。
用法：python3 tools/board_whatif.py <快照> --tier 1 --auc 0.64408 --sp 0.86667 --mp 0.70000 --mppe 8.40806 --date "2026-11-02 12:00:00.000000" --out <新快照>
最后一行 WHATIF_DONE。
"""
import argparse
import re

HEAD = re.compile(r"^== v2d-challenge-track2-tier(\d)-([a-z-]+)\s*$")
ROW = re.compile(r"^(\s*\d+\s+)(maniphys\.ai)(\s+)(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:\.\d+)?)(\s+)(-?\d+(?:\.\d+)?)(\s*)$")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("snapshot")
    ap.add_argument("--tier", type=int, required=True, choices=(1, 2, 3))
    ap.add_argument("--auc", type=float, required=True)
    ap.add_argument("--sp", type=float, required=True)
    ap.add_argument("--mp", type=float, required=True)
    ap.add_argument("--mppe", type=float, required=True)
    ap.add_argument("--date", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    new = {"auc": a.auc, "sp-sr": a.sp, "mp-sr": a.mp, "mppe": a.mppe}
    cur, done, out = None, set(), []
    for line in open(a.snapshot, encoding="utf-8"):
        m = HEAD.match(line)
        if m:
            cur = (int(m.group(1)), m.group(2))
        r = ROW.match(line.rstrip("\n"))
        if r and cur and cur[0] == a.tier and cur[1] in new:
            line = f"{r.group(1)}{r.group(2)}{r.group(3)}{a.date}{r.group(5)}{new[cur[1]]:.5f}{r.group(7)}\n"
            done.add(cur[1])
            print(f"Tier {a.tier} {cur[1]}：{r.group(6)} → {new[cur[1]]:.5f}")
        out.append(line)
    assert done == set(new), f"快照里没找到 maniphys.ai 的 Tier {a.tier} 行：缺 {set(new) - done}"
    with open(a.out, "w", encoding="utf-8") as f:
        f.writelines(out)
    print(f"写入 {a.out}")
    print("WHATIF_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
