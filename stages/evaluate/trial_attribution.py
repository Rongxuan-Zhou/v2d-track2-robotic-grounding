"""第 3 步配方试验的归因项（只报）：读 wandb_local_history.py --by-stage 的输出，比较变体 A 与基线在第 6–8 段（VOC = 0）的“终止:物体”与“进度”。

计划 agent-doc/plan-step3-recipe-trial-2026-10-04.md 第 4 节“只报、作归因”：这一档不通过时，若变体 A 两个种子这两项的平均，终止:物体 ≥ 基线的 2 倍
且进度比基线低 0.1 以上，记为“朝向终止过严或参考做不到”，与“配方无效”分开写。
每条 run 取第 6、7、8 段三行的简单平均（各段已是该段最后 25% 行的均值）；episode 的变体 = 两个变体 run 的平均，基线 = 两个基线 run 的平均；
档位的数 = 5 条 episode 的平均。只用标准库。

用法：python3 stages/evaluate/trial_attribution.py <by-stage 输出文件> <trial 清单 json>
"""

import json
import re
import sys

STAGES = (6, 7, 8)
RATIO = 2.0
PROGRESS_DROP = 0.1


def parse(text):
    """返回 {run 名: {段号: {列名: 值}}}。run 名取自 '## <run>' 行。"""
    runs, run, header = {}, None, None
    for line in text.splitlines():
        if line.startswith("## "):
            run = line[3:].strip()
            runs[run] = {}
            header = None
            continue
        if run is None or "|" not in line:
            continue
        cells = [c.strip() for c in line.strip().split("|")]
        if cells[0] == "段":
            header = cells
            continue
        if header and re.fullmatch(r"\d+", cells[0]):
            row = {}
            for k, v in zip(header, cells):
                try:
                    row[k] = float(v)
                except ValueError:
                    row[k] = v
            runs[run][int(cells[0])] = row
    return runs


def late(runs, run, key):
    rows = runs.get(run, {})
    missing = [s for s in STAGES if s not in rows]
    if missing:
        raise SystemExit(f"{run} 缺第 {missing} 段")
    return sum(rows[s][key] for s in STAGES) / len(STAGES)


def names(e):
    return [e["base42"], e["base43"]], [e["var42"], e["var43"]]


def main(argv):
    runs = parse(open(argv[1], encoding="utf8").read())
    spec = json.load(open(argv[2], encoding="utf8"))
    print("第 6–8 段（VOC = 0）的终止:物体与进度；变体与基线各取两种子平均")
    print("  ep     终止:物体 A   B     比     进度 A   B     差")
    sums = {"ta": 0.0, "tb": 0.0, "pa": 0.0, "pb": 0.0}
    eps = spec["episodes"]
    for e in eps:
        ep = e["ep"]
        b, a = names(e)
        ta = sum(late(runs, r, "终止:物体") for r in a) / 2
        tb = sum(late(runs, r, "终止:物体") for r in b) / 2
        pa = sum(late(runs, r, "进度") for r in a) / 2
        pb = sum(late(runs, r, "进度") for r in b) / 2
        for k, v in (("ta", ta), ("tb", tb), ("pa", pa), ("pb", pb)):
            sums[k] += v
        ratio = ta / tb if tb > 0 else float("inf")
        print(f"  ep{ep:<4} {ta:9.4f} {tb:7.4f} {ratio:6.2f}   {pa:7.3f} {pb:6.3f} {pa - pb:+7.3f}")
    n = len(eps)
    ta, tb, pa, pb = (sums[k] / n for k in ("ta", "tb", "pa", "pb"))
    ratio = ta / tb if tb > 0 else float("inf")
    hit = ratio >= RATIO and (pb - pa) > PROGRESS_DROP
    print(f"  平均   {ta:9.4f} {tb:7.4f} {ratio:6.2f}   {pa:7.3f} {pb:6.3f} {pa - pb:+7.3f}")
    print(f"ATTRIBUTION {'朝向终止过严或参考做不到' if hit else '不满足“终止过严”的归因条件'}"
          f"（终止:物体比 {ratio:.2f}，要 ≥ {RATIO}；进度差 {pa - pb:+.3f}，要 < −{PROGRESS_DROP}）")


if __name__ == "__main__":
    main(sys.argv)
