"""同一份 rollout 上新旧两版 diag_report.py 输出的逐行比对（只用标准库，任何机器都能跑）。

回放速度对照的计划第 2 版要求：Day 1b 已有的最终 checkpoint rollout 不重录，只用新版报告重算，并与旧报告逐行比对。
新版只新增 Q2b 一节与 summary_json 的字段，改了 G1 那一行的说明（容差随世界数缩放）；svg 行随输出路径变。
去掉这几类行（和两边 scripts/dev/record 的头部）之后，其余每一行都应一字不差，否则列出差异、退出码 1。
用法：python3 stages/evaluate/diag_report_diff.py <旧报告> <新报告>
"""
import difflib
import sys
from pathlib import Path


def body(path: str) -> list:
    lines = Path(path).read_text().splitlines()
    heads = [i for i, ln in enumerate(lines) if ln.startswith("# ---")]
    if heads:
        lines = lines[heads[0] + 1:]
    out, in_q2b = [], False
    for ln in lines:
        if ln.startswith("== "):
            in_q2b = ln.startswith("== Q2b")
        s = ln.strip()
        if in_q2b or s.startswith(("summary_json ", "svg:")) or "G1 同卡" in s:
            continue
        out.append(ln)
    return out


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__.strip().splitlines()[-1])
    old, new = body(sys.argv[1]), body(sys.argv[2])
    diff = [d for d in difflib.unified_diff(old, new, lineterm="", n=0) if not d.startswith(("---", "+++", "@@"))]
    print(f"比对 {sys.argv[1]} → {sys.argv[2]}：去掉 Q2b、summary_json、svg 与 G1 行之后，旧 {len(old)} 行、新 {len(new)} 行")
    if diff:
        print("有意料之外的差异：\n" + "\n".join(diff))
        return 1
    print("其余逐行相同")
    return 0


if __name__ == "__main__":
    sys.exit(main())
