"""把我们的 P2′ 表与提供方在他们那边跑同一份门槛代码的记录逐集对照（Mac 上跑，只用标准库，只读已落盘的结果）。
计划 agent-doc/plan-t3-hm-p2-2026-10-12.md 第 4 节第 9 步、第 5 节“与提供方记录对照”：只报，不改 P2′ 的结论。
输入：
  <我们的表>   stages/retarget/p2_table.py 的输出（经 scripts/dev/record 落盘），按表头行（以 “ep\t序列” 开头）认列；
  <对方记录>   results/t3hm/their_gate_bhm_ml40.txt 这样的记录：几段 “==== <文件名>（…）” 开头的原文。用其中两段：
               --p2 段（默认 p2_bhm.txt）：对方的 p2_table.py 输出，取帧、左右手接触、IK 中位、脚底 Z（这几栏与支撑面无关）；
               --sup 段（默认 oursup_bhm.txt）：对方换上自己的支撑面重判的结果，每行 “<集> RESULT gap_mm=<x> surface=<面> ok=<0|1>”，
               取间隙、最近的面与对齐是否通过。
对照（我们的表里每一集）：帧、左手接触、右手接触、IK中位、脚底Z_m 与 --p2 段逐字相同；间隙与 --sup 段之差的绝对值 ≤ 0.1 mm；
最近的面相同；我们的状态以 OK 开头当且仅当 --sup 段 ok=1。每集打一行，全部相同打 THEIRS_SAME <n>/<n>，否则打 THEIRS_DIFF 与不同的集和栏；
ONLY_OURS、ONLY_THEIRS 列出只在一边的集（只在对方的不算不同）。我们的表里 “失败” 行（数值是 “-”）算不同。
用法：python3 stages/retarget/p2_vs_theirs.py <我们的表> <对方记录> [--p2 p2_bhm.txt] [--sup oursup_bhm.txt]
测试：stages/retarget/test_p2_vs_theirs.py。
"""
import argparse
import re
import sys
from pathlib import Path

SAME_COLS = ("帧", "左手接触", "右手接触", "IK中位", "脚底Z_m")
NEED = ("序列",) + SAME_COLS + ("间隙mm", "支撑面", "状态")
SECTION = re.compile(r"^==== (\S+?)（")
SUP = re.compile(r"^(\d+) RESULT gap_mm=(\S+) surface=(\S+) ok=([01])$")
GAP_TOL_MM = 0.1


def table(lines: list[str], where: str) -> dict:
    head = next((x.split("\t") for x in lines if x.startswith("ep\t序列")), None)
    if head is None:
        raise SystemExit(f"p2_vs_theirs: {where} 里没有表头行（ep\\t序列…）")
    missing = [c for c in NEED if c not in head]
    if missing:
        raise SystemExit(f"p2_vs_theirs: {where} 的表头缺列 {missing}")
    rows = {}
    for x in lines:
        f = x.split("\t")
        if len(f) == len(head) and f[0].isdigit():
            ep = int(f[0])
            if ep in rows:
                raise SystemExit(f"p2_vs_theirs: {where} 里 ep{ep} 出现两次")
            rows[ep] = {c: f[head.index(c)] for c in NEED}
    if not rows:
        raise SystemExit(f"p2_vs_theirs: {where} 里没有数据行")
    return rows


def sections(path: Path) -> dict:
    out, cur = {}, None
    for x in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = SECTION.match(x)
        if m:
            cur = m.group(1)
            if cur in out:
                raise SystemExit(f"p2_vs_theirs: {path} 里 {cur} 这一段出现两次")
            out[cur] = []
        elif cur is not None:
            out[cur].append(x)
    return out


def sup_rows(lines: list[str], where: str) -> dict:
    rows = {}
    for x in lines:
        m = SUP.match(x.strip())
        if m:
            ep = int(m.group(1))
            if ep in rows:
                raise SystemExit(f"p2_vs_theirs: {where} 里 ep{ep} 出现两次")
            rows[ep] = {"gap": float(m.group(2)), "surface": m.group(3), "ok": m.group(4) == "1"}
    if not rows:
        raise SystemExit(f"p2_vs_theirs: {where} 里一行 RESULT 也没有")
    return rows


def compare(ours: dict, theirs: dict, sup: dict) -> tuple:
    lines, diff = [], {}
    for ep in sorted(ours):
        if ep not in theirs or ep not in sup:
            continue
        o, t, s = ours[ep], theirs[ep], sup[ep]
        bad = [c for c in SAME_COLS if o[c] != t[c]]
        try:
            g = float(o["间隙mm"])
            if abs(g - s["gap"]) > GAP_TOL_MM:
                bad.append("间隙mm")
        except ValueError:
            bad.append("间隙mm")
        if o["支撑面"] != s["surface"]:
            bad.append("支撑面")
        if o["状态"].startswith("OK") != s["ok"]:
            bad.append("状态")
        if bad:
            diff[ep] = bad
        lines.append("\t".join([str(ep), o["序列"], o["帧"], t["帧"], o["左手接触"], t["左手接触"], o["右手接触"], t["右手接触"],
                                o["IK中位"], t["IK中位"], o["脚底Z_m"], t["脚底Z_m"], o["间隙mm"], f"{s['gap']:+.1f}",
                                o["支撑面"], s["surface"], o["状态"], f"ok={int(s['ok'])}",
                                "同" if not bad else "不同：" + "、".join(bad)]))
    return lines, diff


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ours", type=Path)
    ap.add_argument("theirs", type=Path)
    ap.add_argument("--p2", default="p2_bhm.txt")
    ap.add_argument("--sup", default="oursup_bhm.txt")
    a = ap.parse_args(argv)
    ours = table(a.ours.read_text(encoding="utf-8", errors="replace").splitlines(), str(a.ours))
    sec = sections(a.theirs)
    for name in (a.p2, a.sup):
        if name not in sec:
            raise SystemExit(f"p2_vs_theirs: {a.theirs} 里没有 {name} 这一段（有：{', '.join(sec) or '无'}）")
    theirs = table(sec[a.p2], f"{a.theirs} 的 {a.p2} 段")
    sup = sup_rows(sec[a.sup], f"{a.theirs} 的 {a.sup} 段")
    print(f"我们：{a.ours}；对方：{a.theirs} 的 {a.p2} 段（帧、接触、IK、脚底 Z）与 {a.sup} 段（间隙、最近的面、对齐）；间隙容差 {GAP_TOL_MM} mm")
    print("ep\t序列\t帧_我们\t帧_对方\t左手_我们\t左手_对方\t右手_我们\t右手_对方\tIK_我们\tIK_对方\t脚底Z_我们\t脚底Z_对方"
          "\t间隙_我们\t间隙_对方\t面_我们\t面_对方\t状态_我们\t对齐_对方\t对照")
    lines, diff = compare(ours, theirs, sup)
    print("\n".join(lines))
    n = len(lines)
    if diff:
        items = "；".join("ep%d（%s）" % (e, "、".join(c)) for e, c in sorted(diff.items()))
        print(f"THEIRS_DIFF {items}（{len(diff)}/{n}）")
    else:
        print(f"THEIRS_SAME {n}/{n}")
    both = set(theirs) & set(sup)
    print(f"ONLY_OURS {' '.join(f'ep{e}' for e in sorted(set(ours) - both)) or '无'}")
    print(f"ONLY_THEIRS {' '.join(f'ep{e}' for e in sorted(both - set(ours))) or '无'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
