"""按 P2′ 逐集判一份批量重定向记录（Mac 上跑，只用标准库，只读已落盘的结果）。
计划 agent-doc/plan-t3-v2-retarget18-2026-10-10.md 第 4 节；P2′ 出自 agent-doc/plan-t3-pilot-v2-2026-10-10.md 第 4 节（10/10 偏离后按集判）：
重定向后 |间隙| ≤ 30 mm（正负都算，状态是 ALIGN 也照算）且至少一只手有接触，记“过”；否则记“不过”；结果行是 FAIL:*、
或记录头里要求的集没有结果行，记“失败”。
输入：<记录> stages/retarget/eps.sh 经 scripts/dev/record 落盘的输出（要有记录头 “# command  : … eps.sh <档位> <集>…” 与结束标记 EPS_DONE）。
取其中的结果行（档位 ep 序列 帧数 fps 左手接触 右手接触 IK中位 间隙mm 支撑面 状态）与每集日志段里的 “Saved robot sole Z range”。
--no-contact <集>：这几集的接触一条不适用，只按间隙判（任务本身不用手，例如用脚推；要在看到结果之前登记）。
--a-phase <记录>：stages/recovery/t3_float_facts.py 的输出，加三列只报：A 段（第 0–4 帧）的面高、间隙与“重定向间隙 − A 段间隙”。
  A 段静置在支撑面上（面高 > 0）而重定向后最近的是地面时，在原因里注明（只报，不改结论：可能是任务把物体放到了地上，
  也可能是物体中心的水平投影不在支撑面范围内，后者的间隙会是几百 mm 的悬空）。
输出：每集一行，然后 P2_PASS、P2_FAIL、RETARGET_FAIL 三行（集数与集号），有 --no-contact、--a-phase 时另加 CONTACT_NA、SURFACE_TO_FLOOR 行。
用法：python3 stages/retarget/p2_table.py <记录> [--no-contact <集>...] [--a-phase <记录>]
测试：stages/retarget/test_p2_table.py。
"""
import argparse
import re
import sys
from pathlib import Path

GATE_MM = 30.0
COMMAND = re.compile(r"^# command  : .*eps\.sh ([123])((?: \d+)+) *$")
LOG_HEAD = re.compile(r"^## runs/logs/batch/t\d\S*_ep(\d+)\.log（.*）$")
SOLE = re.compile(r"Saved robot sole Z range: \[([+-]?\d+\.\d+), ([+-]?\d+\.\d+)\] m")


def parse(path: Path) -> tuple:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not any(x.strip() == "EPS_DONE" for x in lines):
        raise SystemExit(f"p2_table: {path} 没有结束标记 EPS_DONE")
    heads = [m for m in map(COMMAND.match, lines) if m]
    if len(heads) != 1:
        raise SystemExit(f"p2_table: {path} 里 eps.sh 的记录头有 {len(heads)} 行，应当恰好 1 行")
    want = [int(e) for e in heads[0].group(2).split()]
    rows, sole, cur = {}, {}, None
    for x in lines:
        c = x.split("\t")
        if len(c) == 11 and c[0] in ("1", "2", "3") and c[1].isdigit():
            ep = int(c[1])
            if ep in rows:
                raise SystemExit(f"p2_table: {path} 里 ep{ep} 的结果行出现两次")
            rows[ep] = c
            continue
        m = LOG_HEAD.match(x.strip())
        if m:
            cur = int(m.group(1))
            continue
        m = SOLE.search(x)
        if m and cur is not None:
            sole[cur] = (float(m.group(1)), float(m.group(2)))
    extra = sorted(set(rows) - set(want))
    if extra:
        raise SystemExit(f"p2_table: {path} 里有记录头没要求的集：ep{extra}")
    return want, rows, sole


def a_phase(path: Path) -> dict:
    """集 → (面高_m, 间隙mm 或 None)。"""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not any(x.startswith("T3_FLOAT_FACTS_DONE") for x in lines):
        raise SystemExit(f"p2_table: {path} 没有结束标记 T3_FLOAT_FACTS_DONE")
    out = {}
    for x in lines:
        c = x.split("\t")
        if len(c) == 15 and c[0].isdigit() and c[6]:
            out[int(c[0])] = (float(c[6]), float(c[7]) if c[7] else None)   # 窗口里没有可见帧的那种行，间隙列是空的
    return out


def judge(c: list, contact_na: bool) -> tuple:
    """(结论, 原因)。结论是“过”“不过”“失败”。"""
    st = c[10]
    if st.startswith("FAIL:"):
        return "失败", st
    try:
        gap, cl, cr = float(c[8]), int(c[5]), int(c[6])
    except ValueError:
        return "失败", f"结果行读不出间隙或接触：{st}"
    why = []
    if abs(gap) > GATE_MM:
        why.append(f"|间隙| {abs(gap):.1f} mm > {GATE_MM:.0f} mm（{'穿地' if gap < 0 else '悬空'}）")
    if cl == 0 and cr == 0 and not contact_na:
        why.append("两手都没有接触")
    return ("不过", "；".join(why)) if why else ("过", "")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("record", type=Path)
    ap.add_argument("--no-contact", type=int, nargs="+", default=[])
    ap.add_argument("--a-phase", type=Path)
    a = ap.parse_args(argv)
    want, rows, sole = parse(a.record)
    ag = a_phase(a.a_phase) if a.a_phase else {}
    na = set(a.no_contact)
    print(f"记录 {a.record}{f'；A 段 {a.a_phase}' if a.a_phase else ''}；P2′：|间隙| ≤ {GATE_MM:.0f} mm 且至少一只手有接触"
          f"{f'（接触一条不适用：{sorted(na)}）' if na else ''}")
    print("ep\t序列\t帧\t左手接触\t右手接触\tIK中位\t间隙mm\t支撑面\t状态\t脚底Z_m\tA段面高_m\tA段间隙mm\t差mm\tP2′\t原因")
    out = {"过": [], "不过": [], "失败": []}
    to_floor = []
    for ep in want:
        if ep not in rows:
            out["失败"].append(ep)
            print("\t".join([str(ep)] + ["-"] * 12 + ["失败", "没有结果行"]))
            continue
        c = rows[ep]
        verdict, why = judge(c, ep in na)
        if ep in na and verdict != "失败":
            why = "；".join(x for x in (why, "接触一条不适用，只按间隙判") if x)
        out[verdict].append(ep)
        h, ga = ag.get(ep, (None, None))
        if h is not None and h > 0 and c[9] == "floor":
            to_floor.append(ep)
            why = "；".join(x for x in (why, f"A 段静置在面高 {h:.3f} m 的支撑面上，重定向后最近的是地面") if x)
        z = f"{sole[ep][0]:+.4f}～{sole[ep][1]:+.4f}" if ep in sole else "-"
        diff = f"{float(c[8]) - ga:+.1f}" if ga is not None and verdict != "失败" else "-"
        print("\t".join([str(ep), c[2], c[3], c[5], c[6], c[7], c[8], c[9], c[10], z,
                         f"{h:.3f}" if h is not None else "-", f"{ga:+.1f}" if ga is not None else "-", diff, verdict, why or "-"]))
    n = len(want)
    print(f"P2_PASS {len(out['过'])}/{n}：{' '.join(f'ep{e}' for e in out['过']) or '无'}")
    print(f"P2_FAIL {len(out['不过'])}/{n}：{' '.join(f'ep{e}' for e in out['不过']) or '无'}")
    print(f"RETARGET_FAIL {len(out['失败'])}/{n}：{' '.join(f'ep{e}' for e in out['失败']) or '无'}")
    if na:
        print("CONTACT_NA " + " ".join(f"ep{e}={next((k for k, v in out.items() if e in v), '不在记录里')}" for e in sorted(na)))
    if a.a_phase:
        print(f"SURFACE_TO_FLOOR {len(to_floor)}：{' '.join(f'ep{e}' for e in to_floor) or '无'}（A 段在支撑面上、重定向后最近的是地面；只报）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
