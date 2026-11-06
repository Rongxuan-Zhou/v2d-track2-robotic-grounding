"""t3hand 诊断的汇总（Mac 或训练机上用 python3 跑，只用标准库）。计划 agent-doc/plan-t3hand-2026-10-27.md 第 0 节。

读 results/ml40/t3hand/ 下 t3hand_diag.sh 的三类记录，每集一行、Tier 3 gb5 与 Tier 2 并排：
  clearance_<档>_ep<N>.txt   逐帧（每 3 帧）两只手掌到物体表面的有向距离（mm，外正内负）：
      首次 ≤ 20 mm 的帧、首次 ≤ 50 mm 的帧、参考接触 t_c 前 30 帧里较近那只手的最小距离、
      接触段（t_c 起）里较近那只手的距离中位与 90% 分位、接触段里两只手都 > 50 mm 的帧比例（“手没贴住”）；
  contact_<档>_ep<N>.txt     首次接触帧、物体首次平移 > 3 cm 的帧、首次接触时已平移、无接触而在动的帧数；
  penetration_<档>_ep<N>.txt 手掌点最小有向距离的接触帧中位（mm）、手掌有点在物体里面的接触帧比例。
用法：python3 stages/retarget/t3hand_summary.py results/ml40/t3hand   最后一行 T3HAND_SUMMARY_DONE
"""

import re
import statistics as st
import sys
from pathlib import Path

EPS = (16, 19, 13, 15, 27, 3, 12, 23)
TIERS = ("tier3_gb5", "tier2")


def clearance(path: Path):
    if not path.exists():
        return None
    rows, tc = [], None
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.search(r"参考第一次接触 t_c (\S+)", ln)
        if m:
            tc = None if m.group(1) in ("None", "—") else int(m.group(1))
        p = ln.strip().split("\t")
        if len(p) >= 5 and p[0].isdigit():
            try:
                rows.append((int(p[0]), float(p[3]), float(p[4])))
            except ValueError:
                pass
    if not rows:
        return None
    near = [(t, min(l, r)) for t, l, r in rows]

    def first(th):
        return next((t for t, d in near if d <= th), None)

    out = {"tc": tc, "first20": first(20), "first50": first(50), "frames": len(rows)}
    if tc is not None:
        pre = [d for t, d in near if tc - 30 <= t < tc]
        post = [d for t, d in near if t >= tc]
        out["pre_min"] = round(min(pre), 1) if pre else None
        out["post_p50"] = round(st.median(post), 1) if post else None
        out["post_p90"] = round(sorted(post)[int(0.9 * (len(post) - 1))], 1) if post else None
        out["post_off50"] = round(sum(d > 50 for d in post) / len(post), 2) if post else None
    return out


def contact(path: Path):
    if not path.exists():
        return None
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "首次接触" in ln and "\t" in ln:
            g = lambda k: re.search(k + r"\s*(\S+)", ln)
            return {"first_contact": g("首次接触").group(1), "obj_move": g("物体首次平移>3cm").group(1),
                    "moved_at_contact": g("首次接触时已平移 m").group(1), "nocontact_moving": g("无接触而在动的帧").group(1)}
    return None


def penetration(path: Path):
    if not path.exists():
        return None
    txt = path.read_text(encoding="utf-8", errors="replace")
    lines = [ln for ln in txt.splitlines() if "\t" in ln and not ln.startswith("#") and "REF_PENETRATION" not in ln]
    return {"line": " || ".join(ln[:220] for ln in lines)} if lines else None


def main() -> None:
    d = Path(sys.argv[1])
    print("集\t档\t接触t_c\t手首≤20mm\t手首≤50mm\t接触前30帧最小mm\t接触段中位/90%mm\t接触段双手>50mm比例\t物体首动\t接触时已移m\t无接触而动帧")
    for ep in EPS:
        for tier in TIERS:
            c = clearance(d / f"clearance_{tier}_ep{ep}.txt") or {}
            k = contact(d / f"contact_{tier}_ep{ep}.txt") or {}
            print(f"ep{ep}\t{tier}\t{c.get('tc')}\t{c.get('first20')}\t{c.get('first50')}\t{c.get('pre_min')}\t{c.get('post_p50')}/{c.get('post_p90')}\t{c.get('post_off50')}\t"
                  f"{k.get('obj_move')}\t{k.get('moved_at_contact')}\t{k.get('nocontact_moving')}")
    print("== 穿透（接触帧里手掌点最小有向距离；原始行）")
    for ep in EPS:
        for tier in TIERS:
            p = penetration(d / f"penetration_{tier}_ep{ep}.txt")
            print(f"ep{ep}\t{tier}\t{p['line'] if p else None}")
    print("T3HAND_SUMMARY_DONE")


if __name__ == "__main__":
    main()
