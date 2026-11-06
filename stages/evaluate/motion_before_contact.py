"""一档参考里物体开始动的帧与接触标记开始的帧（训练机上经 scripts/dev/in-flash-chord 在容器里跑，只用 CPU；只读这一档的重定向参考；只报不判）。
起因：agent-doc/plan-c10-extend-2026-10-16.md 执行记录：Tier 2 的 ep28、ep4 物体越线比首次接触标记早（delay −288、−82），
猜测参考里物体先动、接触标记很晚才开始，接触提前几帧管不到。这里量两样：
  物体离开初始位姿的帧：第一个相对第 0 帧平移超过 --pos-cm 或转角超过 --rot-deg 的帧（物体朝向取 object_body_wxyz 的第 0 个部件）；
  接触标记开始的帧：所有手里最早的首次接触帧（hand_contact_active 是 手 × 帧）；
  另报：物体已离开初始位姿、而没有一只手标了接触的帧数（占全长的比例）。
--tier 2、3 时只收 runs/motion/tier<档>/ 下的参考，启动时装 Tier 1 原始数据禁读（同 contact_onset_table.py）；tier_guard 只拦原始数据目录 tier_1_multiview_caption，派生参考靠这里的路径检查（独立复核 MINOR 4）。门槛只用于描述，不进任何方法。
用法（容器内）：python /repo/stages/evaluate/motion_before_contact.py --tier 2 --lost 1,2,4,9,11,17,25,28 [--pos-cm 3] [--rot-deg 10] <parquet>...；
最后一行 MOTION_CONTACT_DONE。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from contact_onset_table import PATH_RE, quat_to_mat, tier_guard  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, required=True, choices=(1, 2, 3))
    ap.add_argument("--lost", required=True)
    ap.add_argument("--pos-cm", type=float, default=3.0)
    ap.add_argument("--rot-deg", type=float, default=10.0)
    ap.add_argument("parquets", nargs="+")
    a = ap.parse_args()
    if a.tier != 1:
        tier_guard.forbid_tier1()
        print("motion_before_contact：已装 Tier 1 原始数据禁读；派生参考只收 runs/motion/tier<档>/（路径检查）", flush=True)
    lost = {int(x) for x in a.lost.split(",") if x.strip()}
    rows = []
    for p in a.parquets:
        m = PATH_RE.search(p)
        if not m or m.group(1) != f"tier{a.tier}":
            raise SystemExit(f"只收 runs/motion/tier{a.tier}/ 下的参考：{p}")
        meta = Path(p.split("/runs/motion/")[0]) / "runs" / "motion_src" / m.group(1) / m.group(2) / "v2d_meta.json"
        j = json.loads(meta.read_text())
        if int(j["tier"]) != a.tier:
            raise SystemExit(f"{meta} 的档位是 {j['tier']}，不是 {a.tier}")
        r = pq.read_table(p, columns=["hand_sides", "hand_contact_active", "object_root_position", "object_body_wxyz"]).to_pylist()[0]
        obj = np.asarray(r["object_root_position"], float)
        c = np.asarray(r["hand_contact_active"], float)
        if c.ndim != 2 or c.shape != (len(r["hand_sides"]), len(obj)):
            raise SystemExit(f"{p}：hand_contact_active 形状 {c.shape}，应为 手 × 帧")
        R = quat_to_mat(np.asarray([b[0] for b in r["object_body_wxyz"]], float))
        dpos = np.linalg.norm(obj - obj[0], axis=1) * 100
        drot = np.degrees(np.arccos(np.clip((np.einsum("nij,ij->n", R, R[0]) - 1) / 2, -1, 1)))
        away = (dpos > a.pos_cm) | (drot > a.rot_deg)
        anyc = (c > 0.5).any(axis=0)
        t_move = int(np.argmax(away)) if away.any() else None
        t_con = int(np.argmax(anyc)) if anyc.any() else None
        lone = int((away & ~anyc).sum())
        rows.append((int(j["episode"]), len(obj), t_move, t_con, lone))
    rows.sort()
    for ep, n, tm, tc, lone in rows:
        gap = "" if tm is None or tc is None else f"，接触比离开晚 {tc - tm:+d} 帧"
        print(f"ep{ep}{'（跟丢）' if ep in lost else ''}：{n} 帧；物体离开初始位姿第 {tm if tm is not None else '无'} 帧、首次接触第 "
              f"{tc if tc is not None else '无'} 帧{gap}；离开初始位姿而没有接触标记的 {lone} 帧（{lone / n:.0%}）")
    for name, sel in (("跟丢", [r for r in rows if r[0] in lost]), ("其余", [r for r in rows if r[0] not in lost])):
        g = [tc - tm for _, _, tm, tc, _ in sel if tm is not None and tc is not None]
        f = [lone / n for _, n, _, _, lone in sel]
        if sel:
            print(f"{name} {len(sel)} 集：接触晚于离开的帧数中位 {float(np.median(g)) if g else float('nan'):+.0f}（{len(g)} 集有两者），"
                  f"晚 30 帧以上的 {sum(x > 30 for x in g)} 集；离开而没有接触的帧占比中位 {float(np.median(f)):.0%}")
    print("MOTION_CONTACT_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
