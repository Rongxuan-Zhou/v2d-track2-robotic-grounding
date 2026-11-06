"""t3hand 第 0 节的两处补探（ml40 上用 t3recon 的 gate 环境跑，只读；只报不判）。计划 agent-doc/plan-t3hand-2026-10-27.md 第 0 节。

1. ep15：Tier 3 gb5 参考里接触之后机器人手掌离警示牌一度超过 40 cm（results/ml40/t3hand/clearance_tier3_gb5_ep15.txt）。
   逐帧并排：机器人较近那只手掌到物体的距离（clearance 记录，每 3 帧）、重建里人手到物体的距离（物理检查 hand_d）、
   物体速度（物理检查 speed）、物体轮廓是否对得上视频（t3_frame_confidence 的逐帧标签，若有）。分开的那段里人手也分开 = 重建的人与物体不一致；
   人手没分开 = 重定向把手放错了。
2. ep19：gb5 交付层第 60–220 帧物体的速度与相邻帧转角，标出单帧转角超过 24°（dyn7 的 JUMP_DEG）或速度超过 2 m/s 的帧。
3. 5 集在导出那一局丢物体的那段（results/ml40/t3hand/rollout_error_*.txt）前后，参考物体的最低点高度 obj_low（负 = 穿地）、脚的高度 feet、
   速度与相邻帧转角，每 5 帧一行；另报整集 obj_low < −2 cm 的帧数。
用法：python t3hand_probe.py <t3recon 根> <t3hand 记录目录>   最后一行 T3HAND_PROBE_DONE
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CODE = Path("/lab_data/rxzhou/v2d/t3recon/dev_t3fx/code/stages/recovery/tier3")
sys.path[:0] = [str(CODE), str(CODE / "gate"), str(CODE.parents[2] / "libs" / "v2d_infra")]

VERSION = "tier3_cari4d_gb5_best_dr2s_dyn7"
PHYS = "runs/physics/tier3_cari4d_gb5_best_dr2s/out"


def palm_series(path: Path) -> dict:
    out = {}
    for ln in path.read_text(encoding="utf-8", errors="replace").splitlines():
        p = ln.strip().split("\t")
        if len(p) >= 5 and p[0].isdigit():
            out[int(p[0])] = min(float(p[3]), float(p[4]))
    return out


def main() -> None:
    from t3_align_check import world_pose
    from t3_fix_units import turn_deg
    root, rec = Path(sys.argv[1]) / "t3fx", Path(sys.argv[2])
    sel = {r["episode_index"]: r["episode"] for r in json.loads((root / "runs/tier3_cari4d_gb5_best.selection.json").read_text())}
    # 1. ep15
    name = sel[15]
    z = np.load(root / PHYS / f"{name}.npz")
    palm = palm_series(rec / "clearance_tier3_gb5_ep15.txt")
    hd, sp = z["hand_d"], z["speed"]
    conf = root / "runs" / f"{VERSION}.frame_confidence.json"
    labels = None
    if conf.exists():
        d = json.loads(conf.read_text())
        labels = next((r.get("labels") for r in d if r.get("episode") == name), None)
    print(f"== ep15 {name}：帧 {len(hd)}；分开的帧（机器人手掌 > 100 mm 且在接触段 127 帧之后）")
    print("帧\t机器人手掌mm\t人手m\t物体速度m/s\t轮廓标签")
    apart = [t for t, d in sorted(palm.items()) if t >= 127 and d > 100]
    for t in apart[:: max(1, len(apart) // 25)]:
        print(f"{t}\t{palm[t]:.0f}\t{hd[t]:.3f}\t{sp[t]:.2f}\t{labels[t] if labels else '—'}")
    if apart:
        hh = np.array([hd[t] for t in apart])
        print(f"分开段共 {len(apart)} 个采样帧，{apart[0]}–{apart[-1]}；这些帧里人手到物体的距离中位 {np.median(hh):.3f} m、最大 {hh.max():.3f} m；"
              f"人手 ≤ 0.10 m 的占 {np.mean(hh <= 0.10):.2f}")
    # 2. ep19
    name = sel[19]
    Tw = world_pose(root, VERSION, 19)
    z = np.load(root / PHYS / f"{name}.npz")
    td = turn_deg(Tw[:, :3, :3])
    sp = z["speed"]
    print(f"== ep19 {name}：第 60–220 帧物体速度与相邻帧转角（每 5 帧），*= 转角 > 24° 或速度 > 2 m/s")
    for t in range(60, 221, 5):
        mark = " *" if td[t] > 24 or sp[t] > 2 else ""
        print(f"{t}\t{sp[t]:.2f} m/s\t{td[t]:.1f}°{mark}")
    bad = [t for t in range(len(td)) if td[t] > 24]
    print(f"整集转角 > 24° 的帧 {len(bad)} 个：{bad[:40]}")
    # 3. 丢物体那段的参考物体高度
    for ep, (a, b) in {16: (200, 330), 15: (460, 560), 27: (150, 240), 13: (160, 230), 19: (170, 240)}.items():
        name = sel[ep]
        z = np.load(root / PHYS / f"{name}.npz")
        Tw = world_pose(root, VERSION, ep)
        td = turn_deg(Tw[:, :3, :3])
        ol, ft, sp = z["obj_low"], z["feet"], z["speed"]
        print(f"== {name}：第 {a}–{b} 帧 参考物体最低点高度 / 脚高 / 速度 / 转角；整集 obj_low < −2 cm 的帧 {int((ol < -0.02).sum())}，最低 {ol.min():.3f} m（第 {int(ol.argmin())} 帧）")
        for t in range(a, min(b, len(ol)) + 1, 5):
            print(f"{t}\t{ol[t]:+.3f}\t{ft[t]:+.3f}\t{sp[t]:.2f}\t{td[t]:.1f}°")
    print("T3HAND_PROBE_DONE")


if __name__ == "__main__":
    main()
