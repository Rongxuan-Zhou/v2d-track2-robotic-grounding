"""试跑用：读 diag_rollout.py --palms 存的 npz，查两件事，都过才退出码 0（只用 numpy，FlashCHORD 容器里跑）。
计划 agent-doc/plan-diag-mpsr0-2026-10-02.md 第 3.1 节、第 5 节第 4 步（第 2 版）：
  1. 同步闸：抄下的物体位置与评测器记的 achieved_pose_w 最大差 < 1e-6 m，手掌坐标全有限（抄的时机对）；
  2. 手掌坐标帧闸：重算的“两只手掌到参考手掌距离的较大者”与评测器 tracking_error 的 palm_position_m 最大差 < 1e-4 m
     （抄的是评测器打分用的那两个手掌坐标帧、左右没对调、帧没错位；同 diag_phase.py 的 V0）。
用法：python /repo/stages/evaluate/diag2_sync.py <npz>
"""
import sys

import numpy as np

from diag_phase import palm_frame_check

d = np.load(sys.argv[1], allow_pickle=False)
if "robot_palm_pos" not in d.files:
    print("SYNC_FAIL：npz 里没有 robot_palm_pos（没加 --palms？）")
    sys.exit(1)
palm, sync = d["robot_palm_pos"], float(d["palm_object_sync_max_abs_m"])
rsides = [str(s) for s in d["robot_palm_sides"]]
rnames = [str(s) for s in d["ref_palm_names"]]
ref_idx = {s: next((i for i, n in enumerate(rnames) if n.startswith(f"{s}_")), None) for s in rsides}
pfc = palm_frame_check(d, None, rsides, ref_idx) if all(v is not None for v in ref_idx.values()) else {}
ok = bool(np.isfinite(palm).all()) and np.isfinite(sync) and sync < 1e-6 and 0 in pfc and pfc[0] < 1e-4
print(f"robot_palm_pos {palm.shape}，手 {rsides}，参考手掌 {rnames}，同步最大差 {sync:.3g} m，"
      f"手掌坐标帧闸 {pfc.get(0, float('nan')):.3g} m（错一帧 +1 {pfc.get(1, float('nan')):.3g}、−1 {pfc.get(-1, float('nan')):.3g}）；"
      f"第 0 帧世界 0 两只手掌 {np.round(palm[0, 0], 3).tolist()}")
print("SYNC_OK" if ok else "SYNC_FAIL")
sys.exit(0 if ok else 1)
