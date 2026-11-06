"""诊断 rollout 的物体误差随帧变化（只用 numpy；读 diag_rollout.py 存的 npz；容器内跑；只报不判）。
起因：agent-doc/experiments/2026-10-30-t1hard-diag.md：ep1 的 1024 个世界都在第 3 帧物体误差就超过 0.10 m，与导出那一局（第 281 帧才超）不一致，要看误差的形状。
报：指定帧上，物体位置误差（m）与仿真物体 z、参考物体 z 的世界分位 10/50/90；以及第 0 帧仿真物体与参考物体的位置差。
用法（容器内）：python /repo/stages/evaluate/diag_err_profile.py <npz> [--frames 0,1,2,3,5,10,20,50,100,130,200,300]   最后一行 DIAG_ERR_PROFILE_DONE
"""
import argparse
import sys

import numpy as np


def q3(x) -> str:
    return " / ".join(f"{np.percentile(x, p):.3f}" for p in (10, 50, 90))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--frames", default="0,1,2,3,5,10,20,50,100,130,200,300")
    a = ap.parse_args()
    d = np.load(a.npz, allow_pickle=False)
    ach = d["achieved_pose_w"][:, :, 0].astype(np.float64)
    refp = d["reference_pose_w"][:, 0].astype(np.float64)
    T, W = ach.shape[:2]
    print(f"{a.npz}：T={T} W={W}；start_frame {int(d['start_frame'])}")
    print("帧\t位置误差 m（10/50/90）\t仿真 z（10/50/90）\t参考 z\t仿真 xy 相对参考（中位 dx, dy）")
    for t in [int(x) for x in a.frames.split(",") if int(x) < T]:
        err = np.linalg.norm(ach[t, :, :3] - refp[t, None, :3], axis=-1)
        dxy = np.median(ach[t, :, :2] - refp[t, None, :2], axis=0)
        print(f"{t}\t{q3(err)}\t{q3(ach[t, :, 2])}\t{refp[t, 2]:.3f}\t{dxy[0]:+.3f}, {dxy[1]:+.3f}")
    print("DIAG_ERR_PROFILE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
