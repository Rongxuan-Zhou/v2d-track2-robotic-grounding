"""冒烟 rollout 里物体在第一次接触之前有没有自己动（训练机上经 scripts/dev/in-flash-chord 在容器里跑，只用 CPU；只读 diag_rollout.py 存的 npz；只报不判）。
计划 agent-doc/plan-t3g5-2026-10-21.md：10/15 的逐帧终止分析里，best_hm 上 ep16、ep25 的物体开局放在约 1 m 高的台面上、仿真里掉了，ep1 倒了；gb5 修了支撑面，
这里用冒烟的 256 个世界看开局物体还掉不掉。冒烟策略几乎没训，机器人在第一次接触之前碰不到物体，物体若在那之前动了，就是自己掉、倒或滑。
读 npz 的 achieved_pose_w（仿真里物体的位姿，(T,W,B,7)，只看第 0 个刚体）、reference_pose_w、contact_active、start_frame、non_finite_worlds。
报：参考第一次接触的帧 t_c（两侧任一侧）；第 25、51、100 帧与 t_c − 1 帧上，仿真物体相对它自己第 0 帧的平移（cm）与高度变化（cm）的世界中位与 p90；
到 t_c 之前平移超过 --pos-cm（默认 5）的世界比例；同期参考物体相对第 0 帧的平移（cm）。
用法（容器内）：python /repo/stages/evaluate/rollout_object_start.py [--pos-cm 5] <rollout npz>...   最后一行 ROLLOUT_OBJECT_START_DONE
"""
import argparse
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="+")
    ap.add_argument("--pos-cm", type=float, default=5.0)
    a = ap.parse_args()
    for f in a.npz:
        z = np.load(f)
        ach = z["achieved_pose_w"][:, :, 0, :3].astype(float)          # (T,W,3)
        ref = z["reference_pose_w"][:, 0, :3].astype(float) if z["reference_pose_w"].ndim == 3 else z["reference_pose_w"][:, 0, 0, :3].astype(float)
        bad = np.asarray(z["non_finite_worlds"], dtype=int) if "non_finite_worlds" in z.files else np.zeros(0, int)
        keep = np.setdiff1d(np.arange(ach.shape[1]), bad)
        ach = ach[:, keep]
        T = ach.shape[0]
        s0 = int(z["start_frame"]) if "start_frame" in z.files else 0
        c = np.asarray(z["contact_active"], dtype=bool) if "contact_active" in z.files else np.zeros((T, 0), bool)
        hit = np.flatnonzero(c.any(axis=1)) if c.size else np.zeros(0, int)
        tc = int(hit[0]) if hit.size else T
        d = np.linalg.norm(ach - ach[:1], axis=-1) * 100                 # (T,W) cm
        dz = (ach[..., 2] - ach[:1, ..., 2]) * 100
        dr = np.linalg.norm(ref - ref[:1], axis=-1) * 100                # (T,) cm
        print(f"{f}：{T} 帧、{len(keep)} 个世界（剔除非有限 {len(bad)} 个）；参考起始帧 {s0}；参考第一次接触在第 {s0 + tc} 帧")
        for t in sorted({25, 51, 100, tc - 1}):
            if 0 <= t < min(T, tc):
                print(f"  第 {s0 + t} 帧：仿真物体平移中位 {np.median(d[t]):.1f} cm、p90 {np.percentile(d[t], 90):.1f} cm；高度变化中位 {np.median(dz[t]):+.1f} cm、"
                      f"p10 {np.percentile(dz[t], 10):+.1f} cm；参考物体平移 {dr[t]:.1f} cm")
        if tc > 0:
            moved = (d[:tc] > a.pos_cm).any(axis=0)
            print(f"  接触之前仿真物体平移超过 {a.pos_cm:g} cm 的世界 {int(moved.sum())}/{len(keep)}（{moved.mean():.1%}）")
    print("ROLLOUT_OBJECT_START_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
