"""机器人从哪一帧开始失衡，与参考在那段要它做什么（训练机上经 scripts/dev/in-flash-chord 在容器里跑，只用 numpy 与 pyarrow；只读已有文件，只报不判）。
起因：agent-doc/plan-t12lift-2026-10-20.md 事实 1：Tier 2 的全失败集都摔了（ep1 第 27 帧、ep4 第 99 帧在参考接触之前就摔），成功的 ep0 没摔。
每 --every 帧一行：参考（motion parquet）的根水平速度（按 --rate 帧/秒回放，--win 帧窗口平均）、偏航角速度、骨盆高度、腰部三个关节角（waist_yaw/roll/pitch）、两膝关节角；
跟踪误差（评测 dump 或诊断 rollout 的 npz）的骨盆位置、骨盆朝向、手掌位置误差的世界中位。另报骨盆朝向误差中位第一次 > 0.2、0.5 rad 的帧。
用法（容器内）：python /repo/stages/evaluate/fall_timeline.py --motion <parquet> --npz <npz> [--to 200] [--every 5] [--rate 50] [--win 10]   最后一行 FALL_TIMELINE_DONE
"""
import argparse
import sys

import numpy as np
import pyarrow.parquet as pq


def yaw(wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = (wxyz[:, i] for i in range(4))
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def smooth(x: np.ndarray, win: int) -> np.ndarray:
    if len(x) <= win:
        return x
    k = np.ones(win) / win
    return np.convolve(np.pad(x, (win // 2, win - 1 - win // 2), mode="edge"), k, mode="valid")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True)
    ap.add_argument("--npz", required=True)
    ap.add_argument("--to", type=int, default=200)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--win", type=int, default=10)
    a = ap.parse_args()
    row = pq.read_table(a.motion, columns=["robot_joint_names", "robot_joint_positions", "robot_root_position", "robot_root_wxyz", "hand_contact_active"]).to_pylist()[0]
    names = list(row["robot_joint_names"])
    qj = np.asarray(row["robot_joint_positions"], dtype=float)
    p = np.asarray(row["robot_root_position"], dtype=float)
    yw = np.unwrap(yaw(np.asarray(row["robot_root_wxyz"], dtype=float)))
    T = p.shape[0]
    v = smooth(np.r_[0.0, np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)] * a.rate, a.win)
    wv = smooth(np.abs(np.r_[0.0, np.diff(yw)]) * a.rate, a.win)
    J = {n: qj[:, names.index(n)] for n in ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_knee_joint", "right_knee_joint")}
    z = np.load(a.npz)
    en = [str(n) for n in (z["error_names"] if "error_names" in z.files else z["tracking_error_names"])]
    err = z["tracking_error"].astype(float)
    med = {k: np.median(err[:, :, en.index(k)], axis=1) for k in ("pelvis_position_m", "pelvis_orientation_rad", "palm_position_m")}
    c = np.asarray(row["hand_contact_active"], dtype=bool)
    t_c = min((int(np.flatnonzero(x)[0]) for x in c if x.any()), default=-1)
    ori = med["pelvis_orientation_rad"]
    f2 = np.flatnonzero(ori > 0.2)
    f5 = np.flatnonzero(ori > 0.5)
    print(f"参考 {a.motion}：{T} 帧，t_c {t_c}；骨盆朝向误差中位第一次 > 0.2 rad 第 {f2[0] if f2.size else '—'} 帧、> 0.5 rad 第 {f5[0] if f5.size else '—'} 帧")
    print("帧\t根速度 m/s\t偏航角速度 rad/s\t骨盆高 m\t腰 yaw/roll/pitch rad\t左右膝 rad\t骨盆位置误差 m\t骨盆朝向误差 rad\t手掌误差 m")
    for t in range(0, min(T, a.to + 1), a.every):
        print(f"{t}\t{v[t]:.2f}\t{wv[t]:.2f}\t{p[t, 2]:.3f}\t{J['waist_yaw_joint'][t]:+.2f}/{J['waist_roll_joint'][t]:+.2f}/{J['waist_pitch_joint'][t]:+.2f}\t"
              f"{J['left_knee_joint'][t]:.2f}/{J['right_knee_joint'][t]:.2f}\t{med['pelvis_position_m'][t]:.3f}\t{ori[t]:.3f}\t{med['palm_position_m'][t]:.3f}")
    print("FALL_TIMELINE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
