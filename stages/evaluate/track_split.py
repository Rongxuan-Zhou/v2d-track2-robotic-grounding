"""零残差回放跟丢参考时，骨盆偏差落在哪个方向（训练机上经 scripts/dev/in-flash-chord 在容器里跑，只用 CPU；读 diag_rollout.py --bodies 存的 npz；只报不判）。
计划 agent-doc/plan-t12ref-2026-10-20.md 执行记录之后的诊断。参考骨盆与躯干用 URDF 正向运动学算（stages/retarget/finger_close.py 的 Robot，同 fall_mechanism.py）。
逐帧逐世界：骨盆水平偏差（仿真 − 参考）投到参考骨盆水平速度方向上（沿向，负 = 落后）与它的左法向上（横向，正 = 偏左）；
躯干朝向的偏航差（仿真 − 参考，包到 ±π）。另报参考的水平速度、偏航角速度（按 --rate 帧/秒回放，--win 帧窗口平均）。每 --every 帧一行，取世界中位。
用法（容器内）：python /repo/stages/evaluate/track_split.py --npz <npz> --motion <parquet> [--to 120] [--every 5] [--rate 50] [--win 10]   最后一行 TRACK_SPLIT_DONE
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stages" / "retarget"))
from finger_close import URDF, Robot, quat_mat


def smooth(x: np.ndarray, w: int) -> np.ndarray:
    return np.convolve(x, np.ones(w) / w, mode="same") if w > 1 else x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--motion", required=True)
    ap.add_argument("--to", type=int, default=120)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--win", type=int, default=10)
    ap.add_argument("--urdf", default=URDF)
    a = ap.parse_args()
    z = np.load(a.npz)
    names = [str(n) for n in z["robot_body_names"]]
    ip, it = names.index("pelvis"), names.index("torso_link")
    bad = np.asarray(z["non_finite_worlds"], dtype=int)
    bp = z["robot_body_pose"].astype(float)
    bp = bp[:, np.setdiff1d(np.arange(bp.shape[1]), bad)]
    T = bp.shape[0]
    s0 = int(z["start_frame"])
    row = pq.read_table(a.motion, columns=["robot_joint_names", "robot_joint_positions", "robot_root_position", "robot_root_wxyz"]).to_pylist()[0]
    jn = list(row["robot_joint_names"])
    qj = np.asarray(row["robot_joint_positions"], dtype=float)
    L = Robot(a.urdf).fk(np.asarray(row["robot_root_position"], dtype=float), quat_mat(np.asarray(row["robot_root_wxyz"], dtype=float), "wxyz"),
                         {n: qj[:, i] for i, n in enumerate(jn)}, {"pelvis", "torso_link"})
    pr = L["pelvis"][0][s0:s0 + T]                                     # (T,3)
    Rr = L["torso_link"][1][s0:s0 + T]
    v = np.gradient(pr[:, :2], axis=0) * a.rate                        # (T,2) m/s
    spd = np.linalg.norm(v, axis=1)
    fwd = v / np.maximum(spd[:, None], 1e-6)
    left = np.stack([-fwd[:, 1], fwd[:, 0]], -1)
    yaw_r = np.unwrap(np.arctan2(Rr[:, 1, 0], Rr[:, 0, 0]))
    yr = smooth(np.gradient(yaw_r) * a.rate, a.win)
    spd = smooth(spd, a.win)
    d = bp[:, :, ip, :2] - pr[:, None, :2]                             # (T,W,2)
    along = np.einsum("twi,ti->tw", d, fwd)
    cross = np.einsum("twi,ti->tw", d, left)
    Rs = quat_mat(bp[:, :, it, 3:], "xyzw")
    yaw_s = np.arctan2(Rs[..., 1, 0], Rs[..., 0, 0])
    dyaw = (yaw_s - yaw_r[:, None] + np.pi) % (2 * np.pi) - np.pi
    print(f"{a.npz}：{T} 帧、{bp.shape[1]} 个世界（剔除非有限 {len(bad)} 个）；参考起始帧 {s0}；速度按 {a.rate:g} 帧/秒")
    print("帧\t参考速度 m/s\t参考偏航角速度 rad/s\t沿向偏差 m（负=落后）\t横向偏差 m（正=偏左）\t偏航差 rad\t|偏差| m")
    for t in range(0, min(a.to, T), a.every):
        print(f"{s0 + t}\t{spd[t]:.2f}\t{yr[t]:+.2f}\t{np.median(along[t]):+.3f}\t{np.median(cross[t]):+.3f}\t{np.median(dyaw[t]):+.3f}\t"
              f"{np.median(np.linalg.norm(d[t], axis=-1)):.3f}")
    print("TRACK_SPLIT_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
