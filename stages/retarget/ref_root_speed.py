"""一个档位全部参考里机器人根（骨盆）要走多快（训练机上经 scripts/dev/in-flash-chord 在容器里跑，只用 numpy 与 pyarrow；只读，只报不判）。
起因：agent-doc/plan-t2dg-2026-10-20.md 执行记录：Tier 2 ep4 的骨盆误差第 60 帧 0.12 m、第 100 帧 0.56 m，机器人在接触物体之前就没跟上参考。
FlashCHORD 的 G1 配方逐帧回放（source_frame_playback: true，仿真 50 Hz、一帧源数据一个控制步），30 fps 的动作按 --rate（默认 50）帧/秒放。
每集一行：序列、帧、按回放速率算的根水平速度在 --win 帧滑动窗口里平均后的最大值（m/s）与出现的帧、整段水平路程（m）、偏航角速度的同样最大值（rad/s）、
参考第一次接触之前的平均水平速度（m/s）、骨盆高度的最小值与出现的帧、第 0 帧物体中心高度。用法（容器内）：python /repo/stages/retarget/ref_root_speed.py <档位 motion 根> [--rate 50] [--win 25]   最后一行 REF_ROOT_SPEED_DONE
"""
import argparse
import glob
import sys

import numpy as np
import pyarrow.parquet as pq


def yaw(wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = (wxyz[:, i] for i in range(4))
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--win", type=int, default=25)
    a = ap.parse_args()
    files = sorted(glob.glob(f"{a.root}/whole_body/soma/sequence_id=*/robot_name=g1/data.parquet"))
    print(f"回放速率 {a.rate} 帧/秒，窗口 {a.win} 帧")
    print("序列\t帧\t最大水平速度 m/s\t出现在第几帧\t整段水平路程 m\t最大偏航角速度 rad/s\t接触前平均水平速度 m/s\t第一次接触帧\t骨盆最低 m\t出现在第几帧\t第 0 帧物体高 m")
    ker = np.ones(a.win) / a.win
    for f in files:
        row = pq.read_table(f, columns=["sequence_id", "robot_root_position", "robot_root_wxyz", "hand_contact_active", "object_body_position"]).to_pylist()[0]
        p = np.asarray(row["robot_root_position"], dtype=float)
        q = np.asarray(row["robot_root_wxyz"], dtype=float)
        T = p.shape[0]
        step = np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)
        v = np.convolve(step * a.rate, ker, mode="valid") if T > a.win else step * a.rate
        dyaw = np.abs(np.angle(np.exp(1j * np.diff(np.unwrap(yaw(q))))))
        wv = np.convolve(dyaw * a.rate, ker, mode="valid") if T > a.win else dyaw * a.rate
        c = np.asarray(row["hand_contact_active"], dtype=bool)
        tc = min((int(np.flatnonzero(x)[0]) for x in c if x.any()), default=T - 1)
        pre = step[:max(tc, 1)].sum() / (max(tc, 1) / a.rate)
        oz = float(np.asarray(row["object_body_position"], dtype=float)[0, 0, 2])
        print(f"{row['sequence_id']}\t{T}\t{v.max():.2f}\t{int(v.argmax()) + a.win // 2}\t{step.sum():.2f}\t{wv.max():.2f}\t{pre:.2f}\t{tc}\t"
              f"{p[:, 2].min():.3f}\t{int(p[:, 2].argmin())}\t{oz:.3f}")
    print("REF_ROOT_SPEED_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
