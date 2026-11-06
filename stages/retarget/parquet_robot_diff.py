"""Two motion parquets of the same episode: is the robot motion identical? (container, CPU, read only; report plus a verdict)

Plan: agent-doc/plan-t3b2n12-2026-10-28.md v1.2, H0 (a'): after a B2n turn of the object, the retarget should see only the
human. Compares robot_joint_positions and robot_root_position frame by frame (max abs difference, registered <= 1e-6),
robot_root_wxyz, the per-hand count of frames with hand_contact_active (registered within +-5 frames each), and the
median IK error per frame (ik_error_per_frame, registered equal to 1e-6 relative).

    python /repo/stages/retarget/parquet_robot_diff.py <old parquet> <new parquet>
Last stdout line PARQUET_ROBOT_SAME or PARQUET_ROBOT_DIFF <reasons>.
"""
import sys

import numpy as np
import pyarrow.parquet as pq

COLS = ["robot_joint_positions", "robot_root_position", "robot_root_wxyz", "hand_contact_active", "hand_sides", "ik_error_per_frame"]


def load(path):
    r = pq.read_table(path, columns=COLS).to_pylist()[0]
    return {k: (np.asarray(r[k], dtype=float) if k not in ("hand_sides",) else list(r[k])) for k in COLS}


def main() -> int:
    a, b = load(sys.argv[1]), load(sys.argv[2])
    bad = []
    for k in ("robot_joint_positions", "robot_root_position", "robot_root_wxyz"):
        if a[k].shape != b[k].shape:
            bad.append(f"{k} 形状 {a[k].shape} 对 {b[k].shape}")
            continue
        d = float(np.abs(a[k] - b[k]).max())
        print(f"{k}：形状 {a[k].shape}，最大差 {d:.3e}")
        if d > 1e-6:
            bad.append(f"{k} 最大差 {d:.3e}")
    ca, cb = a["hand_contact_active"], b["hand_contact_active"]
    ca = ca.reshape(len(ca), -1) if ca.ndim > 1 else ca.reshape(-1, len(a["hand_sides"]))
    cb = cb.reshape(len(cb), -1) if cb.ndim > 1 else cb.reshape(-1, len(b["hand_sides"]))
    for i, side in enumerate(a["hand_sides"]):
        na, nb = int(ca[:, i].astype(bool).sum()), int(cb[:, i].astype(bool).sum())
        print(f"{side} 手接触帧：旧 {na}、新 {nb}（差 {nb - na}）")
        if abs(nb - na) > 5:
            bad.append(f"{side} 手接触帧差 {nb - na}")
    ma, mb = float(np.median(a["ik_error_per_frame"])), float(np.median(b["ik_error_per_frame"]))
    print(f"IK 误差中位：旧 {ma:.6g}、新 {mb:.6g}")
    if abs(ma - mb) > 1e-6 * max(1.0, abs(ma)):
        bad.append(f"IK 误差中位 {ma:.6g} → {mb:.6g}")
    print("PARQUET_ROBOT_SAME" if not bad else "PARQUET_ROBOT_DIFF " + "；".join(bad))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
