"""改过的参考有没有真的进到仿真里（训练机上经 scripts/dev/in-flash-chord 在容器里跑，只用 CPU；计划 agent-doc/plan-t12ref-2026-10-20.md 的 Q0b）。
读两份 diag_rollout.py --bodies 存的 npz（同一个 checkpoint、同样世界数，一份用原参考 A，一份用改过的参考 B）与两份参考 parquet。
t0 = 两份参考的 robot_joint_positions 之差第一次 > --t0-thr（默认 1e-9 rad）的帧。比同一世界编号在两份 rollout 里的躯干朝向（四元数 xyzw）夹角，逐帧取世界中位：
  对照段 [s0, t0)：两份参考相同，夹角应接近 0（这一段的最大值 = 两次运行本身的差，含仿真的不确定性）；
  检验段 [t0, t0 + --window)：参考不同，夹角最大值要 > max(--min-rad, 5 × 对照段最大值)，否则判“改动没到仿真”。
用法（容器内）：python /repo/stages/evaluate/replay_diff.py --npz-a <原> --motion-a <原 parquet> --npz-b <改> --motion-b <改 parquet> [--window 20] [--min-rad 0.01] [--t0-thr 1e-9]
最后一行 REPLAY_DIFF PASS|FAIL
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stages" / "retarget"))
from finger_close import quat_mat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz-a", required=True)
    ap.add_argument("--motion-a", required=True)
    ap.add_argument("--npz-b", required=True)
    ap.add_argument("--motion-b", required=True)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--t0-thr", type=float, default=1e-9, help="t0 = 两份参考关节角之差第一次 > 这个值（rad）的帧")
    ap.add_argument("--min-rad", type=float, default=0.01)
    a = ap.parse_args()
    za, zb = np.load(a.npz_a), np.load(a.npz_b)
    for z, n in ((za, a.npz_a), (zb, a.npz_b)):
        if "robot_body_pose" not in z.files or "torso_link" not in [str(x) for x in z["robot_body_names"]]:
            print(f"{n} 没有 robot_body_pose 或没有 torso_link（要 diag_rollout.py --bodies）")
            return 2
    s0 = int(za["start_frame"])
    if int(zb["start_frame"]) != s0 or za["robot_body_pose"].shape != zb["robot_body_pose"].shape:
        print(f"两份 rollout 的起始帧或形状不同：{s0} vs {int(zb['start_frame'])}，{za['robot_body_pose'].shape} vs {zb['robot_body_pose'].shape}")
        return 2
    qa = np.asarray(pq.read_table(a.motion_a, columns=["robot_joint_positions"]).to_pylist()[0]["robot_joint_positions"], float)
    qb = np.asarray(pq.read_table(a.motion_b, columns=["robot_joint_positions"]).to_pylist()[0]["robot_joint_positions"], float)
    diff = np.flatnonzero(np.abs(qa - qb).max(axis=1) > a.t0_thr)
    if qa.shape != qb.shape or diff.size == 0:
        print(f"两份参考的关节角形状不同或完全相同：{qa.shape} vs {qb.shape}，不同的帧 {diff.size} 个")
        return 2
    t0 = int(diff[0])
    k = [str(x) for x in za["robot_body_names"]].index("torso_link")
    bad = np.union1d(za["non_finite_worlds"], zb["non_finite_worlds"]).astype(int)
    keep = np.setdiff1d(np.arange(za["robot_body_pose"].shape[1]), bad)
    Ra = quat_mat(za["robot_body_pose"][:, keep, k, 3:].astype(float), "xyzw")
    Rb = quat_mat(zb["robot_body_pose"][:, keep, k, 3:].astype(float), "xyzw")
    ang = np.median(np.arccos(np.clip((np.einsum("twij,twij->tw", Ra, Rb) - 1) / 2, -1, 1)), axis=1)   # (T,) 世界中位
    i0 = t0 - s0
    T = ang.shape[0]
    if not 0 < i0 < T:
        print(f"参考第一次不同的帧 {t0} 不在 rollout 的范围（起始帧 {s0}、{T} 步）里")
        return 2
    ctrl = float(ang[:i0].max())
    test = ang[i0:i0 + a.window]
    thr = max(a.min_rad, 5 * ctrl)
    ok = float(test.max()) > thr
    print(f"A {a.npz_a}\nB {a.npz_b}\n  参考第一次不同的帧 t0 = {t0}（起始帧 {s0}）；比了 {len(keep)} 个世界（剔除非有限 {len(bad)} 个）")
    print(f"  躯干朝向夹角（世界中位）：对照段 [{s0}, {t0}) 最大 {ctrl:.5f} rad；检验段 [{t0}, {t0 + len(test)}) 最大 {float(test.max()):.5f} rad，"
          f"门槛 {thr:.5f} → {'PASS' if ok else 'FAIL（改动没到仿真）'}")
    marks = [m for m in (0, 10, 20, 40, 60, 90) if i0 + m < T and m < max(a.window, 61)]
    print("  t0 之后第 " + "、".join(str(m) for m in marks) + " 帧：" + "、".join(f"{ang[i0 + m]:.4f}" for m in marks) + " rad")
    print(f"REPLAY_DIFF {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
