"""机器人怎么摔的：摔之前哪一项先坏（训练机上经 scripts/dev/in-flash-chord 在容器里跑，只用 CPU；读 diag_rollout.py --bodies 存的 npz；只报不判）。
计划 agent-doc/plan-t12fall-2026-10-20.md 第 2 版（判据写在计划里）。输入：npz（robot_body_pose (T,W,4,7)：pelvis、torso_link、left/right_ankle_roll_link 的位置 + 四元数 xyzw）
与同一集的参考 parquet（参考的同四个刚体用 URDF 正向运动学算，stages/retarget/finger_close.py 的 Robot）。非有限世界先剔除（报剔了几个）。
闸 G0（第 2 版，复核 MAJOR 3）：第 0 步四个刚体与参考第 start_frame 帧的位置差（世界中位）都 < 2 cm、躯干朝向差 < 0.1 rad，不过就退出码 3、不判。
逐世界逐帧：躯干倾斜误差 = 仿真躯干 z 轴与参考躯干 z 轴的夹角（第 2 版，复核 MAJOR 1；另报对世界 z 轴的绝对倾角）、骨盆与参考骨盆的水平距离、
两只脚与参考同侧脚的水平距离（取较大者）、较低的那只脚离地高度（地面 = 这个世界里两只脚踝出现过的最低高度，复核 MINOR 4）。
摔倒帧 t_f = 躯干倾斜误差第一次 > --fall-rad（默认 0.6 rad）。四项门槛：躯干先歪（倾斜误差 > 0.3 rad）、骨盆偏（> 0.15 m）、脚踩偏（> 0.15 m）、
双脚离地（较低的脚离地 > 0.05 m 连续 3 帧，取起始帧）。每个摔了的世界里比四项第一次越过门槛的帧：最早的一项“先坏”（与第二早的相差 ≤ 3 帧记混合），
并且要早于 t_f 至少 10 帧才算。报各项在摔了的世界里“先坏”的比例；另报各项在全部世界里越过门槛的比例（对照集用它做 F1 的闸）。
--before N：另报摔倒帧早于参考第 N 帧的世界比例（机读行 FALL_BEFORE N 比例），以及其中摔时物体离起点 < 0.05 m 的比例（FALL_BEFORE_STILL N 比例）。
--save <npz>（计划 agent-doc/plan-trajopt-2026-10-26.md 第 5 节第 3 版，复核 MAJOR 1）：另存每个世界的 t_f（--fall-rad）与 t_03（倾斜误差第一次 > 0.3 rad），世界编号是 rollout 的原编号（worlds），没摔记 −1。
用法（容器内）：python /repo/stages/evaluate/fall_mechanism.py --npz <npz> --motion <parquet> [--label 名] [--marks 0 50 100 ...] [--before N] [--save <npz>]   最后一行 FALL_MECHANISM_DONE
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "stages" / "retarget"))
from finger_close import URDF, Robot, quat_mat

NAMES = ("pelvis", "torso_link", "left_ankle_roll_link", "right_ankle_roll_link")
ITEMS = ("躯干先歪", "骨盆偏", "脚踩偏", "双脚离地")


def first_over(x: np.ndarray, thr: float, run: int = 1) -> np.ndarray:
    """x (T,W) → 每个世界第一次连续 run 帧 > thr 的起始帧，没有为 -1。"""
    over = x > thr
    if run > 1:
        T = over.shape[0]
        c = np.cumsum(np.vstack([np.zeros((1, over.shape[1]), int), over.astype(int)]), axis=0)
        full = np.zeros_like(over)
        full[: T - run + 1] = (c[run:] - c[:-run]) >= run
        over = full
    return np.where(over.any(0), over.argmax(0), -1)


def q5(x: np.ndarray) -> str:
    return " / ".join(f"{np.percentile(x, p):.0f}" for p in (10, 25, 50, 75, 90)) if x.size else "（无）"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--motion", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--fall-rad", type=float, default=0.6)
    ap.add_argument("--marks", type=int, nargs="*", default=[0, 25, 50, 75, 100, 125, 150, 200, 300])
    ap.add_argument("--before", type=int, help="另报摔倒帧早于这一帧（参考帧号）与不早于它的世界比例（计划 agent-doc/plan-t12ref-2026-10-20.md 的 Q1）")
    ap.add_argument("--urdf", default=URDF)
    ap.add_argument("--save", help="另存每个世界的 t_f、t_03 与世界编号（计划 agent-doc/plan-trajopt-2026-10-26.md 第 5 节）")
    a = ap.parse_args()
    z = np.load(a.npz)
    if "robot_body_pose" not in z.files:
        print("npz 里没有 robot_body_pose（要 diag_rollout.py --bodies）")
        return 2
    names = [str(n) for n in z["robot_body_names"]]
    if tuple(names) != NAMES:
        print(f"刚体名 {names} 与期望 {NAMES} 不同")
        return 2
    bp = z["robot_body_pose"].astype(float)                     # (T,W,4,7)
    bad = np.asarray(z["non_finite_worlds"], dtype=int) if "non_finite_worlds" in z.files else np.zeros(0, int)
    keep = np.setdiff1d(np.arange(bp.shape[1]), bad)
    bp = bp[:, keep]
    T, W = bp.shape[:2]
    row = pq.read_table(a.motion, columns=["robot_joint_names", "robot_joint_positions", "robot_root_position", "robot_root_wxyz"]).to_pylist()[0]
    jn = list(row["robot_joint_names"])
    qj = np.asarray(row["robot_joint_positions"], dtype=float)
    s0 = int(z["start_frame"]) if "start_frame" in z.files else 0
    rob = Robot(a.urdf)
    L = rob.fk(np.asarray(row["robot_root_position"], dtype=float), quat_mat(np.asarray(row["robot_root_wxyz"], dtype=float), "wxyz"),
               {n: qj[:, i] for i, n in enumerate(jn)}, set(NAMES))
    ref = {n: L[n][0][s0:s0 + T] for n in NAMES}                # (T,3)
    Rref_t = L["torso_link"][1][s0:s0 + T]                      # (T,3,3)
    pos = {n: bp[:, :, k, :3] for k, n in enumerate(NAMES)}
    Rt = quat_mat(bp[:, :, 1, 3:], "xyzw")                       # 仿真躯干 (T,W,3,3)
    print(f"{a.label} {a.npz}：{T} 帧、{W} 个世界（剔除非有限世界 {len(bad)} 个）；参考起始帧 {s0}")
    # G0：第 0 步的刚体位置与躯干朝向要对上参考
    gaps = {n: float(np.median(np.linalg.norm(pos[n][0] - ref[n][0], axis=-1))) for n in NAMES}
    rot0 = float(np.median(np.arccos(np.clip((np.einsum("wij,ij->w", Rt[0], Rref_t[0]) - 1) / 2, -1, 1))))
    ok = all(v < 0.02 for v in gaps.values()) and rot0 < 0.1
    print("  G0 第 0 步：位置差（世界中位）" + "，".join(f"{n} {v * 100:.2f} cm" for n, v in gaps.items()) + f"；躯干朝向差 {rot0:.3f} rad → {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("G0 不过：刚体对不上参考（名字、四元数顺序、左右或帧号），不判")
        return 3
    tilt = np.arccos(np.clip(np.einsum("twi,ti->tw", Rt[..., :, 2], Rref_t[:, :, 2]), -1, 1))   # 仿真躯干 z 与参考躯干 z 的夹角
    tilt_abs = np.arccos(np.clip(Rt[..., 2, 2], -1, 1))
    pel_d = np.linalg.norm(pos["pelvis"][..., :2] - ref["pelvis"][:, None, :2], axis=-1)
    foot_d = np.maximum(*(np.linalg.norm(pos[f"{s}_ankle_roll_link"][..., :2] - ref[f"{s}_ankle_roll_link"][:, None, :2], axis=-1) for s in ("left", "right")))
    low = np.minimum(pos["left_ankle_roll_link"][..., 2], pos["right_ankle_roll_link"][..., 2])   # (T,W)
    foot_lift = low - low.min(axis=0, keepdims=True)
    t_f = first_over(tilt, a.fall_rad)
    fell = t_f >= 0
    if a.save:
        np.savez(a.save, worlds=keep.astype(np.int64), t_f=t_f.astype(np.int64), t_03=first_over(tilt, 0.3).astype(np.int64), start_frame=np.int64(s0))
    print(f"  摔倒（躯干倾斜误差 > {a.fall_rad} rad）的世界 {int(fell.sum())}/{W}（{fell.mean():.1%}），摔倒帧分位 10/25/50/75/90 {q5(t_f[fell].astype(float))}")
    if a.before is not None:   # t_f 是 rollout 的步号，参考帧号 = s0 + t_f
        early = fell & (s0 + t_f < a.before)
        late = fell & ~early
        print(f"  第 {a.before} 帧之前摔倒的世界 {int(early.sum())}/{W}（{early.mean():.1%}）；第 {a.before} 帧及以后摔倒 {int(late.sum())}/{W}（{late.mean():.1%}），"
              f"摔倒帧分位 10/25/50/75/90 {q5((s0 + t_f[late]).astype(float))}")
        print(f"FALL_BEFORE {a.before} {early.mean():.6f}")
        if "achieved_pose_w" in z.files:   # 计划 agent-doc/plan-t12ctl-2026-10-20.md 第 2 版 C1：只算摔时物体离起点 < 0.05 m 的（排除先碰物体再摔）
            ob = z["achieved_pose_w"][:, keep, 0, :3].astype(float)
            we = np.flatnonzero(early)
            still = np.zeros(W, bool)
            still[we] = np.linalg.norm(ob[t_f[we], we] - ob[0, we], axis=-1) < 0.05
            print(f"  其中摔时物体离起点 < 0.05 m 的 {int(still.sum())}/{W}（{still.mean():.1%}）")
            print(f"FALL_BEFORE_STILL {a.before} {still.mean():.6f}")
    if fell.any() and "achieved_pose_w" in z.files:   # 计划 agent-doc/plan-t12replay-2026-10-20.md 第 2 版 R1(a)：摔倒那一帧物体离起点多远
        obj = z["achieved_pose_w"][:, keep, 0, :3].astype(float)
        wf = np.flatnonzero(fell)
        d = np.linalg.norm(obj[t_f[wf], wf] - obj[0, wf], axis=-1)
        print(f"  摔倒那一帧物体离起点 < 0.05 m 的世界（摔了的世界里）{(d < 0.05).mean():.1%}；离起点分位 10/50/90 "
              + " / ".join(f"{np.percentile(d, p):.3f}" for p in (10, 50, 90)) + " m")
    firsts = {"躯干先歪": first_over(tilt, 0.3), "骨盆偏": first_over(pel_d, 0.15), "脚踩偏": first_over(foot_d, 0.15), "双脚离地": first_over(foot_lift, 0.05, 3)}
    print("  各项在全部世界里越过门槛的比例（对照集用作 F1 的闸）：" + "，".join(f"{k} {(v >= 0).mean():.1%}" for k, v in firsts.items()))
    if fell.any():
        F = np.stack([firsts[k][fell] for k in ITEMS], 1).astype(float)          # (Wf,4)
        F[F < 0] = np.inf
        order = np.argsort(F, axis=1)
        best, second = F[np.arange(len(F)), order[:, 0]], F[np.arange(len(F)), order[:, 1]]
        early = best <= (t_f[fell] - 10)
        mixed = (second - best) <= 3
        win = np.where(early & ~mixed, order[:, 0], -1)
        nf = int(fell.sum())
        print("  摔了的世界里“先坏”的一项（早于摔倒 ≥ 10 帧、领先第二早 > 3 帧）：" + "，".join(f"{k} {(win == i).mean():.1%}" for i, k in enumerate(ITEMS))
              + f"；混合 {(early & mixed).mean():.1%}；没有一项早于摔倒 10 帧 {(~early).mean():.1%}（共 {nf} 个）")
        for i, k in enumerate(ITEMS):
            f = firsts[k][fell]
            have = f >= 0
            rel = (f[have] - t_f[fell][have]).astype(float)
            print(f"    {k}：摔了的世界里出现过 {int(have.sum())}/{nf}；相对摔倒帧 分位 10/25/50/75/90 {q5(rel)}")
    print("  帧\t躯干倾斜误差 rad（绝对倾角）\t骨盆高 m（参考）\t骨盆偏 m\t脚踩偏 m\t较低的脚离地 m（世界中位）")
    for t in [m for m in a.marks if m < T]:
        print(f"  {t}\t{np.median(tilt[t]):.3f}（{np.median(tilt_abs[t]):.3f}）\t{np.median(pos['pelvis'][t, :, 2]):.3f}（{ref['pelvis'][t, 2]:.3f}）\t"
              f"{np.median(pel_d[t]):.3f}\t{np.median(foot_d[t]):.3f}\t{np.median(foot_lift[t]):.3f}")
    print("FALL_MECHANISM_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
