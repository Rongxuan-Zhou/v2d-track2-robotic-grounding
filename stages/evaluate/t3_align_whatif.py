"""Tier 3 第 0 帧朝向错了的集：各对齐方案下全程误差的推算（训练机上经 scripts/dev/in-flash-chord 在容器里跑，只用 CPU；要 numpy、pyarrow）。
初步验证用，只报不判；计划另立。真值只用 Tier 1 第 0 帧（tier1_frame0.py），只用于算对齐。

推算的前提（假设 H-body，量不出来真假）：重建的错只是物体系里一个固定的转动 E，人体与世界是对的。于是按人体对齐（yaw_hum；人体不可用时 pos）之后，
真值的第 t 帧 = 我们第 t 帧右乘 E（绕质心），再整体平移使第 0 帧原点落在 Tier 1 第 0 帧。E 由第 0 帧定：E = (R_H R̂₀)ᵀ R₁。这个"真值"只是假设下的推算，不是 Tier 1 的轨迹。
对每个方案（yaw_hum、yaw_obj、obj、pos，加对称群文件里的近对称候选右乘后再 yaw_hum），逐帧算我们对齐后的位姿与推算真值的差：
位置差、朝向差、物体系 6 个 ±5 cm 点的平均距离（MPPE 的量法）；报全程均值与 SP-SR（均值 ≤ 10 cm 且 ≤ 0.5 rad）、MP-SR（< 3 cm 且 < 30°）是否过，
另报物体离第 0 帧位置的最大位移与全程转动的最大角（看"原地转"成不成立）。
用法：python t3_align_whatif.py <帧参考目录> --tier1 <Tier 1 数据根> --t3 <runs/tier3> --sym <对称群 json> [--episodes 12,14]   最后一行 ALIGN_WHATIF_DONE n。
"""
import argparse
import json
from pathlib import Path

import numpy as np

import align_frefs_t3 as al
import t3_frame0_table as ft
import tier1_frame0

PTS = np.array([[s * 0.05 if i == k else 0.0 for i in range(3)] for k in range(3) for s in (1, -1)])


def poses_R(P):
    return np.stack([al.q_to_R(q) for q in P[:, 3:]])


def errors(P_ours, P_true):
    """逐帧（位置差 m、朝向差 rad、6 点平均距离 m）。"""
    Ro, Rt = poses_R(P_ours), poses_R(P_true)
    dp = np.linalg.norm(P_ours[:, :3] - P_true[:, :3], axis=1)
    dr = np.array([np.arccos(np.clip((np.trace(a.T @ b) - 1) / 2, -1, 1)) for a, b in zip(Ro, Rt)])
    qo = P_ours[:, None, :3] + np.einsum("tij,kj->tki", Ro, PTS)
    qt = P_true[:, None, :3] + np.einsum("tij,kj->tki", Rt, PTS)
    return dp, dr, np.linalg.norm(qo - qt, axis=2).mean(1)


def verdict(dp, dr, d6):
    sp = dp.mean() <= 0.10 and dr.mean() <= 0.5
    mp = dp.mean() < 0.03 and np.degrees(dr.mean()) < 30
    return f"位置 {dp.mean() * 100:.1f} cm、朝向 {np.degrees(dr.mean()):.1f}°、6 点 {d6.mean() * 100:.1f} cm，SP-SR {'过' if sp else '不过'}、MP-SR {'过' if mp else '不过'}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("frefs", type=Path)
    ap.add_argument("--tier1", type=Path, required=True)
    ap.add_argument("--t3", type=Path, required=True)
    ap.add_argument("--sym", type=Path, required=True)
    ap.add_argument("--episodes", default="")
    a = ap.parse_args()
    sym = json.loads(a.sym.read_text())["episodes"]
    rows = ft.read_rows(a.frefs)
    only = {int(x) for x in a.episodes.split(",") if x.strip()}
    n = 0
    for ep in sorted(rows):
        if only and ep not in only:
            continue
        r = rows[ep]
        P = ft.poses(a.frefs, r)
        truth = tier1_frame0.read(a.tier1, ep)
        nrm, human, _, seq_root = al.our_frame0(a.t3, r)
        if truth["sequence_id"] != r["sequence"] or seq_root != r["sequence"]:
            raise SystemExit(f"ep{ep}：序列对不上")
        s = sym[str(ep)]
        com_b = np.asarray(s["com_b"])
        R1 = al.q_to_R(truth["q_xyzw"])
        fixed = "yaw_hum" if human is not None else "pos"
        R_H, t_H, _ = al.align_episode(P, truth, nrm, human, fixed)
        E = (R_H @ al.q_to_R(P[0, 3:])).T @ R1
        true_P = ft.right_mul(al.apply(P, R_H, t_H), E, com_b)
        true_P[:, :3] += truth["p"] - true_P[0, :3]   # 推算真值的第 0 帧原点落在 Tier 1 第 0 帧（对齐时我们的原点也对到那里）
        com_path = P[:, :3] + np.einsum("tij,j->ti", poses_R(P), com_b)
        disp = float(np.linalg.norm(com_path - com_path[0], axis=1).max())
        R0 = al.q_to_R(P[0, 3:])
        turn = float(max(al.angle_deg(R0.T @ al.q_to_R(q)) for q in P[:, 3:]))
        _, _, rep = al.align_episode(P, truth, nrm, human, "auto")
        print(f"ep{ep}（{r['tier_dir']}，{len(P)} 帧，自动方案 {rep['choice']}）：E 转角 {al.angle_deg(E):.1f}°；质心离第 0 帧最远 {disp * 100:.0f} cm，"
              f"相对第 0 帧最大转角 {turn:.0f}°（假设 H-body：真值 = 我们 · E，按 {fixed} 对齐）")
        for v in ("yaw_hum", "yaw_obj", "obj", "pos"):
            if v == "yaw_hum" and human is None:
                continue
            R_T, t_T, _ = al.align_episode(P, truth, nrm, human, v)
            print(f"  {v:<8} {verdict(*errors(al.apply(P, R_T, t_T), true_P))}")
        axes = np.asarray(s["principal_axes"])
        best = min(s["near"], key=lambda x: ft.frame0_diff(ft.right_mul(P[:1], ft.rot(axes[x["axis"]], x["deg"]), com_b), R_H, R1))
        g = ft.rot(axes[best["axis"]], best["deg"])
        P2 = ft.right_mul(P, g, com_b)
        R_T2, t_T2, _ = al.align_episode(P2, truth, nrm, human, fixed)
        print(f"  近对称 g（第 {best['axis']} 主轴 {best['deg']}°，表面最多移动 {best['shift_m'] * 100:.1f} cm）后 {fixed}：{verdict(*errors(al.apply(P2, R_T2, t_T2), true_P))}")
        n += 1
    print(f"ALIGN_WHATIF_DONE {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
