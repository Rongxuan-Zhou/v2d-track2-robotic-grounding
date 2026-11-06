"""参考里让机器人较近那只手在物体动之前到位（训练机上经 scripts/dev/in-flash-chord 在容器里跑，只用 CPU）。
计划 agent-doc/plan-t3hand-2026-10-27.md 第 1 节 P1；只改重定向后的 motion parquet，物体轨迹一字不动。

起因：Tier 3 gb5 的 ep16 参考里瓶子第 129 帧开始动，机器人手掌第 132 帧才到 2 cm 内、接触标记第 137 帧，首次接触时瓶子已走 21 cm
（results/t3hand/summary.txt）；重建里人手也晚（CARI4D 原始结果），重定向照搬。策略学到的是“物体自己走”，训练在接触处就把物体碰丢。
做法（每集一次）：
  1. 读参考（同 finger_close.py）：关节名、关节角、根位姿、两只手的接触标记、ee_pose_w；URDF 正向运动学自检（手掌位置最大差 ≤ --fk-tol-m，不过就退出）。
  2. 物体表面：object_mesh_paths 的网格按 object_body_position / object_body_wxyz 放进世界，表面采 --surf 个点（带面法向），
     手掌点（<side>_hand_palm_link 的碰撞网格顶点）到表面的有向距离 = (p − s)·n（外正内负），取各点的最小值。
  3. 窗口：t_move = 物体相对第 0 帧首次平移 > --move-cm 的帧；t_c = 这只手参考里的首次接触帧；w0 = max(0, t_move − --lead)；
     t_reach = w0 起第一个“原参考里手掌距离 ≤ D + --tol-m”的帧（没有就取 t_c）；窗口 [w0, t_reach)。手取“接触开始最早”的那只；
     两只都没有接触标记就退出。t_reach ≤ w0（手本来就先到）也退出，不改。
  4. 目标距离 D：这只手接触段里手掌最小有向距离的中位（为负取 0），可用 --d-target 覆盖。
  5. 逐帧阻尼最小二乘逆运动学：末端取手掌点里离物体最近的那个点 k，目标位置 g = p_k + (d − D) × 单位方向（手掌点 → 表面最近点）；
     雅可比对这只手臂的 7 个关节数值求（步长 1e-4），每步 Δq = (JᵀJ + (λ² + μ)I)⁻¹(Jᵀ(g − p_k) − μ(q − q_原)) × --step，
     每步各关节不超过 2°，关节限位按 URDF 截断；最近点换了就重定目标（最多 5 轮）；
     |d − D| ≤ --tol-m 或到 --iters 步为止。d ≤ D 的帧不动。每帧从“原关节角 + 上一帧的改动量”起算（时间上连续）。
     窗口开头 --ramp 帧把 Δq 按余弦从 0 过渡到 1；窗口末尾原参考已贴到目标附近，改动量自然收到 0，不另做过渡。
  6. 检查（H0，任一不过就不写文件）：过渡区外每帧 |d − D| ≤ --tol-m（d ≤ D 的帧除外）；手臂关节逐帧改动 ≤ --max-dq-deg；
     相邻帧手臂关节转角比原参考多出的部分 ≤ --max-step-deg；窗口外逐值相同。
  7. 写出：新 motion 根 <out_root>/whole_body/soma/sequence_id=<seq>/robot_name=g1/data.parquet 只换 robot_joint_positions（这只手臂的 7 列）、
     ee_pose_w（这只手掌，按 URDF 重算，四元数 wxyz、w ≥ 0）、hand_contact_active（窗口内 d ≤ --contact-cm 的帧补成接触，只增不减）；
     其余列、object 目录、支撑面、v2d_meta.json 照抄（同 finger_close.py）。网格路径换成新目录。目标已存在就拒绝。
穿透与全身间距的检查（计划 H0 的 c、d）用 ref_penetration.py、ref_clearance.py 在输出上另跑。
用法（容器内，仓库根 /repo）：python stages/retarget/hand_reach.py <motion parquet> <out_root> [--meta-src … --meta-dst …] [--dry-run]
最后一行 HAND_REACH_DONE；不写文件时 HAND_REACH_SKIP <原因> 或 HAND_REACH_FAIL <原因>。
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")


def cosine_ramp(n_frames: int, ramp: int) -> np.ndarray:
    """(n_frames,) 权重：两端各 ramp 帧从 0 余弦升到 1，中间 1。窗口比 2×ramp 短时两边各占一半。"""
    w = np.ones(n_frames)
    r = min(ramp, n_frames // 2)
    for i in range(r):
        a = 0.5 - 0.5 * np.cos(np.pi * (i + 1) / (r + 1))
        w[i] = min(w[i], a)
        w[n_frames - 1 - i] = min(w[n_frames - 1 - i], a)
    return w


def pull_frame(fk_point, sdist_dir, q0: np.ndarray, lower: np.ndarray, upper: np.ndarray, d_target: float,
               step: float = 0.5, damping: float = 0.02, tol: float = 0.005, iters: int = 200, mu: float = 0.002,
               max_iter_deg: float = 2.0, q_ref: np.ndarray | None = None) -> tuple:
    """一帧的阻尼最小二乘：fk_point(q, k) 给手掌第 k 个点的世界坐标，sdist_dir(q) 给 (最小有向距离 d, 单位方向 u 从该点指向表面最近点, k)。
    外层最多 5 轮：按当前位姿定最近点 k 与目标位置 g = p_k + (d − D) u；内层把 p_k 拉到 g（位置目标，带向 q_ref 的弱回拉 mu，
    每步各关节不超过 max_iter_deg）。返回 (q, d, 迭代数)。起始 d ≤ D 时原样返回。"""
    q = q0.astype(float).copy()
    q_ref = q0 if q_ref is None else q_ref
    d, u, k = sdist_dir(q)
    if d <= d_target:
        return q, d, 0
    n = 0
    cap = np.radians(max_iter_deg)
    for _outer in range(5):
        d, u, k = sdist_dir(q)
        if abs(d - d_target) <= tol:
            break
        g = fk_point(q, k) + (d - d_target) * u
        for _ in range(iters):
            p0 = fk_point(q, k)
            err = g - p0
            if np.linalg.norm(err) <= tol:
                break
            n += 1
            J = np.zeros((3, len(q)))
            for j in range(len(q)):
                dq = np.zeros(len(q))
                dq[j] = 1e-4
                J[:, j] = (fk_point(q + dq, k) - p0) / 1e-4
            A = J.T @ J + (damping ** 2 + mu) * np.eye(len(q))
            dqv = np.linalg.solve(A, J.T @ err - mu * (q - q_ref)) * step
            dqv = np.clip(dqv, -cap, cap)
            q = np.clip(q + dqv, lower, upper)
    d, u, k = sdist_dir(q)
    return q, d, n


def main() -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import trimesh
    from scipy.spatial import cKDTree

    from finger_close import URDF, Robot, quat_mat
    from hands_off import mat_to_wxyz

    ap = argparse.ArgumentParser()
    ap.add_argument("motion")
    ap.add_argument("out_root")
    ap.add_argument("--urdf", default=URDF)
    ap.add_argument("--move-cm", type=float, default=3.0)
    ap.add_argument("--lead", type=int, default=15)
    ap.add_argument("--ramp", type=int, default=10)
    ap.add_argument("--d-target", type=float, default=None, help="米；不给就取接触段手掌最小有向距离的中位（负取 0）")
    ap.add_argument("--contact-cm", type=float, default=2.0)
    ap.add_argument("--step", type=float, default=0.5)
    ap.add_argument("--damping", type=float, default=0.02)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--tol-m", type=float, default=0.005)
    ap.add_argument("--max-dq-deg", type=float, default=20.0)
    ap.add_argument("--max-step-deg", type=float, default=10.0)
    ap.add_argument("--surf", type=int, default=20000)
    ap.add_argument("--fk-tol-m", type=float, default=0.002)
    ap.add_argument("--meta-src")
    ap.add_argument("--meta-dst")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    motion = Path(a.motion)
    seq_dir = motion.parents[1]
    seq = seq_dir.name.split("=", 1)[1]
    whole = seq_dir.parents[1]
    out_seq = Path(a.out_root) / "whole_body" / "soma" / seq_dir.name
    if out_seq.exists() and not a.dry_run:
        print(f"HAND_REACH_FAIL {out_seq} 已存在，拒绝")
        return 2
    if bool(a.meta_src) != bool(a.meta_dst):
        print("HAND_REACH_FAIL --meta-src 与 --meta-dst 要一起给")
        return 2
    meta_src = Path(a.meta_src) / seq / "v2d_meta.json" if a.meta_src else None
    meta_dst = Path(a.meta_dst) / seq / "v2d_meta.json" if a.meta_dst else None
    if meta_src and not meta_src.is_file():
        print(f"HAND_REACH_FAIL 没有 {meta_src}")
        return 2
    if meta_dst and meta_dst.exists() and not a.dry_run:
        print(f"HAND_REACH_FAIL {meta_dst} 已存在，拒绝")
        return 2

    t = pq.read_table(motion)
    row = t.to_pylist()[0]
    names = list(row["robot_joint_names"])
    qj = np.asarray(row["robot_joint_positions"], dtype=float)
    T = qj.shape[0]
    rob = Robot(a.urdf)
    root_p = np.asarray(row["robot_root_position"], dtype=float)
    root_R = quat_mat(np.asarray(row["robot_root_wxyz"], dtype=float), "wxyz")
    sides = list(row["hand_sides"])
    contact = {s: np.asarray(c, dtype=bool) for s, c in zip(sides, row["hand_contact_active"])}
    ee = np.asarray(row["ee_pose_w"], dtype=float)
    ee_names = list(row["ee_link_names"])
    print(f"参考 {motion}：{T} 帧，{len(names)} 个关节；URDF {a.urdf}")

    # 1. 自检
    palms = {f"{s}_hand_palm_link" for s in sides}
    qd = {n: qj[:, i] for i, n in enumerate(names)}
    fkp = rob.fk(root_p, root_R, qd, palms)
    for s in sides:
        ln = f"{s}_hand_palm_link"
        k = ee_names.index(ln)
        dp = np.linalg.norm(fkp[ln][0] - ee[:, k, :3], axis=1).max()
        print(f"  自检 {ln}：位置最大差 {dp * 1000:.2f} mm")
        if dp > a.fk_tol_m:
            print(f"HAND_REACH_FAIL FK 自检 {ln} 位置差 {dp * 1000:.1f} mm > {a.fk_tol_m * 1000:.0f} mm")
            return 3

    # 2. 物体表面
    mesh = trimesh.load(row["object_mesh_paths"][0], force="mesh")
    pts, fid = trimesh.sample.sample_surface(mesh, a.surf, seed=0)
    nrm = mesh.face_normals[fid]
    tree = cKDTree(pts)
    op = np.asarray(row["object_body_position"], dtype=float)[:, 0]
    oR = quat_mat(np.asarray(row["object_body_wxyz"], dtype=float)[:, 0], "wxyz")

    # 3. 窗口与手
    moved = np.linalg.norm(op - op[0], axis=1) > a.move_cm / 100
    if not moved.any():
        print("HAND_REACH_SKIP 物体整集没有平移超过阈值")
        return 0
    t_move = int(np.argmax(moved))
    firsts = {s: int(np.argmax(c)) for s, c in contact.items() if c.any()}
    if not firsts:
        print("HAND_REACH_SKIP 两只手都没有接触标记")
        return 0
    side = min(firsts, key=firsts.get)
    t_c = firsts[side]
    w0 = max(0, t_move - a.lead)
    arm_idx = [names.index(f"{side}_{j}_joint") for j in ARM]
    lower = np.array([rob.joints[f"{side}_{j}_joint"]["lower"] if rob.joints[f"{side}_{j}_joint"]["lower"] is not None else -np.pi for j in ARM])
    upper = np.array([rob.joints[f"{side}_{j}_joint"]["upper"] if rob.joints[f"{side}_{j}_joint"]["upper"] is not None else np.pi for j in ARM])
    palm = f"{side}_hand_palm_link"
    pv = rob.link_points(palm)

    def palm_pose(q_full_t: np.ndarray, t_: int):
        qd1 = {n: np.array([q_full_t[i]]) for i, n in enumerate(names)}
        p, R = rob.fk(root_p[t_:t_ + 1], root_R[t_:t_ + 1], qd1, {palm})[palm]
        return p[0], R[0]

    def palm_sdist(q_full_t: np.ndarray, t_: int):
        """(最小有向距离, 最近手掌点的世界坐标, 该点指向表面最近点的单位方向, 该点在连杆系的索引)"""
        p, R = palm_pose(q_full_t, t_)
        pw = p + pv @ R.T
        p_obj = (pw - op[t_]) @ oR[t_]
        _, i = tree.query(p_obj)
        d = np.einsum("ij,ij->i", p_obj - pts[i], nrm[i])
        k = int(np.argmin(d))
        s_w = oR[t_] @ pts[i[k]] + op[t_]
        v = s_w - pw[k]
        nv = np.linalg.norm(v)
        u = v / nv if nv > 1e-9 else -(oR[t_] @ nrm[i[k]])
        return float(d[k]), pw[k], u, k

    # 4. 目标距离
    if a.d_target is None:
        cf = np.flatnonzero(contact[side])
        ds = [palm_sdist(qj[t_], t_)[0] for t_ in cf[:: max(1, len(cf) // 60)]]
        d_target = max(0.0, float(np.median(ds)))
    else:
        d_target = a.d_target
    print(f"  目标距离 D {d_target * 1000:.1f} mm")
    t_reach = next((t_ for t_ in range(w0, t_c) if palm_sdist(qj[t_], t_)[0] <= d_target + a.tol_m), t_c)
    if t_reach <= w0:
        print(f"HAND_REACH_SKIP 手本来就先到：到位 {t_reach}，接触 {t_c}，物体动 {t_move}，窗口起点 {w0}")
        return 0
    frames = list(range(w0, t_reach))
    print(f"  手 {side}：接触标记 {t_c} 帧起，原参考第 {t_reach} 帧贴到目标；物体第 {t_move} 帧动；窗口 [{w0}, {t_reach})，{len(frames)} 帧；手掌点 {len(pv)} 个")

    # 5. 逐帧 IK
    new_q = qj.copy()
    before, after, iters_used = [], [], []
    dq_prev = np.zeros(len(arm_idx))
    for t_ in frames:
        q_full = qj[t_].copy()

        def fk_point(qa, k, t_=t_, q_full=q_full):
            qf = q_full.copy()
            qf[arm_idx] = qa
            p, R = palm_pose(qf, t_)
            return p + R @ pv[k]

        def sdist_dir(qa, t_=t_, q_full=q_full):
            qf = q_full.copy()
            qf[arm_idx] = qa
            d, _, u, k = palm_sdist(qf, t_)
            return d, u, k

        d0 = palm_sdist(q_full, t_)[0]
        q_start = np.clip(q_full[arm_idx] + dq_prev, lower, upper)
        qa, d1, n = pull_frame(fk_point, sdist_dir, q_start, lower, upper, d_target, a.step, a.damping, a.tol_m, a.iters, q_ref=q_full[arm_idx])
        if d0 <= d_target:
            qa = q_full[arm_idx]
        new_q[t_, arm_idx] = qa
        dq_prev = qa - q_full[arm_idx]
        before.append(d0)
        after.append(d1)
        iters_used.append(n)
    w = np.ones(len(frames))
    r = min(a.ramp, len(frames))
    w[:r] = cosine_ramp(2 * r, r)[:r]
    for i, t_ in enumerate(frames):
        new_q[t_, arm_idx] = qj[t_, arm_idx] + w[i] * (new_q[t_, arm_idx] - qj[t_, arm_idx])
    final = [palm_sdist(new_q[t_], t_)[0] for t_ in frames]
    print("  帧\t改前mm\tIK后mm\t过渡后mm\t迭代")
    for i, t_ in enumerate(frames):
        print(f"  {t_}\t{before[i] * 1000:.1f}\t{after[i] * 1000:.1f}\t{final[i] * 1000:.1f}\t{iters_used[i]}")

    # 6. 检查
    fails = []
    core = [i for i in range(len(frames)) if w[i] >= 0.999 and before[i] > d_target]
    miss = [frames[i] for i in core if abs(final[i] - d_target) > a.tol_m]
    if miss:
        fails.append(f"过渡区外 {len(miss)} 帧没到目标距离（{miss[:8]}）")
    dq = np.degrees(np.abs(new_q[:, arm_idx] - qj[:, arm_idx]))
    if dq.max() > a.max_dq_deg:
        fails.append(f"手臂关节改动最大 {dq.max():.1f}° > {a.max_dq_deg}°")
    stepdeg = np.degrees(np.abs(np.diff(new_q[:, arm_idx], axis=0))).max(axis=1)
    step0 = np.degrees(np.abs(np.diff(qj[:, arm_idx], axis=0))).max(axis=1)
    big = np.flatnonzero(stepdeg - step0 > a.max_step_deg)
    if len(big):
        fails.append(f"相邻帧手臂转角比原参考多出 > {a.max_step_deg}° 的有 {len(big)} 处（{big[:8].tolist()}）")
    outside = np.ones(T, bool)
    outside[frames] = False
    if not np.array_equal(new_q[outside], qj[outside]):
        fails.append("窗口外的关节角变了")
    print(f"  手臂关节改动最大 {dq.max():.1f}°、相邻帧转角最大 {stepdeg.max():.1f}°（原参考 {step0.max():.1f}°，多出最大 {(stepdeg - step0).max():.1f}°）；改了 {int((dq > 1e-9).any(1).sum())} 帧")
    if fails:
        print("HAND_REACH_FAIL " + "；".join(fails))
        return 4

    # 接触标记：窗口内 d ≤ contact-cm 补成接触
    new_contact = contact[side].copy()
    added = 0
    for i, t_ in enumerate(frames):
        if final[i] <= a.contact_cm / 100 and not new_contact[t_]:
            new_contact[t_] = True
            added += 1
    print(f"  接触标记补了 {added} 帧（{side}）")
    if a.dry_run:
        print("dry-run：不写文件")
        print("HAND_REACH_DONE")
        return 0

    # 7. 写出
    qd_new = {n: new_q[:, i] for i, n in enumerate(names)}
    fk1 = rob.fk(root_p, root_R, qd_new, {palm})
    new_ee = ee.copy()
    k = ee_names.index(palm)
    new_ee[:, k, :3] = fk1[palm][0]
    new_ee[:, k, 3:] = mat_to_wxyz(fk1[palm][1])
    out_seq.mkdir(parents=True)
    shutil.copytree(seq_dir / "object", out_seq / "object")
    (out_seq / "robot_name=g1").mkdir()
    t2 = t
    hc = [list(map(bool, contact[s])) if s != side else list(map(bool, new_contact)) for s in sides]
    for col, val in (("robot_joint_positions", [new_q.tolist()]), ("ee_pose_w", [new_ee.tolist()]), ("hand_contact_active", [hc])):
        t2 = t2.set_column(t2.schema.get_field_index(col), t2.schema.field(col), pa.array(val, type=t2.schema.field(col).type))
    swapped = 0
    for col in ("object_mesh_paths", "object_urdf_paths"):
        if col in t2.column_names:
            old_paths = t2.column(col).to_pylist()[0]
            vals = [[p.replace(str(seq_dir), str(out_seq)) for p in old_paths]]
            swapped += sum(n != o for n, o in zip(vals[0], old_paths))
            t2 = t2.set_column(t2.schema.get_field_index(col), t2.schema.field(col), pa.array(vals, type=t2.schema.field(col).type))
    if swapped == 0:
        shutil.rmtree(out_seq)
        print("HAND_REACH_FAIL 网格路径一条也没换到新目录")
        return 5
    pq.write_table(t2, out_seq / "robot_name=g1" / "data.parquet")
    sup = whole / "reconstructed_stage" / f"{seq}_support.usda"
    if sup.is_file():
        (Path(a.out_root) / "whole_body" / "reconstructed_stage").mkdir(parents=True, exist_ok=True)
        shutil.copy2(sup, Path(a.out_root) / "whole_body" / "reconstructed_stage" / sup.name)
    if meta_src:
        meta_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(meta_src, meta_dst)
    # 读回核对
    back = pq.read_table(out_seq / "robot_name=g1" / "data.parquet").to_pylist()[0]
    bq = np.asarray(back["robot_joint_positions"], dtype=float)
    same = all(np.array_equal(np.asarray(back[c]), np.asarray(row[c])) for c in ("robot_root_position", "robot_root_wxyz", "object_body_position", "object_body_wxyz"))
    if not (np.allclose(bq, new_q) and same):
        print("HAND_REACH_FAIL 读回核对不过")
        return 6
    print(f"  写出 {out_seq}；根位姿与物体轨迹逐值相同")
    print(json.dumps({"side": side, "t_move": t_move, "t_c": t_c, "window": [w0, t_c], "d_target_mm": round(d_target * 1000, 1),
                      "max_dq_deg": round(float(dq.max()), 2), "contact_added": added}))
    print("HAND_REACH_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
