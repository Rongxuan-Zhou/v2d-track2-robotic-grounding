"""参考里让机器人不去够物体（训练机上经 scripts/dev/in-flash-chord 在容器里跑，只用 CPU）。
起因：agent-doc/plan-t12lift-2026-10-20.md：Tier 2 ep25 训好的策略把瓶子弄到地上、4096 世界 MPPE 117.09 cm，而物体一直不动时只有 49.51 cm（`results/t2diag/t2_static_floor.txt`）；
Tier 1 ep25 同样（114.10 对 49.71，`results/select_t1/t1_static_floor_c10s43.txt`）。这里给出一份“走路照参考、两条手臂不去够物体”的参考，训出的策略只当保底候选。
做法：两条手臂（shoulder_pitch/roll/yaw、elbow、wrist_roll/pitch/yaw）与 14 个 Dex3 手指关节在全部帧都换成 --arm-frame 那一帧的值（默认第 0 帧：参考开头人还没伸手），
腿、腰、根照抄；ee_pose_w（两只手掌）按 URDF 正向运动学重算（四元数 wxyz，与原 parquet 一致；先自检原关节角下算出的手掌与原 ee_pose_w 最大差 ≤ 2 mm，不过就退出）；
hand_contact_active 全部置 0（没有接触段，force_closure 不计）。其余列、object 目录、支撑面、v2d_meta.json 照抄（同 finger_close.py 的写法）。
报告：手臂关节的最大改动；新、原参考里“手臂点”（两只手各连杆碰撞网格的顶点，加肩、肘、腕连杆原点与上臂、前臂连线上的点）到停在第 0 帧位姿的物体表面采样点的最近距离
（不去够物体时仿真里它就停在那里），
与到支撑面方盒（用挂载的 flash_chord 导入器建 reconstructed_stage 里这一集的支撑面，只取 BOX；要挂训练用的那份副本，v0.4.0 原版不读方盒的 scale）的最近有向距离（全部帧的最小值与出现的帧）。
--arm-frame-scan（计划 agent-doc/plan-t12lift-2026-10-20.md 第 2 版，复核 MAJOR 3）：在 [0, 参考第一次接触) 里每 50 帧取一个候选帧（含第 0 帧），
选“全部帧里手臂点到物体与到支撑面的最近距离两者的较小值”最大的那一帧；并列取较早的帧。不给时用 --arm-frame。
说明：ik_error_per_frame 是重定向时的诊断量，本工具不改，新 parquet 里它已过期；ee_pose_w 的四元数取 w ≥ 0，可能与原来的符号相反（同一旋转）；
hand_object_contact_positions/normals/part_ids 不改：FlashCHORD 只在可视化与接触标记为 1 的帧的 force_closure 里用到它们，接触标记已全部置 0；
hand_frames_w 不为空时拒绝（G1 的参考里它是空的）。
用法（容器内）：python /repo/stages/retarget/hands_off.py <motion parquet> <out_root> --meta-src <motion_src 档位目录> --meta-dst <新档位目录>
  [--arm-frame 0 | --arm-frame-scan] [--dry-run]
最后一行 HANDS_OFF_DONE
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finger_close import URDF, Robot, quat_mat

ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")


def mat_to_wxyz(R: np.ndarray) -> np.ndarray:
    """(N,3,3) → (N,4) wxyz，w ≥ 0。"""
    out = np.empty((R.shape[0], 4))
    for i, m in enumerate(R):
        tr = np.trace(m)
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2
            out[i] = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            out[i] = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        elif m[1, 1] > m[2, 2]:
            s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            out[i] = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
        else:
            s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            out[i] = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    out[out[:, 0] < 0] *= -1
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("motion")
    ap.add_argument("out_root")
    ap.add_argument("--meta-src", required=True)
    ap.add_argument("--meta-dst", required=True)
    ap.add_argument("--arm-frame", type=int, default=0)
    ap.add_argument("--arm-frame-scan", action="store_true")
    ap.add_argument("--urdf", default=URDF)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    motion = Path(a.motion)
    seq_dir = motion.parents[1]
    seq = seq_dir.name.split("=", 1)[1]
    whole = seq_dir.parents[1]
    out_seq = Path(a.out_root) / "whole_body" / "soma" / seq_dir.name
    meta_src = Path(a.meta_src) / seq / "v2d_meta.json"
    meta_dst = Path(a.meta_dst) / seq / "v2d_meta.json"
    if not a.dry_run and (out_seq.exists() or meta_dst.exists()):
        print(f"{out_seq} 或 {meta_dst} 已存在，拒绝")
        return 2
    if not meta_src.is_file():
        print(f"没有 {meta_src}")
        return 2
    t = pq.read_table(motion)
    row = t.to_pylist()[0]
    names = list(row["robot_joint_names"])
    qj = np.asarray(row["robot_joint_positions"], dtype=float)
    T = qj.shape[0]
    rob = Robot(a.urdf)
    root_p = np.asarray(row["robot_root_position"], dtype=float)
    root_R = quat_mat(np.asarray(row["robot_root_wxyz"], dtype=float), "wxyz")
    ee = np.asarray(row["ee_pose_w"], dtype=float)
    ee_names = list(row["ee_link_names"])
    palms = set(ee_names)
    fk0 = rob.fk(root_p, root_R, {n: qj[:, i] for i, n in enumerate(names)}, palms)
    for k, ln in enumerate(ee_names):
        dp = float(np.linalg.norm(fk0[ln][0] - ee[:, k, :3], axis=1).max())
        R0 = quat_mat(ee[:, k, 3:], "wxyz")
        ang = float(np.degrees(np.arccos(np.clip((np.einsum("nij,nij->n", R0, fk0[ln][1]) - 1) / 2, -1, 1))).max())
        print(f"  自检 {ln}：原关节角下位置最大差 {dp * 1000:.2f} mm、朝向最大差 {ang:.3f}°")
        if dp > 0.002 or ang > 1.0:
            print("FK_SELFCHECK_FAIL，不写文件")
            return 3
    arm_idx = [i for i, n in enumerate(names) if any(n == f"{s}_{j}_joint" for s in ("left", "right") for j in ARM)]
    hand_idx = [i for i, n in enumerate(names) if "_hand_" in n]
    if len(arm_idx) != 14 or len(hand_idx) != 14:
        print(f"手臂关节应为 14 个、手指 14 个，找到 {len(arm_idx)}、{len(hand_idx)} 个")
        return 3
    if any(len(x) for x in (row.get("hand_frames_w") or [])):
        print("hand_frames_w 不为空，本工具没处理它，拒绝")
        return 3
    idx = arm_idx + hand_idx
    c_all = np.asarray(row["hand_contact_active"], dtype=bool)
    t_c = min((int(np.flatnonzero(x)[0]) for x in c_all if x.any()), default=T)
    # 手臂点：手的网格顶点，加肩、肘、腕原点与上臂、前臂连线上的点
    mesh_links = [f"{s}_hand_{p}_link" for s in ("left", "right") for p in ("palm", "thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")]
    lpts = {ln: rob.link_points(ln, 80) for ln in mesh_links}
    chain = [f"{s}_{p}_link" for s in ("left", "right") for p in ("shoulder_yaw", "elbow", "wrist_roll", "wrist_yaw")]
    want = set(mesh_links) | set(chain)
    mesh = trimesh.load(row["object_mesh_paths"][0], force="mesh")
    sp_obj, _ = trimesh.sample.sample_surface(mesh, 20000, seed=0)
    tree = cKDTree(sp_obj)
    op = np.asarray(row["object_body_position"], dtype=float)[:, 0]
    oR = quat_mat(np.asarray(row["object_body_wxyz"], dtype=float)[:, 0], "wxyz")
    boxes = []   # (center, R, half)
    usd = sorted((whole / "reconstructed_stage").glob(f"{seq}_*support.usda"))
    try:
        import newton
        from flash_chord.scene import support as fsup
        for u in usd:
            b = newton.ModelBuilder()
            try:
                fsup.add_support_surfaces(b, str(u))
            except ValueError as e:   # 支撑面文件里没有方盒或圆柱（如 Tier 2 ep4 的空 Xform；计划 agent-doc/plan-t12arm-2026-10-20.md 复核 BLOCKER 1）
                print(f"  支撑面 {u.name} 读不出形状（{e}），没算进距离")
                continue
            for i in range(b.shape_count):
                kind = newton.GeoType(b.shape_type[i]).name
                if kind not in ("BOX", "CYLINDER"):
                    print(f"  支撑面 {u.name} 第 {i} 个形状是 {kind}，没算进距离")
                    continue
                tr = np.asarray(b.shape_transform[i], dtype=float)          # p(3) + q(4, xyzw)
                boxes.append((kind, tr[:3], quat_mat(tr[3:], "xyzw"), np.asarray(b.shape_scale[i], dtype=float)[:3]))
    except ImportError as e:
        print(f"  读不了支撑面（{e}），只报物体距离")
    print(f"  支撑面文件 {[u.name for u in usd]}；形状 {len(boxes)} 个：" + "；".join(f"{k} 中心 {np.round(c, 3).tolist()} 尺寸 {np.round(h, 3).tolist()}" for k, c, _, h in boxes))
    frames = np.arange(0, T, 2)

    def clearance(q_new: np.ndarray) -> tuple:
        L = rob.fk(root_p[frames], root_R[frames], {n: q_new[frames, i] for i, n in enumerate(names)}, want)
        pts = []
        for ln in mesh_links:
            p, R = L[ln]
            pts.append(p[:, None, :] + np.einsum("nij,vj->nvi", R, lpts[ln]))
        for s in ("left", "right"):
            o = [L[f"{s}_{p}_link"][0] for p in ("shoulder_yaw", "elbow", "wrist_roll", "wrist_yaw")]
            for a_, b_ in ((o[0], o[1]), (o[1], o[2]), (o[2], o[3])):
                pts.append(np.stack([a_ + (b_ - a_) * w for w in np.linspace(0, 1, 6)], 1))
        P = np.concatenate(pts, 1)                                             # (F,V,3)
        F = frames
        po = (P - op[0]) @ oR[0]          # 物体停在第 0 帧的位姿（不去够它时仿真里它就停在那里）
        d_obj = tree.query(po.reshape(-1, 3))[0].reshape(po.shape[:2]).min(1)
        d_sup = np.full(len(F), np.inf)
        for kind, c, R, h in boxes:
            loc = (P - c) @ R                                                  # (F,V,3)，形状的局部系
            if kind == "BOX":
                ex = np.abs(loc) - h
            else:                                                              # CYLINDER：轴向为局部 z，h = (半径, 半高, 0)
                ex = np.stack([np.linalg.norm(loc[..., :2], axis=-1) - h[0], np.abs(loc[..., 2]) - h[1]], -1)
            out = np.linalg.norm(np.maximum(ex, 0.0), axis=-1)
            ins = ex.max(-1)
            d_sup = np.minimum(d_sup, np.where(ins > 0, out, ins).min(1))
        return d_obj, d_sup

    cands = [a.arm_frame] if not a.arm_frame_scan else list(range(0, max(t_c, 1), 50))
    if any(not 0 <= f < T for f in cands):
        print(f"--arm-frame 越界：{cands}（共 {T} 帧）")
        return 2
    best = None
    for f in cands:
        q_try = qj.copy()
        q_try[:, idx] = qj[f, idx][None, :]
        d_obj, d_sup = clearance(q_try)
        score = min(d_obj.min(), d_sup.min())
        print(f"  候选第 {f} 帧：手臂点到物体最近 {d_obj.min():.3f} m（第 {int(frames[d_obj.argmin()])} 帧），到支撑面方盒最近 {d_sup.min():.3f} m"
              + (f"（第 {int(frames[d_sup.argmin()])} 帧）" if np.isfinite(d_sup.min()) else "") + f"；两者较小值 {score:.3f} m")
        if best is None or score > best[1]:
            best = (f, score)
    arm_frame = best[0]
    d0_obj, d0_sup = clearance(qj)
    print(f"  原参考：手臂点到物体最近 {d0_obj.min():.3f} m（第 {int(frames[d0_obj.argmin()])} 帧），到支撑面方盒最近 {d0_sup.min():.3f} m；参考第一次接触 t_c {t_c}")
    new_q = qj.copy()
    new_q[:, idx] = qj[arm_frame, idx][None, :]
    print(f"参考 {motion}：{T} 帧；手臂与手指 28 个关节固定为第 {arm_frame} 帧的值" + ("（扫描选出）" if a.arm_frame_scan else "")
          + f"；关节最大改动 {np.abs(new_q - qj).max():.3f} rad；其余关节改动 {np.abs(np.delete(new_q - qj, idx, axis=1)).max():.3g} rad")
    fk1 = rob.fk(root_p, root_R, {n: new_q[:, i] for i, n in enumerate(names)}, palms)
    new_ee = ee.copy()
    for k, ln in enumerate(ee_names):
        new_ee[:, k, :3] = fk1[ln][0]
        new_ee[:, k, 3:] = mat_to_wxyz(fk1[ln][1])
    if a.dry_run:
        print("dry-run：不写文件")
        print("HANDS_OFF_DONE")
        return 0
    out_seq.mkdir(parents=True)
    shutil.copytree(seq_dir / "object", out_seq / "object")
    (out_seq / "robot_name=g1").mkdir()
    t2 = t
    for col, val in (("robot_joint_positions", [new_q.tolist()]), ("ee_pose_w", [new_ee.tolist()]),
                     ("hand_contact_active", [np.zeros_like(np.asarray(row["hand_contact_active"])).tolist()])):
        f = t2.schema.field(col)
        t2 = t2.set_column(t2.schema.get_field_index(col), f, pa.array(val, type=f.type))
    swapped = 0
    for col in ("object_mesh_paths", "object_urdf_paths"):
        old = t2.column(col).to_pylist()[0]
        vals = [[p.replace(str(seq_dir), str(out_seq)) for p in old]]
        swapped += sum(n != o for n, o in zip(vals[0], old))
        f = t2.schema.field(col)
        t2 = t2.set_column(t2.schema.get_field_index(col), f, pa.array(vals, type=f.type))
    if swapped == 0:
        shutil.rmtree(out_seq)
        print(f"MESH_PATH_FAIL：网格路径没有一条以 {seq_dir} 开头，新目录已撤回")
        return 3
    pq.write_table(t2, out_seq / "robot_name=g1" / "data.parquet")
    st_dst = Path(a.out_root) / "whole_body" / "reconstructed_stage"
    st_dst.mkdir(parents=True, exist_ok=True)
    for f in (whole / "reconstructed_stage").glob(f"{seq}_*"):
        shutil.copy2(f, st_dst / f.name)
    meta_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(meta_src, meta_dst)
    print(f"写出 {out_seq}、{st_dst} 里这一集的支撑面与 {meta_dst}；网格路径换了 {swapped} 条")
    print("HANDS_OFF_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
