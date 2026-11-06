"""参考里让机器人原地站着、一步不走（训练机上经 scripts/dev/in-flash-chord 在容器里跑，只用 CPU）。
计划 agent-doc/plan-t3static-2026-10-21.md 第 1 节第 1 步（第 2 版）：跟丢的 Tier 3 集换成“机器人原地站、物体不动”的保底；零残差时动作只取决于参考与机器人状态，
参考里机器人不动，物体就停在第 0 帧（对齐后就是 Tier 1 第 0 帧的位置）。写法照 stages/retarget/hands_off.py。
随帧变的机器人列（stages/retarget/parquet_columns.py 列出，`results/t3static/parquet_columns_hm_ep16.txt`）：robot_root_position、robot_root_wxyz、robot_joint_positions、ee_pose_w；
没有速度列。前三列全部帧换成第 0 帧的值；ee_pose_w 按 URDF 正向运动学重算（先自检原关节角下算出的手掌与原 ee_pose_w 最大差 ≤ 2 mm、≤ 1°，不过就退出），
hand_contact_active 全部置 0。ik_error_per_frame、ik_num_iterations、frame_task_errors 是重定向时的诊断量，不改（已过期）。
物体列、object 目录、支撑面、v2d_meta.json 照抄；写完自检：机器人三列全帧与第 0 帧逐位相同，物体位姿列（object_body_position/wxyz、object_root_position/axis_angle）与原 parquet 相同。
报告：第 0 帧机器人各连杆（有碰撞网格的取网格顶点，没有的取连杆原点）到停在第 0 帧位姿的物体表面采样点的最近距离，与最近的连杆。
用法（容器内）：python /repo/stages/retarget/stand_still.py <motion parquet> <out_root> --meta-src <motion_src 档位目录> --meta-dst <新档位目录> [--dry-run]
最后一行 STAND_STILL_DONE
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
from finger_close import URDF, Robot, quat_mat  # noqa: E402
from hands_off import mat_to_wxyz  # noqa: E402

HOLD = ("robot_root_position", "robot_root_wxyz", "robot_joint_positions")
OBJ = ("object_body_position", "object_body_wxyz", "object_root_position", "object_root_axis_angle")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("motion")
    ap.add_argument("out_root")
    ap.add_argument("--meta-src", required=True)
    ap.add_argument("--meta-dst", required=True)
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
    if any(len(x) for x in (row.get("hand_frames_w") or [])):
        print("hand_frames_w 不为空，本工具没处理它，拒绝")
        return 3
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

    hold = {c: np.repeat(np.asarray(row[c], dtype=float)[:1], T, axis=0) for c in HOLD}
    new_R = quat_mat(hold["robot_root_wxyz"], "wxyz")
    fk1 = rob.fk(hold["robot_root_position"], new_R, {n: hold["robot_joint_positions"][:, i] for i, n in enumerate(names)}, palms)
    new_ee = ee.copy()
    for k, ln in enumerate(ee_names):
        new_ee[:, k, :3] = fk1[ln][0]
        new_ee[:, k, 3:] = mat_to_wxyz(fk1[ln][1])
    moved = {c: float(np.abs(hold[c] - np.asarray(row[c], dtype=float)).max()) for c in HOLD}
    print(f"参考 {motion}：{T} 帧；机器人根位置、根朝向、43 个关节全部帧固定为第 0 帧；原参考里的最大改动 "
          + "、".join(f"{c} {v:.3f}" for c, v in moved.items()))

    # 第 0 帧机器人各连杆到物体（停在第 0 帧位姿）的最近距离
    links = list(rob.links)
    L0 = rob.fk(root_p[:1], root_R[:1], {n: qj[:1, i] for i, n in enumerate(names)}, set(links))
    mesh = trimesh.load(row["object_mesh_paths"][0], force="mesh")
    sp_obj, _ = trimesh.sample.sample_surface(mesh, 20000, seed=0)
    tree = cKDTree(sp_obj)
    op = np.asarray(row["object_body_position"], dtype=float)[0, 0]
    oR = quat_mat(np.asarray(row["object_body_wxyz"], dtype=float)[:1, 0], "wxyz")[0]
    best = (np.inf, None)
    for ln, (p, R) in L0.items():
        try:
            v = rob.link_points(ln, 200)
        except (AttributeError, TypeError, ValueError, OSError):
            v = np.zeros((1, 3))
        P = p[0] + v @ R[0].T
        d = float(tree.query((P - op) @ oR)[0].min())
        if d < best[0]:
            best = (d, ln)
    print(f"  第 0 帧机器人到物体最近 {best[0]:.3f} m（{best[1]}）；骨盆到物体中心 {float(np.linalg.norm(root_p[0] - op)):.3f} m")
    if a.dry_run:
        print("dry-run：不写文件")
        print("STAND_STILL_DONE")
        return 0

    out_seq.mkdir(parents=True)
    shutil.copytree(seq_dir / "object", out_seq / "object")
    (out_seq / "robot_name=g1").mkdir()
    t2 = t
    cols = [(c, [hold[c].tolist()]) for c in HOLD] + [("ee_pose_w", [new_ee.tolist()]),
            ("hand_contact_active", [np.zeros_like(np.asarray(row["hand_contact_active"])).tolist()])]
    for col, val in cols:
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
    out_pq = out_seq / "robot_name=g1" / "data.parquet"
    pq.write_table(t2, out_pq)
    chk = pq.read_table(out_pq).to_pylist()[0]
    bad = [c for c in HOLD if not np.array_equal(np.asarray(chk[c], dtype=float), np.repeat(np.asarray(chk[c], dtype=float)[:1], T, axis=0))]
    bad += [c for c in OBJ if chk[c] != row[c]]
    if bad:
        shutil.rmtree(out_seq)
        print(f"STAND_CHECK_FAIL：{bad}，新目录已撤回")
        return 3
    st_dst = Path(a.out_root) / "whole_body" / "reconstructed_stage"
    st_dst.mkdir(parents=True, exist_ok=True)
    for f in (whole / "reconstructed_stage").glob(f"{seq}_*"):
        shutil.copy2(f, st_dst / f.name)
    meta_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(meta_src, meta_dst)
    print(f"写出 {out_seq}、{st_dst} 里这一集的支撑面与 {meta_dst}；网格路径换了 {swapped} 条；写后自检过（机器人三列全帧相同、物体四列与原参考相同）")
    print("STAND_STILL_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
