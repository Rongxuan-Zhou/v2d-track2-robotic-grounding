"""参考里接触之前少弯腰（训练机上经 scripts/dev/in-flash-chord 在容器里跑，只用 CPU）。
计划 agent-doc/plan-t12ref-2026-10-20.md：Tier 2 ep4 的零残差回放在 1024 个世界里都在第 88 帧摔（`results/ml40/t12replay/mech_*ep4*`），那一段参考一边以 0.9 m/s 走、
一边把 waist_pitch 从 0.1 加到 0.52 rad（顶在限位上）。这里把 [0, t_end) 里的 waist_pitch 限在 --cap 以下，[t_end − --blend, t_end) 线性过渡回原值（t_end 默认 =
参考第一次接触的帧，可用 --until 改），之后不改；其余关节、根照抄；ee_pose_w 按 URDF 正向运动学重算（原关节角下先自检 ≤ 2 mm、≤ 1°）。
其余列、object 目录、支撑面、v2d_meta.json 照抄（同 stages/retarget/hands_off.py 的写法；ik_error_per_frame 因此不再对应新关节角）；hand_frames_w 不为空就拒绝；目标已存在就拒绝。
报告：改了多少帧、waist_pitch 最大改动、手掌位置的最大改动。
用法（容器内）：python /repo/stages/retarget/waist_cap.py <motion parquet> <out_root> --meta-src <motion_src 档位目录> --meta-dst <新档位目录> --cap 0.25 [--blend 20] [--until 帧] [--dry-run]
最后一行 WAIST_CAP_DONE
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finger_close import URDF, Robot, quat_mat
from hands_off import mat_to_wxyz


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("motion")
    ap.add_argument("out_root")
    ap.add_argument("--meta-src", required=True)
    ap.add_argument("--meta-dst", required=True)
    ap.add_argument("--cap", type=float, required=True)
    ap.add_argument("--blend", type=int, default=20)
    ap.add_argument("--until", type=int)
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
    fk0 = rob.fk(root_p, root_R, {n: qj[:, i] for i, n in enumerate(names)}, set(ee_names))
    for k, ln in enumerate(ee_names):
        dp = float(np.linalg.norm(fk0[ln][0] - ee[:, k, :3], axis=1).max())
        R0 = quat_mat(ee[:, k, 3:], "wxyz")
        ang = float(np.degrees(np.arccos(np.clip((np.einsum("nij,nij->n", R0, fk0[ln][1]) - 1) / 2, -1, 1))).max())
        print(f"  自检 {ln}：原关节角下位置最大差 {dp * 1000:.2f} mm、朝向最大差 {ang:.3f}°")
        if dp > 0.002 or ang > 1.0:
            print("FK_SELFCHECK_FAIL，不写文件")
            return 3
    if any(len(x) for x in (row.get("hand_frames_w") or [])):
        print("hand_frames_w 不为空，本工具没处理它，拒绝")
        return 3
    c = np.asarray(row["hand_contact_active"], dtype=bool)
    t_c = min((int(np.flatnonzero(x)[0]) for x in c if x.any()), default=T)
    t_end = min(T, a.until if a.until is not None else t_c)
    j = names.index("waist_pitch_joint")
    orig = qj[:, j].copy()
    w = np.ones(T)                                   # 1 = 完全限住，0 = 原值
    w[t_end:] = 0.0
    lo = max(0, t_end - a.blend)
    if t_end > lo:
        w[lo:t_end] = np.linspace(1.0, 0.0, t_end - lo, endpoint=False)
    capped = np.minimum(orig, a.cap)
    new_q = qj.copy()
    new_q[:, j] = w * capped + (1 - w) * orig
    changed = np.abs(new_q[:, j] - orig) > 1e-9
    fk1 = rob.fk(root_p, root_R, {n: new_q[:, i] for i, n in enumerate(names)}, set(ee_names))
    new_ee = ee.copy()
    for k, ln in enumerate(ee_names):
        new_ee[:, k, :3] = fk1[ln][0]
        new_ee[:, k, 3:] = mat_to_wxyz(fk1[ln][1])
    dpalm = np.linalg.norm(new_ee[:, :, :3] - ee[:, :, :3], axis=-1).max()
    print(f"参考 {motion}：{T} 帧；t_c {t_c}，限腰到第 {t_end} 帧（过渡 {lo}–{t_end}）；waist_pitch 上限 {a.cap} rad；改了 {int(changed.sum())} 帧，"
          f"最大改动 {np.abs(new_q[:, j] - orig).max():.3f} rad（原最大 {orig[:t_end].max() if t_end > 0 else float('nan'):.3f}）；手掌位置最大改动 {dpalm:.3f} m；"
          f"其余关节改动 {np.abs(np.delete(new_q - qj, j, axis=1)).max():.3g} rad")
    if a.dry_run:
        print("dry-run：不写文件")
        print("WAIST_CAP_DONE")
        return 0
    out_seq.mkdir(parents=True)
    shutil.copytree(seq_dir / "object", out_seq / "object")
    (out_seq / "robot_name=g1").mkdir()
    t2 = t
    for col, val in (("robot_joint_positions", [new_q.tolist()]), ("ee_pose_w", [new_ee.tolist()])):
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
    print(f"写出 {out_seq}、支撑面与 {meta_dst}；网格路径换了 {swapped} 条")
    print("WAIST_CAP_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
