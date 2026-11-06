"""参考里机器人在接触之前离物体多近（训练机上经 scripts/dev/in-flash-chord 在容器里跑，只用 CPU；只读，只报不判）。
起因：agent-doc/plan-t2dg-2026-10-20.md 执行记录：Tier 2 ep4、ep28 的物体在参考第一次接触之前就离开了（第 128、256 帧，参考接触第 133、399 帧）。
要分清两种情形：参考里的机器人自己就穿过物体（策略跟得越准越会碰），还是参考是干净的、策略跟偏了才碰。
逐帧对参考机器人每个带碰撞网格的连杆（URDF 的 collision mesh，每个连杆取至多 --pts 个顶点）求到物体表面采样点的最近距离，
符号按最近采样点的面法向（外正内负；网格不水密时只能这样，报告里标明）。物体位姿取参考的 object_body_position / object_body_wxyz。
报：参考两只手第一次接触的帧 t_c；[--from, t_c + --after) 里每 --every 帧一行：全身最近的连杆与它的有向距离、两只手掌各自的距离；
另报这段里“全身最小有向距离 < 0”的帧数与第一帧，以及离物体最近的 5 个连杆（按这段里的最小距离）。
用法（容器内）：python /repo/stages/retarget/ref_clearance.py <motion parquet> [--from 0] [--after 10] [--every 5] [--pts 150] [--focus 帧…]   最后一行 REF_CLEARANCE_DONE
  --focus：另把这几帧前后各 5 帧逐帧列出（如物体离开的帧）。
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import trimesh
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finger_close import URDF, Robot, quat_mat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("motion")
    ap.add_argument("--urdf", default=URDF)
    ap.add_argument("--from", dest="start", type=int, default=0)
    ap.add_argument("--after", type=int, default=10)
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--pts", type=int, default=150)
    ap.add_argument("--focus", type=int, nargs="*", default=[])
    a = ap.parse_args()
    row = pq.read_table(a.motion).to_pylist()[0]
    names = list(row["robot_joint_names"])
    qj = np.asarray(row["robot_joint_positions"], dtype=float)
    T = qj.shape[0]
    rob = Robot(a.urdf)
    links = [ln for ln, el in rob.links.items() if el.find("collision/geometry/mesh") is not None]
    pts = {}
    for ln in links:
        try:
            pts[ln] = rob.link_points(ln, a.pts)
        except (OSError, ValueError) as e:   # 个别网格读不了就跳过并报出来
            print(f"  跳过 {ln}：{e}")
    links = list(pts)
    root_p = np.asarray(row["robot_root_position"], dtype=float)
    root_R = quat_mat(np.asarray(row["robot_root_wxyz"], dtype=float), "wxyz")
    op = np.asarray(row["object_body_position"], dtype=float)[:, 0]
    oR = quat_mat(np.asarray(row["object_body_wxyz"], dtype=float)[:, 0], "wxyz")
    mesh = trimesh.load(row["object_mesh_paths"][0], force="mesh")
    sp, fid = trimesh.sample.sample_surface(mesh, 20000, seed=0)
    nrm = mesh.face_normals[fid]
    tree = cKDTree(sp)
    contact = np.asarray(row["hand_contact_active"], dtype=bool)
    firsts = [int(np.flatnonzero(c)[0]) for c in contact if c.any()]
    t_c = min(firsts) if firsts else T
    hi = min(T, t_c + a.after)
    frames = sorted(set(range(a.start, hi, a.every)) | {f for c in a.focus for f in range(max(0, c - 5), min(T, c + 6))})
    print(f"参考 {a.motion}：{T} 帧；物体网格水密 {mesh.is_watertight}（符号按面法向）；带碰撞网格的连杆 {len(links)} 个；参考第一次接触 t_c {t_c}")
    qd = {n: qj[frames, i] for i, n in enumerate(names)}
    L = rob.fk(root_p[frames], root_R[frames], qd, set(links))
    per_link_min = {ln: np.inf for ln in links}
    rows, inside = [], []
    for j, t in enumerate(frames):
        best, best_ln, palm = np.inf, "", {}
        for ln in links:
            p, R = L[ln]
            pw = p[j] + pts[ln] @ R[j].T
            po = (pw - op[t]) @ oR[t]
            _, i = tree.query(po)
            d = float(np.einsum("ij,ij->i", po - sp[i], nrm[i]).min())
            per_link_min[ln] = min(per_link_min[ln], d)
            if d < best:
                best, best_ln = d, ln
            if ln.endswith("_hand_palm_link"):
                palm[ln.split("_")[0]] = d
        if best < 0:
            inside.append(t)
        rows.append((t, best_ln, best, palm.get("left", np.nan), palm.get("right", np.nan)))
    print("  帧\t全身最近的连杆\t有向距离 mm\t左手掌 mm\t右手掌 mm")
    for t, ln, d, l_, r_ in rows:
        mark = " ←" if t in a.focus else ""
        print(f"  {t}\t{ln}\t{d * 1000:.1f}\t{l_ * 1000:.1f}\t{r_ * 1000:.1f}{mark}")
    print(f"  [{a.start}, {hi}) 里算了 {len(frames)} 帧；全身最小有向距离 < 0 的 {len(inside)} 帧，第一帧 {inside[0] if inside else '—'}")
    top = sorted(per_link_min.items(), key=lambda kv: kv[1])[:5]
    print("  离物体最近的 5 个连杆（这段里的最小有向距离）：" + "，".join(f"{ln} {d * 1000:.1f} mm" for ln, d in top))
    print("REF_CLEARANCE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
