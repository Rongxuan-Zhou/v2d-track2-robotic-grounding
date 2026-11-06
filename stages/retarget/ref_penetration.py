"""参考里机器人的手插进物体多深（训练机上经 scripts/dev/in-flash-chord 在容器里跑，只用 CPU；只读，只报不判）。
起因：agent-doc/plan-t2dg-2026-10-20.md 执行记录：ep0 的对照里成功世界的手掌也多半“贴不住”（d_rob ≤ d_ref + 3 cm 的帧左手 13%、右手 61%），
stages/retarget/finger_close.py 在 ep25 上量到张开的手指已在瓶子网格里面。猜测：重定向出来的参考手（手掌、手指）插在物体里，仿真里的手插不进去。
这里逐集量：参考接触帧里，各手手掌与各手指连杆碰撞网格上的点，到物体表面的有向距离（外正内负）。
符号：物体网格水密时用 trimesh 的射线判断点在不在里面（contains），距离取到表面采样点的最近距离；不水密时退回最近采样点的面法向（(p − s)·n），
报告里写明用的是哪一种，并自检：物体网格自己的质心到表面的有向距离应为负、物体包围盒外 1 m 的点应为正。
每集一行：序列、帧、网格是否水密、符号方法、各手接触帧数、手掌点最小有向距离的接触帧分位（10/50/90，mm）、手指点的同样分位、
“手掌点有一点在物体里面”的接触帧比例。--every 取每隔几个接触帧算一次（默认 5）。
用法（容器内）：python /repo/stages/retarget/ref_penetration.py [--every 5] <motion parquet>...   最后一行 REF_PENETRATION_DONE
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

PARTS = {"palm": ("palm",), "fingers": ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")}


class Surface:
    def __init__(self, mesh_path: str, n: int = 20000):
        self.mesh = trimesh.load(mesh_path, force="mesh")
        self.watertight = bool(self.mesh.is_watertight)
        pts, fid = trimesh.sample.sample_surface(self.mesh, n, seed=0)
        self.pts, self.nrm = pts, self.mesh.face_normals[fid]
        self.tree = cKDTree(pts)

    def sdist(self, p_obj: np.ndarray) -> np.ndarray:
        d, i = self.tree.query(p_obj)
        if self.watertight:
            inside = self.mesh.contains(p_obj)
            return np.where(inside, -d, d)
        return np.einsum("ij,ij->i", p_obj - self.pts[i], self.nrm[i])


def pct(x) -> str:
    return " / ".join(f"{np.percentile(x, p) * 1000:.1f}" for p in (10, 50, 90)) if len(x) else "（无）"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("motion", nargs="+")
    ap.add_argument("--every", type=int, default=5)
    ap.add_argument("--urdf", default=URDF)
    a = ap.parse_args()
    rob = Robot(a.urdf)
    pts_cache = {}
    print("序列\t帧\t水密\t符号\t手\t接触帧\t手掌最小有向距离 10/50/90 mm\t手指最小有向距离 10/50/90 mm\t手掌有点在里面的帧比例")
    for mp in a.motion:
        row = pq.read_table(mp).to_pylist()[0]
        seq = row["sequence_id"]
        names = list(row["robot_joint_names"])
        qj = np.asarray(row["robot_joint_positions"], dtype=float)
        T = qj.shape[0]
        root_p = np.asarray(row["robot_root_position"], dtype=float)
        root_R = quat_mat(np.asarray(row["robot_root_wxyz"], dtype=float), "wxyz")
        op = np.asarray(row["object_body_position"], dtype=float)[:, 0]
        oR = quat_mat(np.asarray(row["object_body_wxyz"], dtype=float)[:, 0], "wxyz")
        surf = Surface(row["object_mesh_paths"][0])
        c0 = surf.sdist(np.asarray(surf.mesh.centroid)[None])[0]
        far = surf.sdist((surf.mesh.bounds[1] + 1.0)[None])[0]
        sign = "射线" if surf.watertight else "法向"
        if not (c0 < 0 < far):
            print(f"{seq}\t{T}\t{surf.watertight}\t{sign}\t—\t符号自检不过（质心 {c0:+.3f} m、外点 {far:+.3f} m），这一集不报")
            continue
        for s, cact in zip(row["hand_sides"], row["hand_contact_active"]):
            c = np.asarray(cact, dtype=bool)
            frames = np.flatnonzero(c)[:: a.every]
            if frames.size == 0:
                print(f"{seq}\t{T}\t{surf.watertight}\t{sign}\t{s}\t0\t—\t—\t—")
                continue
            links = {k: [f"{s}_hand_{p}_link" for p in v] for k, v in PARTS.items()}
            want = {ln for v in links.values() for ln in v}
            for ln in want:
                if ln not in pts_cache:
                    pts_cache[ln] = rob.link_points(ln, 200)
            sub = {n: qj[frames, i] for i, n in enumerate(names)}
            L = rob.fk(root_p[frames], root_R[frames], sub, want)
            res = {k: [] for k in PARTS}
            inside_palm = []
            for j, t in enumerate(frames):
                for k, lns in links.items():
                    pw = np.concatenate([L[ln][0][j] + pts_cache[ln] @ L[ln][1][j].T for ln in lns])
                    d = surf.sdist((pw - op[t]) @ oR[t])
                    res[k].append(d.min())
                    if k == "palm":
                        inside_palm.append(bool((d < 0).any()))
            print(f"{seq}\t{T}\t{surf.watertight}\t{sign}\t{s}\t{int(c.sum())}\t{pct(res['palm'])}\t{pct(res['fingers'])}\t{np.mean(inside_palm):.2f}")
    print("REF_PENETRATION_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
