"""参考里给 Dex3 合手：接触段里每根手指沿合手方向转到碰到物体表面为止（训练机上经 scripts/dev/in-flash-chord 在容器里跑，只用 CPU）。
起因：agent-doc/plan-t2dg-2026-10-20.md 事实 2 与执行记录：30 集 Tier 2 参考（重定向出来的 motion parquet）的 14 个 Dex3 手指关节全程是 0，
而 index、middle 的 0 位就是伸直顶在限位上；手指的基准目标就取参考的关节角（runtime/actions/sonic.py）。这里只改参考，不改仿真、不改评测。
做法（每集、每只手独立）：
  1. 整机正向运动学（URDF：G1 + Dex3，FlashCHORD 资产里那份）：根位姿 robot_root_position / robot_root_wxyz、关节角 robot_joint_positions。
     自检：算出的手掌（<side>_hand_palm_link）位置与 parquet 的 ee_pose_w 位置逐帧比，四元数按 xyzw、wxyz 两种读法各比一次，取对得上的那种；
     位置最大差 > --fk-tol-m（默认 2 mm）或两种读法都对不上（朝向差 > 1°）就报错退出，不写任何文件。
  2. 物体：object_mesh_paths 的网格，按 object_body_position / object_body_wxyz 放进世界；表面采 --surf 个点（带面法向），KD 树求最近点，
     手指点到表面的有向距离 = (p − s)·n（外正内负）。
  3. 手指：thumb（thumb_1、thumb_2 一起转，thumb_0 保持参考值）、index（index_0、index_1）、middle（middle_0、middle_1）三根，各自一个合手量 α ∈ [0, 1]：
     关节角 = 参考值 + α ×（合手端 × --close-frac − 参考值）；合手端取该关节限位里绝对值较大的那一端（index/middle 左负右正，thumb_1/2 左正右负，
     与 Dex3 左右镜像一致）。该手在参考里接触的帧上，α 从 0 起按 --step 增大，手指各节碰撞网格的点里有向距离第一次 ≤ --margin-m 时停在上一步；
     α = 1 也碰不到就取 --free-alpha（不贴物体的手指只半合，免得握空拳）。α = 0 时已经 ≤ margin（手指本来就贴着或穿进物体）就记 0。
  3b. --mode fixed（计划 agent-doc/plan-t2fc-2026-10-20.md）：不做几何搜索，接触帧的 α 一律取 --alpha。PD 的目标落在物体表面以内，手指才会往里压、有握力；
     “碰到表面就停”的目标只够贴上，不出力。第 3 步的 touch 模式照旧（--mode touch，默认）。
  4. 时间上：只在接触帧有 α；每段接触前后各 --ramp 帧线性渐入、渐出（取渐入渐出端点的 α），再做 --smooth 帧滑动平均（居中，不移时刻；
     渐入渐出段外侧再多 2 帧会带上很小的 α，约 0.02–0.09）。
  5. 写出：新 motion 根 <out_root>/whole_body/soma/sequence_id=<seq>/robot_name=g1/data.parquet 只改 robot_joint_positions 的 14 个手指列，
     其余列、object 目录（网格、URDF）与 reconstructed_stage 里这一集的支撑面照抄；parquet 里的网格与 URDF 路径换到新目录，一条都没换到就报错。
     给了 --meta-src/--meta-dst（runs/motion_src 下的档位目录）时，把 <meta-src>/<序列>/v2d_meta.json 照抄到 <meta-dst>/<序列>/（导出与起链从这里取档位与 episode）。
     目标已存在就拒绝。
报告：自检结果、各手各指 α 的分布、接触帧里“碰到表面”的比例、合手后手指点的最小有向距离（穿进物体多深）、拇指尖与食指尖距离的变化。只报不判。
用法（容器内）：python /repo/stages/retarget/finger_close.py <motion parquet> <out_root> [--urdf …] [--close-frac 0.9] [--margin-m 0.003] [--step 0.02]
  [--free-alpha 0.5] [--ramp 8] [--smooth 5] [--surf 20000] [--dry-run]   最后一行 FINGER_CLOSE_DONE
"""
import argparse
import shutil
import sys
import xml.etree.ElementTree as ET  # 只解析 FlashCHORD 自带的 URDF 资产
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import trimesh
from scipy.spatial import cKDTree

URDF = "/workspace/src/flash_chord/assets/urdfs/g1/main_with_hand.urdf"
FINGERS = {"thumb": ("thumb_1", "thumb_2"), "index": ("index_0", "index_1"), "middle": ("middle_0", "middle_1")}
LINKS = {"thumb": ("thumb_1", "thumb_2"), "index": ("index_0", "index_1"), "middle": ("middle_0", "middle_1")}


def rpy_mat(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def axis_mat(axis, q):
    """绕单位轴 axis 转 q（Rodrigues）；q 可以是 (N,)，返回 (N,3,3)。"""
    q = np.asarray(q, dtype=float).reshape(-1)
    k = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    s, c = np.sin(q)[:, None, None], np.cos(q)[:, None, None]
    return np.eye(3)[None] + s * K[None] + (1 - c) * (K @ K)[None]


def quat_mat(q, order):
    q = np.asarray(q, dtype=float)
    if order == "wxyz":
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    else:
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


class Robot:
    def __init__(self, urdf: str):
        self.dir = Path(urdf).parent
        root = ET.parse(urdf).getroot()
        self.joints = {}
        children = set()
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            ax = j.find("axis")
            lim = j.find("limit")
            self.joints[j.get("name")] = {
                "type": j.get("type"), "parent": j.find("parent").get("link"), "child": j.find("child").get("link"),
                "xyz": xyz, "R": rpy_mat(*rpy), "axis": np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else None,
                "lower": float(lim.get("lower")) if lim is not None and lim.get("lower") else None,
                "upper": float(lim.get("upper")) if lim is not None and lim.get("upper") else None}
            children.add(j.get("child"))
        self.links = {ln.get("name"): ln for ln in root.findall("link")}
        roots = [n for n in self.links if n not in children]
        self.root = "pelvis" if "pelvis" in roots else roots[0]
        self.by_parent = {}
        for n, j in self.joints.items():
            self.by_parent.setdefault(j["parent"], []).append(n)

    def fk(self, root_pos, root_R, q: dict, want: set) -> dict:
        """root_pos (N,3)、root_R (N,3,3)、q {关节名: (N,)} → {连杆名: (pos (N,3), R (N,3,3))}，只返回 want 里的连杆。"""
        N = root_pos.shape[0]
        out, stack = {}, [(self.root, root_pos, root_R)]
        while stack:
            link, p, R = stack.pop()
            if link in want:
                out[link] = (p, R)
            for jn in self.by_parent.get(link, []):
                j = self.joints[jn]
                pc = p + np.einsum("nij,j->ni", R, j["xyz"])
                Rc = R @ j["R"]
                if j["type"] in ("revolute", "continuous"):
                    Rc = Rc @ axis_mat(j["axis"], q.get(jn, np.zeros(N)))
                stack.append((j["child"], pc, Rc))
        return out

    def link_points(self, link: str, n: int = 300) -> np.ndarray:
        el = self.links[link].find("collision/geometry/mesh")
        m = trimesh.load(self.dir / el.get("filename"), force="mesh")
        v = np.asarray(m.vertices, dtype=float)
        if len(v) > n:
            v = v[np.random.default_rng(0).choice(len(v), n, replace=False)]
        return v


def ramp_alpha(alpha: np.ndarray, contact: np.ndarray, ramp: int, smooth: int) -> np.ndarray:
    T = len(alpha)
    a = np.where(contact, alpha, 0.0).astype(float)
    segs, start = [], None
    for t in range(T + 1):
        c = contact[t] if t < T else False
        if c and start is None:
            start = t
        if not c and start is not None:
            segs.append((start, t - 1))
            start = None
    out = a.copy()
    for s, e in segs:
        for k in range(1, ramp + 1):
            if s - k >= 0 and not contact[s - k]:
                out[s - k] = max(out[s - k], a[s] * (1 - k / (ramp + 1)))
            if e + k < T and not contact[e + k]:
                out[e + k] = max(out[e + k], a[e] * (1 - k / (ramp + 1)))
    if smooth > 1:
        ker = np.ones(smooth) / smooth
        out = np.convolve(np.pad(out, (smooth // 2, smooth - 1 - smooth // 2), mode="edge"), ker, mode="valid")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("motion")
    ap.add_argument("out_root")
    ap.add_argument("--urdf", default=URDF)
    ap.add_argument("--close-frac", type=float, default=0.9)
    ap.add_argument("--margin-m", type=float, default=0.003)
    ap.add_argument("--step", type=float, default=0.02)
    ap.add_argument("--free-alpha", type=float, default=0.5)
    ap.add_argument("--ramp", type=int, default=8)
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--surf", type=int, default=20000)
    ap.add_argument("--fk-tol-m", type=float, default=0.002)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--mode", choices=("touch", "fixed"), default="touch")
    ap.add_argument("--meta-src")
    ap.add_argument("--meta-dst")
    ap.add_argument("--alpha", type=float, default=0.8, help="--mode fixed 时接触帧的合手量")
    a = ap.parse_args()
    motion = Path(a.motion)
    seq_dir = motion.parents[1]
    seq = seq_dir.name.split("=", 1)[1]
    whole = seq_dir.parents[1]                       # …/whole_body
    out_seq = Path(a.out_root) / "whole_body" / "soma" / seq_dir.name
    if out_seq.exists() and not a.dry_run:
        print(f"{out_seq} 已存在，拒绝")
        return 2
    if bool(a.meta_src) != bool(a.meta_dst):
        print("--meta-src 与 --meta-dst 要一起给")
        return 2
    meta_src = Path(a.meta_src) / seq / "v2d_meta.json" if a.meta_src else None
    meta_dst = Path(a.meta_dst) / seq / "v2d_meta.json" if a.meta_dst else None
    if meta_src and not meta_src.is_file():
        print(f"没有 {meta_src}")
        return 2
    if meta_dst and meta_dst.exists() and not a.dry_run:
        print(f"{meta_dst} 已存在，拒绝")
        return 2
    t = pq.read_table(motion)
    row = t.to_pylist()[0]
    names = list(row["robot_joint_names"])
    qj = np.asarray(row["robot_joint_positions"], dtype=float)               # (T,J)
    T = qj.shape[0]
    rob = Robot(a.urdf)
    root_p = np.asarray(row["robot_root_position"], dtype=float)
    root_R = quat_mat(np.asarray(row["robot_root_wxyz"], dtype=float), "wxyz")
    qd = {n: qj[:, i] for i, n in enumerate(names)}
    sides = list(row["hand_sides"])
    contact = {s: np.asarray(c, dtype=bool) for s, c in zip(sides, row["hand_contact_active"])}
    ee = np.asarray(row["ee_pose_w"], dtype=float)                          # (T,E,7)
    ee_names = list(row["ee_link_names"])
    print(f"参考 {motion}：{T} 帧，{len(names)} 个关节；URDF {a.urdf}（根 {rob.root}）；模式 {a.mode}" + (f"，α {a.alpha}" if a.mode == "fixed" else ""))

    # 1. 正向运动学自检
    palms = {f"{s}_hand_palm_link" for s in sides}
    fkp = rob.fk(root_p, root_R, qd, palms)
    order = None
    for s in sides:
        ln = f"{s}_hand_palm_link"
        k = ee_names.index(ln)
        dp = np.linalg.norm(fkp[ln][0] - ee[:, k, :3], axis=1).max()
        angs = {}
        for o in ("xyzw", "wxyz"):
            Rq = quat_mat(ee[:, k, 3:], o)
            c = np.clip((np.einsum("nij,nij->n", Rq, fkp[ln][1]) - 1) / 2, -1, 1)
            angs[o] = float(np.degrees(np.arccos(c)).max())
        best = min(angs, key=angs.get)
        print(f"  自检 {ln}：位置最大差 {dp * 1000:.2f} mm；朝向最大差 xyzw {angs['xyzw']:.3f}°、wxyz {angs['wxyz']:.3f}°")
        if dp > a.fk_tol_m or angs[best] > 1.0:
            print(f"FK_SELFCHECK_FAIL：{ln} 对不上，不写文件")
            return 3
        order = order or best
    print(f"  自检通过：ee_pose_w 的四元数是 {order}")

    # 2. 物体表面
    mesh = trimesh.load(row["object_mesh_paths"][0], force="mesh")
    pts, fid = trimesh.sample.sample_surface(mesh, a.surf, seed=0)
    nrm = mesh.face_normals[fid]
    tree = cKDTree(pts)
    op = np.asarray(row["object_body_position"], dtype=float)[:, 0]
    oR = quat_mat(np.asarray(row["object_body_wxyz"], dtype=float)[:, 0], "wxyz")

    def sdist(p_world: np.ndarray, t_: int) -> np.ndarray:
        p_obj = (p_world - op[t_]) @ oR[t_]          # 世界 → 物体系（R^T (p − o)）
        _, i = tree.query(p_obj)
        return np.einsum("ij,ij->i", p_obj - pts[i], nrm[i])

    new_q = qj.copy()
    report = []
    for s in sides:
        c = contact.get(s, np.zeros(T, bool))
        if not c.any():
            report.append(f"  {s} 手：参考里没有接触帧，不改")
            continue
        jidx, jclose = {}, {}
        for f, js in FINGERS.items():
            for jn in js:
                full = f"{s}_hand_{jn}_joint"
                j = rob.joints[full]
                end = j["lower"] if abs(j["lower"]) >= abs(j["upper"]) else j["upper"]
                jidx[full] = names.index(full)
                jclose[full] = end * a.close_frac
        lp = {f: [(f"{s}_hand_{ln}_link", rob.link_points(f"{s}_hand_{ln}_link")) for ln in LINKS[f]] for f in FINGERS}
        want = {ln for f in lp for ln, _ in lp[f]}
        frames = np.flatnonzero(c)
        # 合手方向自检：第一个接触帧上手指全部合到 jclose，拇指末节到食指、中指末节的距离都要变小
        t0 = frames[:1]
        q_open = {n: qj[t0, i] for i, n in enumerate(names)}
        q_shut = dict(q_open)
        for full, end in jclose.items():
            q_shut[full] = np.array([end])
        Lo = rob.fk(root_p[t0], root_R[t0], q_open, want)
        Ls = rob.fk(root_p[t0], root_R[t0], q_shut, want)
        for other in ("index_1", "middle_1"):
            a_, b_ = f"{s}_hand_thumb_2_link", f"{s}_hand_{other}_link"
            d_open = float(np.linalg.norm(Lo[a_][0] - Lo[b_][0]))
            d_shut = float(np.linalg.norm(Ls[a_][0] - Ls[b_][0]))
            print(f"  合手方向自检 {s} 手：拇指 2 节到 {other} 原点 {d_open * 1000:.1f} → {d_shut * 1000:.1f} mm")
            if not d_shut < d_open:
                print(f"CLOSE_DIRECTION_FAIL：{s} 手合到底时拇指与 {other} 没有靠近，不写文件")
                return 3
        alpha = {f: np.zeros(T) for f in FINGERS}
        touched = {f: 0 for f in FINGERS}
        steps = np.arange(0.0, 1.0 + 1e-9, a.step)
        for t_ in (frames if a.mode == "touch" else []):
            for f, js in FINGERS.items():
                # 一次算完这一帧这根手指所有 α 档位（N = 档位数）
                N = len(steps)
                qq = {n: np.full(N, qj[t_, i]) for i, n in enumerate(names)}
                for jn in js:
                    full = f"{s}_hand_{jn}_joint"
                    q0 = qj[t_, jidx[full]]
                    qq[full] = q0 + steps * (jclose[full] - q0)
                L = rob.fk(np.repeat(root_p[t_:t_ + 1], N, 0), np.repeat(root_R[t_:t_ + 1], N, 0), qq, want)
                dmin = np.full(N, np.inf)
                for ln, v in lp[f]:
                    p, R = L[ln]
                    pw = p[:, None, :] + np.einsum("nij,vj->nvi", R, v)          # (N,V,3)
                    dmin = np.minimum(dmin, sdist(pw.reshape(-1, 3), t_).reshape(N, -1).min(1))
                hit = np.flatnonzero(dmin <= a.margin_m)
                if hit.size == 0:
                    alpha[f][t_] = a.free_alpha
                else:
                    touched[f] += 1
                    alpha[f][t_] = steps[max(hit[0] - 1, 0)] if hit[0] > 0 else 0.0
        if a.mode == "fixed":
            for f in FINGERS:
                alpha[f][frames] = a.alpha
        for f, js in FINGERS.items():
            al = ramp_alpha(alpha[f], c, a.ramp, a.smooth)
            for jn in js:
                full = f"{s}_hand_{jn}_joint"
                i = jidx[full]
                new_q[:, i] = qj[:, i] + al * (jclose[full] - qj[:, i])
            af = alpha[f][frames]
            report.append(f"  {s} 手 {f}：接触 {len(frames)} 帧，碰到表面 {touched[f]} 帧（{touched[f] / len(frames):.1%}）；"
                          f"接触帧 α 分位 10/50/90 {np.percentile(af, 10):.2f} / {np.percentile(af, 50):.2f} / {np.percentile(af, 90):.2f}")
        # 合手后：各指点的最小有向距离（接触帧），拇指尖与食指尖的距离变化
        qn = {n: new_q[:, i] for i, n in enumerate(names)}
        L1 = rob.fk(root_p, root_R, qn, want)
        L0 = rob.fk(root_p, root_R, qd, want)
        pen = []
        for t_ in frames[:: max(1, len(frames) // 60)]:
            m = np.inf
            for f in FINGERS:
                for ln, v in lp[f]:
                    p, R = L1[ln]
                    m = min(m, sdist(p[t_] + v @ R[t_].T, t_).min())
            pen.append(m)
        tip0 = np.linalg.norm(L0[f"{s}_hand_thumb_2_link"][0] - L0[f"{s}_hand_index_1_link"][0], axis=1)[frames]
        tip1 = np.linalg.norm(L1[f"{s}_hand_thumb_2_link"][0] - L1[f"{s}_hand_index_1_link"][0], axis=1)[frames]
        report.append(f"  {s} 手：合手后接触帧手指点最小有向距离 分位 10/50/90 {np.percentile(pen, 10) * 1000:.1f} / {np.percentile(pen, 50) * 1000:.1f} / "
                      f"{np.percentile(pen, 90) * 1000:.1f} mm（负 = 穿进物体）；拇指 2 节与食指 1 节原点距离中位 {np.median(tip0) * 1000:.1f} → {np.median(tip1) * 1000:.1f} mm")
    print("\n".join(report))
    changed = int((np.abs(new_q - qj) > 1e-9).any(1).sum())
    print(f"改了 {changed}/{T} 帧的手指关节；手指以外的关节最大改动 "
          f"{np.abs(np.delete(new_q - qj, [i for i, n in enumerate(names) if '_hand_' in n], axis=1)).max():.3g} rad")
    if a.dry_run:
        print("dry-run：不写文件")
        print("FINGER_CLOSE_DONE")
        return 0

    # 5. 写出
    out_seq.mkdir(parents=True)
    shutil.copytree(seq_dir / "object", out_seq / "object")
    (out_seq / "robot_name=g1").mkdir()
    old = t.column("robot_joint_positions")
    arr = pa.array([new_q.tolist()], type=old.type)
    t2 = t.set_column(t.schema.get_field_index("robot_joint_positions"), t.schema.field("robot_joint_positions"), arr)
    # 网格路径换成新目录（只换前缀，其余照抄）
    swapped = 0
    for col in ("object_mesh_paths", "object_urdf_paths"):
        if col in t2.column_names:
            old_paths = t2.column(col).to_pylist()[0]
            vals = [[p.replace(str(seq_dir), str(out_seq)) for p in old_paths]]
            swapped += sum(n != o for n, o in zip(vals[0], old_paths))
            t2 = t2.set_column(t2.schema.get_field_index(col), t2.schema.field(col), pa.array(vals, type=t2.schema.field(col).type))
    if swapped == 0:
        shutil.rmtree(out_seq)
        print(f"MESH_PATH_FAIL：parquet 里的网格、URDF 路径没有一条以 {seq_dir} 开头，新目录已撤回")
        return 3
    print(f"网格与 URDF 路径换到新目录 {swapped} 条：{t2.column('object_mesh_paths').to_pylist()[0]}")
    pq.write_table(t2, out_seq / "robot_name=g1" / "data.parquet")
    st_src = whole / "reconstructed_stage"
    st_dst = Path(a.out_root) / "whole_body" / "reconstructed_stage"
    st_dst.mkdir(parents=True, exist_ok=True)
    for f in st_src.glob(f"{seq}_*"):
        shutil.copy2(f, st_dst / f.name)
    print(f"写出 {out_seq}（parquet、object/）与 {st_dst} 里这一集的 {len(list(st_src.glob(f'{seq}_*')))} 个支撑面文件")
    if meta_src:
        meta_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(meta_src, meta_dst)
        print(f"元数据 {meta_src} → {meta_dst}")
    print("FINGER_CLOSE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
