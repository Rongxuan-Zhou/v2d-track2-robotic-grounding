import pathlib
"""increment_noise: 手的逐帧转动增量，能不能当作物体转动增量的观测？

⚠️ 读 Tier 1 真值，仅用于分析。

N9 里整段传播失败（段末 50°），但那是几百帧小误差的累积。另一种用法是把手的
**逐帧增量**当作物体增量的带噪观测，与轨迹自身增量融合（互补/卡尔曼）。
可行的条件：手-物相对增量噪声 σ_h 小于物体漂移步长 σ_o，且不带系统性累积。

测：抓握段内（Tier 1 物体在动且手 <6 cm），
  rel(t) = ΔR_obj(t)·ΔR_hand(t)^T 的旋转向量 → σ_h（每轴每帧）
  窗口 W 内累积的相对转动 rms 随 W 的增长：∝√W 白噪声，∝W 系统性滑移
手用手掌 5 点 Procrustes；Tier 1 手 与 Tier 2 手（多了根的漂移增量）各测一次。
σ_o 取自真实 Tier 2 物体漂移的未饱和帧。
"""
import os,sys,glob,warnings; warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "libs" / "v2d_infra")); sys.path.insert(0, str(HERE))
import numpy as np, pyarrow.parquet as pq, add_metrics as A
from scipy.spatial.transform import Rotation as Rot
from soma_fk import fk, joint_names, col
from denoise_v1 import meta, T1
from denoise_v4 import to_soma, procrustes, segments, LH, RH
from static_window import gt_still

M = lambda q: A.quat_wxyz_to_matrix(q)
N = joint_names()
PALM = {s: [N.index(f"{s}Hand")] + [N.index(f"{s}Hand{f}1") for f in ("Index","Middle","Ring","Pinky")] for s in ("Left","Right")}
WS = (1, 5, 10, 30, 90)

rel = {"t1": [], "t2": []}; acc = {k: {W: [] for W in WS} for k in rel}; so = []
for f1 in sorted(glob.glob(T1 + "/data/chunk-000/*.parquet")):
    f2 = f1.replace("tier_1_multiview_caption", "tier_2_synthetic_noise")
    t1, t2 = pq.read_table(f1), pq.read_table(f2)
    ei = int(np.asarray(col(t1, "episode_index")).ravel()[0])
    vis = np.asarray(col(t1, "observation.object.visible")).astype(bool).ravel()
    p1 = to_soma(col(t1, "observation.object.pose")[vis]); p2 = to_soma(col(t2, "observation.object.pose")[vis])
    R1 = M(p1[:, 3:7])
    DR = np.einsum("tij,tkj->tik", M(p2[:, 3:7]), R1)
    ang = np.linalg.norm(Rot.from_matrix(DR).as_rotvec(), axis=1)
    st_o = Rot.from_matrix(np.einsum("tij,tkj->tik", DR[1:], DR[:-1])).as_rotvec()[ang[1:] < np.radians(25)]
    so.append(st_o)
    J = {"t1": fk(t1, flex=True)[vis], "t2": fk(t2, flex=True)[vis]}
    v200 = A.sample_mesh_vertices(os.path.join(T1, meta[ei]["mesh"]), n=200, seed=ei)
    ov = A.transform_vertices(v200, p1)
    gd = np.linalg.norm(J["t1"][:, LH + RH, None, :] - ov[:, None, :, :], axis=-1).min(axis=(1, 2))
    for s, e in segments((gd < 0.06) & ~gt_still(p1), 10):
        side = "Left" if np.linalg.norm(J["t1"][s][LH].mean(0) - p1[s, :3]) < np.linalg.norm(J["t1"][s][RH].mean(0) - p1[s, :3]) else "Right"
        for k in rel:
            Hh = np.stack([procrustes(J[k][s][PALM[side]], J[k][t][PALM[side]])[0] for t in range(s, e)])   # 相对段首
            Ho = np.einsum("tij,jk->tik", R1[s:e], R1[s].T)                                                  # 物体相对段首
            relc = np.einsum("tij,tkj->tik", Ho, Hh)                                                       # 累积相对转动
            rv = Rot.from_matrix(relc).as_rotvec()
            inc = Rot.from_matrix(np.einsum("tij,tkj->tik", relc[1:], relc[:-1])).as_rotvec()
            rel[k].append(inc)
            for W in WS:
                if e - s > W:
                    dW = Rot.from_matrix(np.einsum("tij,tkj->tik", relc[W:], relc[:-W])).as_rotvec()
                    acc[k][W].append(np.linalg.norm(dW, axis=1))

sig = lambda x: np.degrees(np.sqrt(np.mean(np.concatenate(x) ** 2)))
sig_o = sig(so) ; print(f"σ_o 物体漂移步长（真实 Tier 2，未饱和帧）: {sig_o:.3f}°/轴/帧")
for k, name in (("t1", "Tier 1 手"), ("t2", "Tier 2 手（含根漂移）")):
    sh = sig(rel[k]); print(f"σ_h 手-物相对增量（{name}）: {sh:.3f}°/轴/帧   → σ_h/σ_o = {sh/sig_o:.2f}")
    r1 = np.degrees(np.sqrt(np.mean(np.concatenate(acc[k][1]) ** 2)))
    line = "   窗口累积 rms: " + "  ".join(f"W={W}:{np.degrees(np.sqrt(np.mean(np.concatenate(acc[k][W])**2))):.1f}°"
                                        f"(白噪声预期 {r1*np.sqrt(W):.1f}°)" for W in WS)
    print(line)
print("\n判读：σ_h < σ_o 且累积 ≈ √W 增长 → 增量融合可行；累积 ≫ √W 预期 → 手-物有系统性相对转动，融合会引入偏差")
