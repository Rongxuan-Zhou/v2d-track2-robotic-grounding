import pathlib
"""grasp_rigidity: 抓握锚定这个思路本身还有没有救？

⚠️ 仅用于分析，读取 Tier 1 真值。不得被任何 Tier 2/3 管线 import（档位隔离）。

背景：oracle_root 证明完美根下 v4 也只有 0.225，失败不在手脏，在锚定算法本身。
v4 用所有 LeftHand*/RightHand* 共 25 点做 Procrustes，含会弯曲的手指。

A. Tier 1 刚性：抓握段内从真实起点位姿按手运动传播，与真值比。手坐标系两种：
   fingers = v4 的 25 点；palm = 手腕 + 4 个 *1 关节（LeftHand 的直接子节点，与手腕刚连）。
   左右手都算，另报"选离得近的那只"。
B. 天花板：v1_geom 输出 + 抓握段帧换成真值 → AUC。抓握锚定做到完美的上限。
C. 因素阶梯（Tier 2 物体 + Tier 1 手）：F0 v4 原样 / F1 换 palm / F2 +真值抓握段 / F3 +真值段起点。

预设判据（写在跑之前）：
  B < 0.65                      → 抓握锚定整体不值得，停
  A palm 抓握段 AUC ≥ 0.85       → 刚性成立，看 C 哪步跳升最大，修那一处
  A palm 抓握段 AUC < 0.60       → 刚性不成立，放弃抓握锚定
  其间                           → 按物体拆开，决定能否只对刚性抓握类启用
"""
import os,sys,glob,warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "libs" / "v2d_infra"))
import numpy as np, pyarrow.parquet as pq, add_metrics as A
from soma_fk import fk, joint_names, col
from denoise_v1 import window_rot, window_tr, nq, qmul, qinv, meta, T1
from denoise_v4 import denoise, to_soma, procrustes, segments, mat_to_q, LH, RH
from static_window import gt_still

N = joint_names()
PALM = {s: [N.index(f"{s}Hand")] + [N.index(f"{s}Hand{f}1") for f in ("Index","Middle","Ring","Pinky")]
        for s in ("Left","Right")}
FING = {"Left": LH, "Right": RH}
assert len(LH) == 25 and len(PALM["Left"]) == 5
GRASP_THR, MINLEN = 0.06, 10


def propagate(p_true, J, s, e, idx):
    """从 p_true[s] 起按手关键点 idx 的刚体运动推到 [s,e)，返回 (e-s,7) 位姿与退化帧数。"""
    R0 = A.quat_wxyz_to_matrix(p_true[s:s+1, 3:7])[0]; x0 = p_true[s, :3]
    out = np.empty((e - s, 7)); bad = 0
    for k, t in enumerate(range(s, e)):
        P, Q = J[s][idx], J[t][idx]
        sv = np.linalg.svd(P - P.mean(0), compute_uv=False)
        if sv[1] < 1e-3 * sv[0]:                 # 近共线，Procrustes 退化
            bad += 1; out[k] = out[k-1] if k else p_true[s]; continue
        R, tt = procrustes(P, Q)
        out[k, :3] = R @ x0 + tt; out[k, 3:7] = mat_to_q((R @ R0)[None])[0]
    return out, bad


def grasp_segs(p_obj, J, ov_fn):
    ov = ov_fn(p_obj)
    gd = np.linalg.norm(J[:, LH + RH, None, :] - ov[:, None, :, :], axis=-1).min(axis=(1, 2))
    return gd


A_rows = []; B_auc = []; B_cov = []; C = {k: [] for k in ("F0","F1","F2","F3")}; degen = 0
for f1 in sorted(glob.glob(T1 + "/data/chunk-000/*.parquet")):
    f2 = f1.replace("tier_1_multiview_caption", "tier_2_synthetic_noise")
    t1, t2 = pq.read_table(f1), pq.read_table(f2)
    ei = int(np.asarray(col(t1, "episode_index")).ravel()[0])
    vis = np.asarray(col(t1, "observation.object.visible")).astype(bool).ravel()
    p1 = to_soma(col(t1, "observation.object.pose")[vis]); p2 = to_soma(col(t2, "observation.object.pose")[vis])
    mesh = os.path.join(T1, meta[ei]["mesh"])
    v = A.sample_mesh_vertices(mesh, n=500, seed=ei); v200 = A.sample_mesh_vertices(mesh, n=200, seed=ei)
    J = fk(t1, flex=True)[vis]                                    # Tier 1 手（= 完美根）
    ovf = lambda p: A.transform_vertices(v200, p)

    # ── A：Tier 1 抓握段、真值起点、按手传播 ──
    moving1 = ~gt_still(p1)
    gd1 = grasp_segs(p1, J, ovf)
    segs1 = segments((gd1 < GRASP_THR) & moving1, MINLEN)
    gmask = np.zeros(len(p1), bool)
    acc = {k: [] for k in ("palm_L","palm_R","palm_near","fing_near")}
    ref_frames = []
    for s, e in segs1:
        gmask[s:e] = True
        near = "Left" if np.linalg.norm(J[s][LH].mean(0)-p1[s,:3]) < np.linalg.norm(J[s][RH].mean(0)-p1[s,:3]) else "Right"
        for k, (side, idx) in {"palm_L": ("Left", PALM["Left"]), "palm_R": ("Right", PALM["Right"]),
                               "palm_near": (near, PALM[near]), "fing_near": (near, FING[near])}.items():
            q, b = propagate(p1, J, s, e, idx); degen += b; acc[k].append(q)
        ref_frames.append(p1[s:e])
    row = {"ep": ei, "obj": meta[ei]["object"][:18], "cov": gmask[moving1].mean() if moving1.any() else 0.0,
           "nseg": len(segs1)}
    if segs1:
        ref = np.concatenate(ref_frames)
        for k in acc:
            row[k] = A.auc_from_add(A.add_per_frame(v, ref, np.concatenate(acc[k])))[0]
        # 段末漂移（palm_near）
        dr = []; dx = []
        for (s, e), q in zip(segs1, acc["palm_near"]):
            Rt = A.quat_wxyz_to_matrix(p1[e-1:e, 3:7])[0]; Rq = A.quat_wxyz_to_matrix(q[-1:, 3:7])[0]
            dr.append(np.degrees(np.arccos(np.clip((np.trace(Rt.T @ Rq) - 1) / 2, -1, 1))))
            dx.append(1000 * np.linalg.norm(q[-1, :3] - p1[e-1, :3]))
        row["drift_deg"] = float(np.median(dr)); row["drift_mm"] = float(np.median(dx))
    A_rows.append(row)

    # ── B：v1_geom + 抓握段换真值 ──
    st = (window_rot(p2, 91) < 23.5) & (window_tr(p2, 91) < 0.06)
    gd2 = grasp_segs(p2, J, ovf)
    geom = denoise(p2, J, st, gd2, grasp_thr=0.0)
    ceil = geom.copy(); ceil[gmask] = p1[gmask]
    B_auc.append(A.auc_from_add(A.add_per_frame(v, p1, ceil))[0])

    # ── C：因素阶梯 ──
    auc = lambda q: A.auc_from_add(A.add_per_frame(v, p1, q))[0]
    C["F0"].append(auc(denoise(p2, J, st, gd2)))
    # F1..F3 用本脚本的传播实现替换 v4 的抓握段：起点位姿 = 当前积分器给出的估计 or 真值
    def ladder(segs, true_start, idx_of):
        out = geom.copy()
        for s, e in segs:
            near = "Left" if np.linalg.norm(J[s][LH].mean(0)-p2[s,:3]) < np.linalg.norm(J[s][RH].mean(0)-p2[s,:3]) else "Right"
            base = p1 if true_start else out
            q, _ = propagate(base, J, s, e, idx_of(near)); out[s:e] = q
        return out
    segs2 = segments((gd2 < GRASP_THR) & ~st, MINLEN)
    C["F1"].append(auc(ladder(segs2, False, lambda sd: PALM[sd])))
    C["F2"].append(auc(ladder(segs1, False, lambda sd: PALM[sd])))
    C["F3"].append(auc(ladder(segs1, True,  lambda sd: PALM[sd])))

rows = [r for r in A_rows if r["nseg"]]
m = lambda k: np.mean([r[k] for r in rows])
print(f"A. Tier 1 抓握段内按手传播的 ADD-AUC（{len(rows)}/30 条有抓握段；Procrustes 退化帧 {degen}）")
for k, name in (("fing_near","v4 的 25 点(含手指), 近手"),("palm_near","手掌 5 点, 近手"),
                ("palm_L","手掌 5 点, 固定左手"),("palm_R","手掌 5 点, 固定右手")):
    print(f"   {name:<26} {m(k):.3f}   中位 {np.median([r[k] for r in rows]):.3f}")
print(f"   段末漂移（手掌近手）中位: 旋转 {np.median([r['drift_deg'] for r in rows]):.1f}°  平移 {np.median([r['drift_mm'] for r in rows]):.0f} mm")
print(f"   抓握段覆盖运动帧: 中位 {100*np.median([r['cov'] for r in A_rows]):.0f}%")

print(f"\n逐序列（按 palm_near 排序）")
print(f"{'ep':>3} {'物体':<18}{'段数':>4}{'覆盖':>6}{'fing':>7}{'palm':>7}{'漂移°':>7}{'漂移mm':>8}")
for r in sorted(rows, key=lambda r: r["palm_near"]):
    print(f"{r['ep']:>3} {r['obj']:<18}{r['nseg']:>4}{100*r['cov']:>5.0f}%{r['fing_near']:>7.3f}{r['palm_near']:>7.3f}{r['drift_deg']:>7.1f}{r['drift_mm']:>8.0f}")

print(f"\nB. 天花板：v1_geom + 抓握段完美 = {np.mean(B_auc):.3f}（v1_geom 0.593）")
print(f"\nC. 因素阶梯（Tier 2 物体 + Tier 1 手）")
for k, name in (("F0","v4 原样"),("F1","换手掌 5 点"),("F2","+ 真值抓握段"),("F3","+ 真值段起点")):
    print(f"   {k} {name:<14} {np.mean(C[k]):.3f}")

pa, b = m("palm_near"), np.mean(B_auc)
v = ("抓握锚定整体不值得，停" if b < 0.65 else
     "刚性成立，修 C 中跳升最大的一步" if pa >= 0.85 else
     "刚性不成立，放弃抓握锚定" if pa < 0.60 else "按物体拆开决定")
print(f"\n判据：B {b:.3f}，A palm {pa:.3f} → {v}")
