import pathlib
"""oracle_root: 两级方案（先用脚修根，再用手锚定物体）的天花板有多高？

⚠️ 仅用于分析，读取 Tier 1 真值。不得被任何 Tier 2/3 管线 import（档位隔离）。

问题：v4 抓握锚定只有 0.221，归因于"手在世界系比物体脏"。手脏的唯一来源是根——
两档 77 个关节里 76 个逐字节相同，只有 Hips 旋转和根平移被污染。那么把根修好，
v4 能到多少？这决定值不值得花几天写脚部无滑移求解器。

做法：构造根残差比例 α 的人体
    Hips 旋转 = slerp(Tier1, Tier2, α)，根平移 = lerp(Tier1, Tier2, α)，其余列取 Tier 2。
α=0 是完美根，α=1 是原始 Tier 2。求解器修不到完美，所以看整条曲线，不只看 α=0。
物体位姿始终用带噪的 Tier 2——只修根，不混入别的真值。

预设判据（写在跑之前）：
  α=0 ≥ 0.80 且 α=0.25 ≥ 0.70 → 值得写脚部求解器
  α=0 ∈ [0.65, 0.80)            → 瓶颈在 v4 抓握锚定本身，先改算法
  α=0 < 0.65                    → 两级路线放弃
自检：v1_geom 必须复现 0.593（与人体无关）；α=1 必须复现 v4 的 0.221。
"""
import os,sys,glob,warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "libs" / "v2d_infra"))
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, add_metrics as A
from scipy.spatial.transform import Rotation as Rot, Slerp
from soma_fk import fk, joint_names, col
from denoise_v1 import window_rot, window_tr, meta, T1
from denoise_v4 import denoise, to_soma, LH, RH

assert joint_names()[0] == "Hips", "Hips 应是 77 关节的第 0 个"
ALPHAS = (0.0, 0.25, 0.5, 1.0)
POSE, TR = "observation.human.pose", "observation.human.translation"


def blend_root(t1, t2, alpha):
    """返回 Tier 2 表的拷贝，其中根（Hips 局部旋转 + 平移）按 α 在两档间插值。"""
    P1, P2 = col(t1, POSE), col(t2, POSE)
    assert np.array_equal(P1[:, 3:], P2[:, 3:]), "除 Hips 外的 76 个关节应逐字节相同"
    for c in ("identity_coeffs", "scale_params", "bone_length_flexibles"):
        c = "observation.human." + c
        assert np.array_equal(col(t1, c), col(t2, c)), f"{c} 两档不同，α=0 不再只差根"
    T = len(P1)
    hips = np.empty((T, 3))
    for i in range(T):
        s = Slerp([0.0, 1.0], Rot.from_rotvec(np.stack([P1[i, :3], P2[i, :3]])))
        hips[i] = s([alpha]).as_rotvec()[0]
    P = P2.astype(np.float32).copy(); P[:, :3] = hips
    tr = (1 - alpha) * col(t1, TR) + alpha * col(t2, TR)
    fl = lambda a, w: pa.FixedSizeListArray.from_arrays(pa.array(a.astype(np.float32).ravel()), w)
    t = t2.set_column(t2.schema.get_field_index(POSE), POSE, fl(P, 231))
    return t.set_column(t.schema.get_field_index(TR), TR, fl(tr, 3))


res = {"raw": [], "v1_geom": []} | {a: [] for a in ALPHAS}
per_ep = []
for f1 in sorted(glob.glob(T1 + "/data/chunk-000/*.parquet")):
    f2 = f1.replace("tier_1_multiview_caption", "tier_2_synthetic_noise")
    t1, t2 = pq.read_table(f1), pq.read_table(f2)
    ei = int(np.asarray(col(t1, "episode_index")).ravel()[0])
    vis = np.asarray(col(t1, "observation.object.visible")).astype(bool).ravel()
    p1 = to_soma(col(t1, "observation.object.pose")[vis]); p2 = to_soma(col(t2, "observation.object.pose")[vis])
    v = A.sample_mesh_vertices(os.path.join(T1, meta[ei]["mesh"]), n=500, seed=ei)
    ov = A.transform_vertices(A.sample_mesh_vertices(os.path.join(T1, meta[ei]["mesh"]), n=200, seed=ei), p2)
    st = (window_rot(p2, 91) < 23.5) & (window_tr(p2, 91) < 0.06)
    auc = lambda q: A.auc_from_add(A.add_per_frame(v, p1, q))[0]
    res["raw"].append(auc(p2))
    row = {"ep": ei, "obj": meta[ei]["object"][:18]}
    for a in ALPHAS:
        tb = t2 if a == 1.0 else (t1 if a == 0.0 else blend_root(t1, t2, a))
        if a == 0.0:   # α=0 用 Tier 1 表本身；先确认它确实只在根上与 Tier 2 不同
            blend_root(t1, t2, 0.0)
        J = fk(tb, flex=True)[vis]
        gd = np.linalg.norm(J[:, LH + RH, None, :] - ov[:, None, :, :], axis=-1).min(axis=(1, 2))
        if a == 1.0:
            res["v1_geom"].append(auc(denoise(p2, J, st, gd, grasp_thr=0.0)))
        res[a].append(auc(denoise(p2, J, st, gd))); row[a] = res[a][-1]
    row["geom"] = res["v1_geom"][-1]; per_ep.append(row)

r = np.array(res["raw"])
print(f"{'方案':<14}{'AUC mean':>9}{'中位':>8}{'改善':>8}")
for k, name in [("raw", "raw"), ("v1_geom", "v1_geom")] + [(a, f"v4 α={a:g}") for a in ALPHAS]:
    x = np.array(res[k]); print(f"{name:<14}{x.mean():>9.3f}{np.median(x):>8.3f}{int((x > r).sum()) if k != 'raw' else 0:>6}/30")

print(f"\n逐序列（按 α=0 相对 geom 的增益排序）")
print(f"{'ep':>3} {'物体':<18}{'geom':>7}" + "".join(f"{'α='+format(a,'g'):>8}" for a in ALPHAS))
for row in sorted(per_ep, key=lambda r: r[0.0] - r["geom"]):
    print(f"{row['ep']:>3} {row['obj']:<18}{row['geom']:>7.3f}" + "".join(f"{row[a]:>8.3f}" for a in ALPHAS))

m0, m25 = np.mean(res[0.0]), np.mean(res[0.25])
verdict = ("值得写脚部求解器" if m0 >= 0.80 and m25 >= 0.70 else
           "瓶颈在 v4 抓握锚定本身，先改算法" if m0 >= 0.65 else "两级路线放弃")
print(f"\n判据：α=0 {m0:.3f}，α=0.25 {m25:.3f} → {verdict}")
print(f"自检：v1_geom {np.mean(res['v1_geom']):.3f}（应为 0.593）；α=1 {np.mean(res[1.0]):.3f}（应为 0.221）")
