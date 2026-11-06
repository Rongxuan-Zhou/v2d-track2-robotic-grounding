import pathlib
"""residual_v51: v5.1 之后剩下的误差在哪里？每一类修到完美值多少 AUC？

⚠️ 读 Tier 1 真值，仅用于分析。

残余漂移 E = 修正后位姿相对真值：E_R = R_hat·R_true^T，E_x = x_hat − x_true。
按地面法向 n 分解：旋转 = 偏航（绕 n 的扭转）+ 倾斜（摆动）；平移 = 水平 + 竖直。
再按帧分静止/运动（Tier 2 窗口判据，推理时可得）。
上帝视角 AUC：只把某一类残差清零、其余保留，看能涨多少——决定下一步找哪种信号。
"""
import os,sys,glob,json,warnings; warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "libs" / "v2d_infra")); sys.path.insert(0, str(HERE))
import numpy as np, pyarrow.parquet as pq, add_metrics as A
from scipy.spatial.transform import Rotation as Rot
from soma_fk import col, R_CV
from paths import TIER2
from denoise_v1 import meta, T1, window_rot, window_tr
from denoise_v4 import to_soma, mat_to_q
from v51 import denoise_v51, surface_tops, stable_ups

M = lambda q: A.quat_wxyz_to_matrix(q)


def rot_between(a, b):
    v = np.cross(a, b); c = float(a @ b); s = np.linalg.norm(v)
    if s < 1e-12: return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s**2)


def split(ER, n):
    """E_R = Swing·Twist(绕 n)。返回 (Swing, Twist) 两组矩阵。"""
    Sw = np.stack([rot_between(n, R @ n) for R in ER])
    Tw = np.einsum("tji,tjk->tik", Sw, ER)
    return Sw, Tw


def compose(pt, R_err, x_err):
    q = pt.copy()
    q[:, 3:7] = mat_to_q(np.einsum("tij,tjk->tik", R_err, M(pt[:, 3:7]))); q[:, :3] = pt[:, :3] + x_err
    return q


rows = []; stats = {k: {"静止": [], "运动": []} for k in ("偏航°", "倾斜°", "水平mm", "竖直mm")}
for f1 in sorted(glob.glob(T1 + "/data/chunk-000/*.parquet")):
    f2 = f1.replace("tier_1_multiview_caption", "tier_2_synthetic_noise")
    t1, t2 = pq.read_table(f1), pq.read_table(f2)
    ei = int(np.asarray(col(t1, "episode_index")).ravel()[0])
    vis = np.asarray(col(t1, "observation.object.visible")).astype(bool).ravel()
    p1 = to_soma(col(t1, "observation.object.pose")[vis]); p2 = to_soma(col(t2, "observation.object.pose")[vis])
    g = json.load(open(TIER2 / meta[ei]["ground_plane"]))["plane"]
    s_ = np.linalg.norm(g[:3]); n = R_CV @ (np.array(g[:3]) / s_); d = g[3] / s_
    if np.median(p2[:, :3] @ n + d) < 0: n, d = -n, -d
    v = A.sample_mesh_vertices(os.path.join(T1, meta[ei]["mesh"]), n=500, seed=ei)
    auc = lambda q: A.auc_from_add(A.add_per_frame(v, p1, q))[0]
    ph = denoise_v51(p2, n, d, surface_tops(TIER2 / meta[ei]["support_surface"]), stable_ups(str(TIER2 / meta[ei]["mesh"])))
    ER = np.einsum("tij,tkj->tik", M(ph[:, 3:7]), M(p1[:, 3:7])); Ex = ph[:, :3] - p1[:, :3]
    Sw, Tw = split(ER, n)
    ang = lambda R: np.degrees(np.linalg.norm(Rot.from_matrix(R).as_rotvec(), axis=1))
    ev = Ex @ n; eh = Ex - np.outer(ev, n)
    st = (window_rot(p2, 91) < 23.5) & (window_tr(p2, 91) < 0.06)
    for k, val in (("偏航°", ang(Tw)), ("倾斜°", ang(Sw)), ("水平mm", 1000*np.linalg.norm(eh, axis=1)), ("竖直mm", 1000*np.abs(ev))):
        stats[k]["静止"].append(val[st]); stats[k]["运动"].append(val[~st])
    I = np.tile(np.eye(3), (len(p1), 1, 1)); Z = np.zeros_like(Ex)
    mv = ~st
    Tw_s = Tw.copy(); Tw_s[mv] = I[mv]                       # 只修运动帧的偏航
    rows.append(dict(
        ep=ei, base=auc(ph),
        no_yaw=auc(compose(p1, Sw, Ex)),                       # 偏航清零
        no_tilt=auc(compose(p1, Tw, Ex)),                      # 倾斜清零
        no_rot=auc(compose(p1, I, Ex)),                        # 旋转全清零
        no_hx=auc(compose(p1, ER, np.outer(ev, n))),          # 水平平移清零
        no_tr=auc(compose(p1, ER, Z)),                          # 平移全清零
        yaw_static=auc(compose(p1, np.einsum("tij,tjk->tik", Sw, np.where(st[:, None, None], I, Tw)), Ex)),
        yaw_moving=auc(compose(p1, np.einsum("tij,tjk->tik", Sw, np.where(st[:, None, None], Tw, I)), Ex)),
        f_static=st.mean()))

print("v5.1 残余误差（中位 / p90），按 Tier 2 窗口判据分静止/运动帧")
print(f"{'':<8}{'静止 中位':>10}{'静止 p90':>10}{'运动 中位':>10}{'运动 p90':>10}")
for k in stats:
    a = np.concatenate(stats[k]["静止"]); b = np.concatenate(stats[k]["运动"])
    print(f"{k:<8}{np.median(a):>10.1f}{np.percentile(a,90):>10.1f}{np.median(b):>10.1f}{np.percentile(b,90):>10.1f}")
print(f"静止帧占比中位 {100*np.median([r['f_static'] for r in rows]):.0f}%")
print("\n上帝视角：只清零某一类残差，其余保留（30 条均值）")
m = lambda k: np.mean([r[k] for r in rows])
for k, name in (("base","v5.1 本身"),("no_yaw","偏航清零"),("yaw_static","  仅静止帧偏航清零"),("yaw_moving","  仅运动帧偏航清零"),
                ("no_tilt","倾斜清零"),("no_rot","旋转全清零"),("no_hx","水平平移清零"),("no_tr","平移全清零")):
    print(f"   {name:<18} {m(k):.3f}   ({m(k)-m('base'):+.3f})")
