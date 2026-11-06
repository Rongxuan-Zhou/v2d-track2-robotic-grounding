import pathlib
"""overfit_audit: v5.1 过拟合排查。

⚠️ 读 Tier 1 真值做评测与生成新噪声，仅用于分析。

v5.1 的静止检测阈值、门限、"先 v5 再退"的分层设计，都是在唯一一份 Tier 2 噪声实现上挑的，
样本只有 30 条序列。三个层面的过拟合，各用一种检验：
  P0 复现：v1 0.593 / v5 0.658 / v5.1 0.693
  P1 配对 bootstrap：v5.1−v5、v5.1−v1 的 95% 置信区间
  P2 敏感性：真实 Tier 2 上逐个参数扫，看是平台还是尖峰
  P3 选择偏差：1000 次对半切分，训练半挑变体、测试半评估
  判据（看结果前写死）：P1 v5.1−v5 的 95%CI 不含 0；P2 默认值距最优 ≤0.01、相邻 ≤0.02；
     P3 乐观偏差 <0.01；P4 新噪声上 v5.1−v5 ≥ +0.015 且 v5.1−v1 ≥ +0.06，v5.1 ≤ v5 则判过拟合退回 v5。
  局限：P4 只检验对噪声实现的过拟合；30 个物体/场景不变，物体层面只有 P3 部分覆盖。
  P4 全新噪声：按实测腐蚀模型（截断随机游走）在 Tier 1 上重新生成 K 份噪声，
     生成器先与真实 Tier 2 对统计量；新噪声上提升仍成立 ⇒ 不是过拟合到这一次实现
"""
import os,sys,glob,json,warnings; warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "libs" / "v2d_infra")); sys.path.insert(0, str(HERE))
import numpy as np, pyarrow.parquet as pq, add_metrics as A
from scipy.spatial.transform import Rotation as Rot
from soma_fk import col, R_CV
from paths import TIER2
from denoise_v1 import meta, T1
from denoise_v4 import to_soma
from support_bridge import integrate, apply
from v51 import denoise_v51, surface_tops, stable_ups, DEFAULT

M = lambda q: A.quat_wxyz_to_matrix(q)
rng = np.random.default_rng(0)

# ── 载入 30 条 ──
E = []
for f1 in sorted(glob.glob(T1 + "/data/chunk-000/*.parquet")):
    f2 = f1.replace("tier_1_multiview_caption", "tier_2_synthetic_noise")
    t1, t2 = pq.read_table(f1), pq.read_table(f2)
    ei = int(np.asarray(col(t1, "episode_index")).ravel()[0])
    vis = np.asarray(col(t1, "observation.object.visible")).astype(bool).ravel()
    p1 = to_soma(col(t1, "observation.object.pose")[vis]); p2 = to_soma(col(t2, "observation.object.pose")[vis])
    g = json.load(open(TIER2 / meta[ei]["ground_plane"]))["plane"]
    s_ = np.linalg.norm(g[:3]); n = R_CV @ (np.array(g[:3]) / s_); d = g[3] / s_
    if np.median(p2[:, :3] @ n + d) < 0: n, d = -n, -d
    E.append(dict(ei=ei, p1=p1, p2=p2, n=n, d=d,
                  v=A.sample_mesh_vertices(os.path.join(T1, meta[ei]["mesh"]), n=500, seed=ei),
                  tops=surface_tops(TIER2 / meta[ei]["support_surface"]),
                  ups=stable_ups(str(TIER2 / meta[ei]["mesh"]))))
auc = lambda e, q: A.auc_from_add(A.add_per_frame(e["v"], e["p1"], q))[0]


def run(e, p, mode, P={}):
    if mode == "raw": return auc(e, p)
    if mode == "v1":  return auc(e, apply(p, *integrate(p, _st(p))))
    return auc(e, denoise_v51(p, e["n"], e["d"], e["tops"], e["ups"], P, mode=mode, truth=e["p1"]))


def _st(p):
    from denoise_v1 import window_rot, window_tr
    return (window_rot(p, 91) < 23.5) & (window_tr(p, 91) < 0.06)


VARS = ["raw", "v1", "ref", "flat", "tier", "oracle"]
NAME = {"raw": "raw", "v1": "v1", "ref": "v5", "flat": "平权多假设", "tier": "v5.1", "oracle": "O_3dof"}

# ── P0 复现 ──
real = {k: np.array([run(e, e["p2"], k) for e in E]) for k in VARS}
print("P0 复现（真实 Tier 2）")
for k in VARS: print(f"   {NAME[k]:<10} {real[k].mean():.3f}")
print(f"   期望：v1 0.593 / v5 0.658 / v5.1 0.693 / O_3dof 0.701")

# ── P1 配对 bootstrap ──
print("\nP1 配对 bootstrap（10000 次，按序列重采样）")
for a, b in (("tier", "ref"), ("tier", "v1"), ("ref", "v1")):
    dlt = real[a] - real[b]; bs = np.array([dlt[rng.integers(0, 30, 30)].mean() for _ in range(10000)])
    print(f"   {NAME[a]} − {NAME[b]:<6} 均值 {dlt.mean():+.3f}  95%CI [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]"
          f"  变好/变差/持平 {int((dlt>1e-3).sum())}/{int((dlt<-1e-3).sum())}/{int((abs(dlt)<=1e-3).sum())}")

# ── P3 选择偏差 ──
print("\nP3 选择偏差（1000 次对半切分：训练半挑最好的变体，测试半评估）")
cand = ["v1", "ref", "flat", "tier"]
chosen = []; opt = []; test_sel = []; test_tier = []
for _ in range(1000):
    idx = rng.permutation(30); tr, te = idx[:15], idx[15:]
    best = max(cand, key=lambda k: real[k][tr].mean()); chosen.append(best)
    opt.append(real[best][tr].mean() - real[best][te].mean())
    test_sel.append(real[best][te].mean()); test_tier.append(real["tier"][te].mean())
from collections import Counter
print(f"   被选中次数: {dict(Counter(chosen))}")
print(f"   '挑最好再报'的乐观偏差（训练半−测试半）均值 {np.mean(opt):+.4f}")
print(f"   测试半：挑选流程 {np.mean(test_sel):.3f}  vs 固定 v5.1 {np.mean(test_tier):.3f}")

# ── P4 全新噪声 ──
def drift_true(e, p):
    DR = np.einsum("tij,tkj->tik", M(p[:, 3:7]), M(e["p1"][:, 3:7])); return DR, p[:, :3] - e["p1"][:, :3]

inc_r, inc_x, mag_r, mag_x, sat_r, ac = [], [], [], [], [], []
for e in E:
    DR, Dx = drift_true(e, e["p2"])
    rv = Rot.from_matrix(DR).as_rotvec(); ang = np.degrees(np.linalg.norm(rv, axis=1)); xm = 1000 * np.linalg.norm(Dx, axis=1)
    step = Rot.from_matrix(np.einsum("tij,tkj->tik", DR[1:], DR[:-1])).as_rotvec()
    free = ang[1:] < 25
    inc_r.append(step[free]); inc_x.append(np.diff(Dx, axis=0)[xm[1:] < 25])
    mag_r.append(ang.mean()); mag_x.append(xm.mean()); sat_r.append((ang > 29).mean())
    a = ang - ang.mean(); ac.append((a[60:] * a[:-60]).mean() / (a.var() + 1e-12))
sig_r = np.sqrt(np.mean(np.concatenate(inc_r) ** 2)); sig_x = np.sqrt(np.mean(np.concatenate(inc_x) ** 2))
real_stats = (np.mean(mag_r), np.mean(mag_x), np.mean(sat_r), np.mean(ac))


def gen(e, seed):
    g = np.random.default_rng(seed); T = len(e["p1"])
    D = np.eye(3); x = np.zeros(3); DR = np.empty((T, 3, 3)); Dx = np.empty((T, 3))
    cr, cx = np.radians(30), 0.030
    for t in range(T):
        if t:
            D = Rot.from_rotvec(g.normal(0, sig_r, 3)).as_matrix() @ D
            rv = Rot.from_matrix(D).as_rotvec(); a = np.linalg.norm(rv)
            if a > cr: D = Rot.from_rotvec(rv * cr / a).as_matrix()
            x = x + g.normal(0, sig_x, 3); nx = np.linalg.norm(x)
            if nx > cx: x = x * cx / nx
        DR[t] = D; Dx[t] = x
    q = e["p1"].copy()
    q[:, 3:7] = Rot.from_matrix(np.einsum("tij,tjk->tik", DR, M(e["p1"][:, 3:7]))).as_quat()[:, [3, 0, 1, 2]]
    q[:, :3] = e["p1"][:, :3] + Dx
    return q, DR, Dx


print(f"\nP4 全新噪声。生成器步长（从真实 Tier 2 未饱和帧估计）：旋转 σ={np.degrees(sig_r):.3f}°/轴/帧，平移 σ={1000*sig_x:.3f} mm/轴/帧")
syn = []
for e in E:
    q, DR, Dx = gen(e, 999 + e["ei"])
    ang = np.degrees(np.linalg.norm(Rot.from_matrix(DR).as_rotvec(), axis=1)); a = ang - ang.mean()
    syn.append((ang.mean(), 1000 * np.linalg.norm(Dx, axis=1).mean(), (ang > 29).mean(),
                (a[60:] * a[:-60]).mean() / (a.var() + 1e-12)))
syn = np.mean(syn, axis=0)
print(f"   {'':<10}{'平均转角°':>10}{'平均平移mm':>11}{'撞上限比例':>11}{'lag60自相关':>12}")
print(f"   {'真实':<10}{real_stats[0]:>10.1f}{real_stats[1]:>11.1f}{real_stats[2]:>11.2f}{real_stats[3]:>12.2f}")
print(f"   {'生成':<10}{syn[0]:>10.1f}{syn[1]:>11.1f}{syn[2]:>11.2f}{syn[3]:>12.2f}")

# ── P2 敏感性 ──
print("\nP2 敏感性（v5.1 均值；每格 真实Tier2/全新噪声种子0；★ = 默认值）")
F0 = [gen(e, 424242 + e["ei"])[0] for e in E]
SWEEP = {"ref_tilt": [25, 30, 35, 40, 45], "ref_h": [0.03, 0.04, 0.05, 0.07, 0.10],
         "fb_tilt": [30, 33, 36, 40], "fb_h": [0.03, 0.035, 0.045, 0.06],
         "margin": [0.0, 0.25, 0.5, 1.0, 2.0], "eq_ang": [2, 5, 10], "eq_h": [0.005, 0.01, 0.02],
         "st_rot": [18.0, 21.0, 23.5, 26.0, 30.0], "st_tr": [0.04, 0.05, 0.06, 0.08]}
for k, vals in SWEEP.items():
    cells = []
    for val in vals:
        m = np.mean([run(e, e["p2"], "tier", {k: val}) for e in E])
        mf = np.mean([run(e, q, "tier", {k: val}) for e, q in zip(E, F0)])
        cells.append(f"{val}{'★' if val == DEFAULT[k] else ''}:{m:.3f}/{mf:.3f}")
    print(f"   {k:<9} " + "  ".join(cells))

K = 5
fresh = {k: [] for k in VARS}
for s in range(K):
    per = {k: [] for k in VARS}
    for e in E:
        q, _, _ = gen(e, 100000 * (s + 1) + e["ei"])
        for k in VARS: per[k].append(run(e, q, k))
    for k in VARS: fresh[k].append(np.mean(per[k]))
    print(f"   种子 {s}: " + "  ".join(f"{NAME[k]} {np.mean(per[k]):.3f}" for k in VARS))
print(f"   {K} 份均值: " + "  ".join(f"{NAME[k]} {np.mean(fresh[k]):.3f}±{np.std(fresh[k]):.3f}" for k in VARS))
dt = np.array(fresh["tier"]) - np.array(fresh["ref"]); dv = np.array(fresh["tier"]) - np.array(fresh["v1"])
print(f"   新噪声上 v5.1−v5 = {dt.mean():+.3f}（真实 {real['tier'].mean()-real['ref'].mean():+.3f}），"
      f"v5.1−v1 = {dv.mean():+.3f}（真实 {real['tier'].mean()-real['v1'].mean():+.3f}）")
print(f"   新噪声上 v5.1 / O_3dof = {np.mean(fresh['tier'])/np.mean(fresh['oracle']):.3f}（真实 {real['tier'].mean()/real['oracle'].mean():.3f}）")
