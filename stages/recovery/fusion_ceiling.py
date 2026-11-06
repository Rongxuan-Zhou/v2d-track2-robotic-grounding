import pathlib
"""fusion_ceiling: 视频要准到什么程度，融合进 v5.1 才值得做？（R1，纯 CPU 模拟）

⚠️ 用 Tier 1 真值模拟"视频测得的物体位姿"，仅用于分析，不进管线。

事实背景：v5.1 残差集中在运动段旋转（全清零 0.913），轨迹+支撑面对它不可观测；手的增量噪声
1.61°/轴/帧 > 物体漂移步长 0.754°，不可用；视频是唯一没用的信息源，但数据集无相机标定。

模拟视频：R_v = exp(b)·exp(n_t)·R_true，x_v = x_true + b_x + n_x
  n_t：σ_v/轴，白噪声或 AR(1)（相关时间 τ 帧）；离群：以概率 p 换成 30–180° 的随机转动（对称翻转/跟丢）
  可用率：按 30 帧成块缺失（遮挡）；b：2° / 2 cm 恒定偏置（相机对齐误差），所有配置都有
融合：在 v5.1 输出上，残差 E(t) 视作随机游走（运动帧过程噪声 σ_o，静止帧≈0），状态含恒定偏置；
  Kalman 前向 + RTS 后向，Huber IRLS 3 轮抗离群。逐轴线性化（残差中位 <8°，小角近似可用）。
判据（写死）：现实条件（离群 20%、可用 60%、τ=30、有偏置）下，融合 ≥ v5.1+0.05 = 0.743 所需的 σ_v：
  ≥5° 达标 → 视频路线可行，提议 R2；只有 ≤2° 才达标 → 搁置。
"""
import os,sys,glob,json,itertools,warnings; warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "libs" / "v2d_infra")); sys.path.insert(0, str(HERE))
import numpy as np, pyarrow.parquet as pq, add_metrics as A
from scipy.spatial.transform import Rotation as Rot
from soma_fk import col, R_CV
from paths import TIER2
from denoise_v1 import meta, T1, window_rot, window_tr
from denoise_v4 import to_soma
from v51 import denoise_v51, surface_tops, stable_ups

SO_R, SO_X = np.radians(0.754), 0.000767
M = lambda q: A.quat_wxyz_to_matrix(q)


def smooth(z, avail, moving, sig_meas, sig_proc, e0, b0, iters=3, k=2.0):
    """逐通道（最后一维）线性 RTS：状态 [E, b]，z = E + b + noise。返回 Ê (T,C)。"""
    T, C = z.shape; w = np.ones((T, C))
    for _ in range(iters):
        m_f = np.zeros((T, C, 2)); P_f = np.zeros((T, C, 2, 2))
        m = np.zeros((C, 2)); P = np.zeros((C, 2, 2)); P[:, 0, 0] = e0 ** 2; P[:, 1, 1] = b0 ** 2
        m_p_all = np.zeros((T, C, 2)); P_p_all = np.zeros((T, C, 2, 2))
        for t in range(T):
            if t:
                P = P.copy(); P[:, 0, 0] += (sig_proc if moving[t] else sig_proc * 1e-3) ** 2
            m_p_all[t] = m; P_p_all[t] = P
            if avail[t]:
                R = sig_meas ** 2 / w[t]
                S = P[:, 0, 0] + P[:, 1, 1] + 2 * P[:, 0, 1] + R
                K = np.stack([P[:, 0, 0] + P[:, 0, 1], P[:, 1, 0] + P[:, 1, 1]], -1) / S[:, None]
                r = z[t] - (m[:, 0] + m[:, 1])
                m = m + K * r[:, None]
                H = np.array([1.0, 1.0])
                P = P - np.einsum("ci,cj->cij", K, np.einsum("j,cjk->ck", H, P))
            m_f[t] = m; P_f[t] = P
        ms = m_f.copy()
        for t in range(T - 2, -1, -1):
            G = np.einsum("cij,cjk->cik", P_f[t], np.linalg.inv(P_p_all[t + 1]))
            ms[t] = m_f[t] + np.einsum("cij,cj->ci", G, ms[t + 1] - m_p_all[t + 1])
        r = np.abs(z - ms[..., 0] - ms[..., 1]) / sig_meas
        w = np.where(r <= k, 1.0, k / np.maximum(r, 1e-9))
    return ms[..., 0]


def fake_video(p_true, cfg, g):
    T = len(p_true); sv, svt, out, av, tau = cfg
    rho = np.exp(-1 / tau) if tau else 0.0
    n = np.zeros((T, 6)); eps = g.normal(size=(T, 6)) * np.r_[[np.radians(sv)] * 3, [svt] * 3]
    for t in range(T):
        n[t] = eps[t] if t == 0 else rho * n[t - 1] + np.sqrt(1 - rho ** 2) * eps[t]
    flip = g.random(T) < out
    ax = g.normal(size=(T, 3)); ax /= np.linalg.norm(ax, axis=1, keepdims=True)
    n[flip, :3] = ax[flip] * np.radians(g.uniform(30, 180, flip.sum()))[:, None]
    b = g.normal(size=3); b = b / np.linalg.norm(b) * np.radians(2); bx = g.normal(size=3); bx = bx / np.linalg.norm(bx) * 0.02
    Rv = Rot.from_rotvec(b) * Rot.from_rotvec(n[:, :3]) * Rot.from_matrix(M(p_true[:, 3:7]))
    xv = p_true[:, :3] + bx + n[:, 3:]
    avail = np.ones(T, bool)
    if av < 1:
        while avail.mean() > av:
            s = g.integers(0, T); avail[s:s + 30] = False
    return Rv, xv, avail


E = []
for f1 in sorted(glob.glob(T1 + "/data/chunk-000/*.parquet")):
    f2 = f1.replace("tier_1_multiview_caption", "tier_2_synthetic_noise")
    t1, t2 = pq.read_table(f1), pq.read_table(f2)
    ei = int(np.asarray(col(t1, "episode_index")).ravel()[0])
    vis = np.asarray(col(t1, "observation.object.visible")).astype(bool).ravel()
    p1 = to_soma(col(t1, "observation.object.pose")[vis]); p2 = to_soma(col(t2, "observation.object.pose")[vis])
    g_ = json.load(open(TIER2 / meta[ei]["ground_plane"]))["plane"]
    s_ = np.linalg.norm(g_[:3]); n_ = R_CV @ (np.array(g_[:3]) / s_); d_ = g_[3] / s_
    if np.median(p2[:, :3] @ n_ + d_) < 0: n_, d_ = -n_, -d_
    ph = denoise_v51(p2, n_, d_, surface_tops(TIER2 / meta[ei]["support_surface"]), stable_ups(str(TIER2 / meta[ei]["mesh"])))
    st = (window_rot(p2, 91) < 23.5) & (window_tr(p2, 91) < 0.06)
    E.append(dict(ei=ei, p1=p1, ph=ph, moving=~st,
                  v=A.sample_mesh_vertices(os.path.join(T1, meta[ei]["mesh"]), n=500, seed=ei)))
auc = lambda e, q: A.auc_from_add(A.add_per_frame(e["v"], e["p1"], q))[0]
base = np.mean([auc(e, e["ph"]) for e in E])


def fused(cfg, seed=0, matched=False):
    out = []
    for e in E:
        g = np.random.default_rng(seed * 1000 + e["ei"])
        Rv, xv, avail = fake_video(e["p1"], cfg, g)
        Rph = Rot.from_matrix(M(e["ph"][:, 3:7]))
        z = np.hstack([(Rph * Rv.inv()).as_rotvec(), e["ph"][:, :3] - xv])       # ≈ E − n − b
        sv, svt = np.radians(max(cfg[0], 0.1)), max(cfg[1], 1e-4)
        if matched and cfg[4]:                      # 事后修正：AR(1) 误差的等效白噪声方差放大 (1+ρ)/(1−ρ)
            rho = np.exp(-1 / cfg[4]); f = np.sqrt((1 + rho) / (1 - rho)); sv, svt = sv * f, svt * f
        Er = smooth(z[:, :3], avail, e["moving"], sv, SO_R, np.radians(0.5), np.radians(5))
        Ex = smooth(z[:, 3:], avail, e["moving"], svt, SO_X, 0.001, 0.05)
        q = e["ph"].copy()
        q[:, 3:7] = (Rot.from_rotvec(Er).inv() * Rph).as_quat()[:, [3, 0, 1, 2]]
        q[:, :3] = e["ph"][:, :3] - Ex
        out.append(auc(e, q))
    return np.mean(out)


print(f"v5.1 基线 {base:.3f}；目标 ≥ {base + 0.05:.3f}")
print(f"完美视频（σ 0.1°/1 mm、无离群、全可用；仍有 2°/2 cm 偏置）: {fused((0.1, 0.001, 0.0, 1.0, 0)):.3f}")
SV = [(2, 0.01), (5, 0.02), (10, 0.04), (20, 0.08)]
COMBO = [(0.0, 1.0, 0), (0.0, 1.0, 30), (0.2, 1.0, 30), (0.2, 0.6, 0), (0.2, 0.6, 30)]
print(f"\n{'σ_v / σ_vt':<14}" + "".join(f"{f'离{int(o*100)}%可{int(a*100)}%τ{t}':>16}" for o, a, t in COMBO))
real_col = {}
for sv, svt in SV:
    cells = []
    for o, a, t in COMBO:
        r = fused((sv, svt, o, a, t)); cells.append(r)
        if (o, a, t) == (0.2, 0.6, 30): real_col[sv] = r
    print(f"{f'{sv}° / {int(svt*100)} cm':<14}" + "".join(f"{c:>16.3f}" for c in cells))
ok = [sv for sv, r in real_col.items() if r >= base + 0.05]
print(f"\n现实条件（离群 20%、可用 60%、τ=30、偏置 2°/2 cm）下达到 {base + 0.05:.3f} 的 σ_v: {ok if ok else '无'}")
print("判据：" + ("视频路线可行，提议 R2" if ok and max(ok) >= 5 else "只有 ≤2° 才达标，搁置" if ok else "都不达标，搁置"))

print("\n── 事后修正（看过上表后加）：融合器按 AR(1) 放大测量方差 (1+ρ)/(1−ρ)，τ 视为可从残差自相关估计 ──")
print(f"{'σ_v / σ_vt':<14}{'离20%可100%τ30':>16}{'离20%可60%τ30':>16}")
real_m = {}
for sv, svt in SV:
    a = fused((sv, svt, 0.2, 1.0, 30), matched=True); b = fused((sv, svt, 0.2, 0.6, 30), matched=True); real_m[sv] = b
    print(f"{f'{sv}° / {int(svt*100)} cm':<14}{a:>16.3f}{b:>16.3f}")
okm = [sv for sv, r in real_m.items() if r >= base + 0.05]
print(f"事后修正版，现实条件下达标的 σ_v: {okm if okm else '无'}（原判据结论不变，此行单独标注）")
