import pathlib
"""support_bridge: 静止段用支撑面修 3 个绝对量 + 运动段布朗桥插值，能到多少？

⚠️ O_* 变体读 Tier 1 真值，仅用于分析；R_real 只用 Tier 2 观测与 Tier 2 的 ground_plane。

腐蚀约定与 v1 一致：obs_R = D_R · true_R（世界系左乘），obs_x = true_x + D_x。
物体静置在支撑面上时，能从几何得到的只有 3 个量：沿地面法向 n 的高度，以及 D_R 把 n 映到哪
（倾斜，2 自由度）。绕 n 的偏航与水平位置不可观测，保留 v1 的估计。
随机游走在两端已知时，中间的最优估计是线性插值（布朗桥）；v1 在运动段保持上一值。

变体：
  v1          复现 denoise_v1（自检，应为 0.593）
  O_all       静止段 D 的 6 自由度取真值 + 布朗桥        （"静止锚点+桥"结构的上限）
  O_3dof      静止段只把高度与倾斜取真值 + 布朗桥       （本候选方向的上限）
  O_3dof_hold 同上但运动段保持                          （拆出布朗桥的贡献）
  R_real      同面同高假设：参考取第 0 帧（d≈0），门限内才修，否则视为换了支撑面跳过
预设判据：O_3dof < 0.65 → 放弃；≥ 0.70 → 值得做，再看 R_real 相对 v1 是否 ≥ +0.07。
"""
import os,sys,glob,json,warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "libs" / "v2d_infra"))
import numpy as np, pyarrow.parquet as pq, add_metrics as A
from soma_fk import col, R_CV
from paths import TIER2
from denoise_v1 import window_rot, window_tr, nq, qmul, qinv, meta, T1
from denoise_v4 import to_soma, segments, mat_to_q

M = lambda q: A.quat_wxyz_to_matrix(q)
TILT_GATE, H_GATE = np.radians(35), 0.05


def integrate(p, st):
    """v1 积分器，返回每帧漂移 (DR:(T,3,3), Dx:(T,3))。"""
    q = nq(p[:, 3:7]); x = p[:, :3]; T = len(p)
    dq = np.tile([1., 0, 0, 0], (T, 1)); dx = np.zeros((T, 3))
    for t in range(1, T):
        dq[t] = dq[t-1]; dx[t] = dx[t-1]
        if st[t] and st[t-1]:
            inc = qmul(q[t], qinv(q[t-1]))
            if inc[0] < 0: inc = -inc
            dq[t] = nq(qmul(inc, dq[t-1])); dx[t] = dx[t-1] + (x[t] - x[t-1])
    return M(dq), dx


def apply(p, DR, Dx):
    out = p.copy()
    out[:, 3:7] = mat_to_q(np.einsum("tji,tjk->tik", DR, M(p[:, 3:7])))   # true = D^T obs
    out[:, :3] = p[:, :3] - Dx
    return out


def rot_between(a, b):
    """把单位向量 a 转到 b 的最小旋转。"""
    v = np.cross(a, b); c = float(a @ b); s = np.linalg.norm(v)
    if s < 1e-12: return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s**2)


def slerp_R(R0, R1, w):
    from scipy.spatial.transform import Rotation as Rot, Slerp
    return Slerp([0, 1], Rot.from_matrix(np.stack([R0, R1])))(w).as_matrix()


def correct(DR, Dx, segs, target, n, bridge):
    """每个静止段求一个常数修正（旋转 C 左乘、平移 c 相加），运动段按 bridge 插值或保持。
    target(s,e) -> (目标 D·n 方向 或 None, 目标 n·Dx 或 None, 目标整体 D 或 None)"""
    DR = DR.copy(); Dx = Dx.copy(); T = len(DR)
    fixed = []
    for s, e in segs:
        z_img, h_d, full = target(s, e)
        m = (s + e) // 2
        if full is not None:
            C = full[0] @ DR[m].T; c = full[1] - Dx[m]
        else:
            C = np.eye(3) if z_img is None else rot_between(DR[m] @ n, z_img)
            c = np.zeros(3) if h_d is None else (h_d - n @ Dx[m]) * n
        DR[s:e] = np.einsum("ij,tjk->tik", C, DR[s:e]); Dx[s:e] += c
        fixed.append((s, e))
    # 运动段：两侧都有静止段时按 bridge 插值，否则保持邻近值
    for (s0, e0), (s1, e1) in zip(fixed, fixed[1:]):
        a, b = e0 - 1, s1
        if b - a <= 1: continue
        idx = np.arange(a + 1, b)
        if bridge:
            w = (idx - a) / (b - a)
            DR[idx] = slerp_R(DR[a], DR[b], w); Dx[idx] = (1 - w)[:, None] * Dx[a] + w[:, None] * Dx[b]
        else:
            DR[idx] = DR[a]; Dx[idx] = Dx[a]
    if fixed:
        DR[fixed[-1][1]:] = DR[fixed[-1][1] - 1]; Dx[fixed[-1][1]:] = Dx[fixed[-1][1] - 1]
    return DR, Dx


if __name__ == "__main__":
    res = {k: [] for k in ("raw", "v1", "O_all", "O_3dof", "O_3dof_hold", "R_real")}; fixed_frac = []
    for f1 in sorted(glob.glob(T1 + "/data/chunk-000/*.parquet")):
        f2 = f1.replace("tier_1_multiview_caption", "tier_2_synthetic_noise")
        t1, t2 = pq.read_table(f1), pq.read_table(f2)
        ei = int(np.asarray(col(t1, "episode_index")).ravel()[0])
        vis = np.asarray(col(t1, "observation.object.visible")).astype(bool).ravel()
        p1 = to_soma(col(t1, "observation.object.pose")[vis]); p2 = to_soma(col(t2, "observation.object.pose")[vis])
        v = A.sample_mesh_vertices(os.path.join(T1, meta[ei]["mesh"]), n=500, seed=ei)
        auc = lambda q: A.auc_from_add(A.add_per_frame(v, p1, q))[0]

        g = json.load(open(TIER2 / meta[ei]["ground_plane"]))["plane"]      # Tier 2 自己的地面
        n = R_CV @ np.array(g[:3]); n /= np.linalg.norm(n)
        if n @ p2[0, :3] + g[3] / np.linalg.norm(g[:3]) < 0: n = -n           # 统一成"朝上"
        # 注：R_CV 为对合且保距，平面在 SOMA 系的法向即 R_CV n；偏移不变

        st = (window_rot(p2, 91) < 23.5) & (window_tr(p2, 91) < 0.06)
        segs = segments(st, 10)
        DR, Dx = integrate(p2, st)
        res["raw"].append(auc(p2)); res["v1"].append(auc(apply(p2, DR, Dx)))

        DRt = np.einsum("tij,tkj->tik", M(p2[:, 3:7]), M(p1[:, 3:7]))       # 真值漂移 obs·true^T
        Dxt = p2[:, :3] - p1[:, :3]
        mid = lambda s, e: (s + e) // 2
        T_all = lambda s, e: (None, None, (DRt[mid(s, e)], Dxt[mid(s, e)]))
        T_3 = lambda s, e: (DRt[mid(s, e)] @ n, n @ Dxt[mid(s, e)], None)
        res["O_all"].append(auc(apply(p2, *correct(DR, Dx, segs, T_all, n, True))))
        res["O_3dof"].append(auc(apply(p2, *correct(DR, Dx, segs, T_3, n, True))))
        res["O_3dof_hold"].append(auc(apply(p2, *correct(DR, Dx, segs, T_3, n, False))))

        # R_real：参考 = 第 0 帧观测（d≈0）。物体朝上的轴 b 在物体系里 = R_ref^T n，参考高度 h_ref = n·x_ref
        b = M(p2[:1, 3:7])[0].T @ n; h_ref = n @ p2[0, :3]; used = [0]
        def T_real(s, e):
            m = mid(s, e); Robs = M(p2[m:m+1, 3:7])[0]
            z_img = Robs @ b                                   # 若同面朝下，D·n = R_obs·b
            h_d = n @ p2[m, :3] - h_ref                        # 若同高，n·D_x = 观测高度 - 参考高度
            ok = np.arccos(np.clip(z_img @ n, -1, 1)) < TILT_GATE and abs(h_d) < H_GATE
            used[0] += ok
            return (z_img, h_d, None) if ok else (None, None, None)
        res["R_real"].append(auc(apply(p2, *correct(DR, Dx, segs, T_real, n, True))))
        fixed_frac.append(used[0] / max(len(segs), 1))

    r = np.array(res["raw"])
    print(f"{'方案':<14}{'AUC mean':>9}{'中位':>8}{'改善':>8}")
    for k in res:
        a = np.array(res[k]); print(f"{k:<14}{a.mean():>9.3f}{np.median(a):>8.3f}{int((a > r).sum()) if k != 'raw' else 0:>6}/30")
    print(f"\nR_real 通过门限的静止段比例：中位 {100*np.median(fixed_frac):.0f}%")
    print(f"参照：v1 0.593 | 抓握段完美天花板 0.816 | 只修旋转 0.881")
    o3, rr, v1 = np.mean(res["O_3dof"]), np.mean(res["R_real"]), np.mean(res["v1"])
    verdict = ("放弃" if o3 < 0.65 else
               ("值得做，且现实版拿到了 ≥ +0.07，正式实现" if rr - v1 >= 0.07 else "上限够，但现实版不够，需要更好的参考/门限")
               if o3 >= 0.70 else "介于之间，按物体看")
    print(f"判据：O_3dof {o3:.3f}，R_real−v1 {rr - v1:+.3f} → {verdict}")
    print(f"自检：v1 {v1:.3f}（应为 0.593）")
