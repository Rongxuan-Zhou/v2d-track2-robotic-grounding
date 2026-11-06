"""不穿地约束：v5.3 的后处理（只用 Tier 2 可得的输入：地面平面 + 物体网格）。

事实依据（计划 A2 诊断）：Tier 2 重定向后 13/30 条穿地，最深 −143 mm；中心高度 T2−T1 基本 ±8 mm，
穿透与物体半径相关（−0.47）⇒ 旋转残差把大物体的角压进地面。真实物体永远不穿地，穿透帧必错。

约束：物体凸包最低点离地 ≥ −ε（ε=10 mm，重建精度量级，事先固定）。三种修法：
  lift   ：沿地面法向 n 抬高（对照；与"中心高度是对的"矛盾）
  rot    ：绕物体原点转，轴 = r×n（r 为最深点相对原点），二分求满足约束的最小角，≤25°，不足再抬
  rot_s  ：rot 的逐帧修正旋转向量做时间高斯平滑（σ=10 帧；漂移是平滑的），平滑后再抬一次兜底
位姿约定同 v51：p (T,7) [x,y,z,qw,qx,qy,qz]，SOMA 世界系；高度 h(X) = X·n + d。
"""
from functools import lru_cache

import numpy as np
import trimesh
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as Rot

EPS = 0.010
MAX_ROT = np.radians(25)


@lru_cache(maxsize=None)
def hull_vertices(mesh_path: str) -> np.ndarray:
    m = trimesh.load(mesh_path, force="scene")
    m = m.dump(concatenate=True) if isinstance(m, trimesh.Scene) else m
    return np.asarray(m.convex_hull.vertices, dtype=np.float64)


def _R(p):
    return Rot.from_quat(p[:, [4, 5, 6, 3]]).as_matrix()


def min_height(p, V, n, d):
    W = np.einsum("tij,vj->tvi", _R(p), V) + p[:, None, :3]
    return (W @ n + d).min(1)


def _pack(Rm, x):
    q = Rot.from_matrix(Rm).as_quat()
    out = np.empty((len(x), 7)); out[:, :3] = x; out[:, 3:7] = q[:, [3, 0, 1, 2]]
    return out


def _lift(Rm, x, V, n, d, eps):
    h = (np.einsum("tij,vj->tvi", Rm, V) + x[:, None]) @ n + d
    need = np.maximum(0.0, -eps - h.min(1))
    return x + need[:, None] * n


def enforce(p, V, n, d, mode="rot_s", eps=EPS, sigma=10.0, max_rot=MAX_ROT):
    Rm = _R(p).copy(); x = p[:, :3].copy()
    if mode == "lift":
        return _pack(Rm, _lift(Rm, x, V, n, d, eps))
    omega = np.zeros((len(p), 3))
    for t in range(len(p)):
        r = Rm[t] @ V.T                                       # (3,V) 相对原点
        h = r.T @ n + (x[t] @ n + d)
        if h.min() >= -eps:
            continue
        rl = r[:, h.argmin()]
        ax = np.cross(rl, n); s = np.linalg.norm(ax)
        if s < 1e-9:
            continue                                          # 最深点正在原点正下方：转不动，交给抬高
        ax /= s
        ok = lambda th: ((Rot.from_rotvec(ax * th).as_matrix() @ r).T @ n + (x[t] @ n + d)).min() >= -eps
        lo, hi = 0.0, max_rot
        if ok(hi):
            for _ in range(30):
                mid = (lo + hi) / 2
                lo, hi = (lo, mid) if ok(mid) else (mid, hi)
            omega[t] = ax * hi
        else:                                                 # 转满也不够：取让最低点最高的角度，余下交给抬高
            ths = np.linspace(0, max_rot, 26)
            mins = [((Rot.from_rotvec(ax * th).as_matrix() @ r).T @ n).min() for th in ths]
            omega[t] = ax * ths[int(np.argmax(mins))]
    if mode == "rot_s":
        omega = gaussian_filter1d(omega, sigma, axis=0, mode="nearest")
    Rm = np.einsum("tij,tjk->tik", Rot.from_rotvec(omega).as_matrix(), Rm)
    return _pack(Rm, _lift(Rm, x, V, n, d, eps))


def enforce_tops(q, p_obs, V, n, d, tops, P, eps=EPS, sigma=10.0, max_rot=MAX_ROT, near=0.08):
    """候选 C1（计划 agent-doc/plan-tier2-denoise-2026-10-04.md 第 2 版第 4 节）：支撑面顶不穿透，只用 Tier 2 的支撑面高度与网格。
    静止段（判据同 v5.4，在观测 p_obs 上算；连续 ≥ 10 帧）里，q 的凸包最低点中位数离某个支撑面顶（v51.surface_tops，相对地面平面，
    只取高于地面的顶）≤ near 时，这一段每帧的最低点不得低于顶面 −eps。修法同 enforce 的 rot_s：逐帧绕原点的最小旋转（≤ max_rot），
    σ 帧高斯平滑，最后只对这些帧沿 n 抬高兜底。不看支撑面的 xy 范围（recovery 拿不到）。enforce 本身不动。"""
    from t2core import segments
    from v51 import static_mask
    tops = [H for H in tops if H > 0.0]
    if not tops:
        return q.copy()
    h_all = min_height(q, V, n, d)
    floor = np.full(len(q), -np.inf)
    for s, e in segments(static_mask(p_obs, P), 10):
        hm = float(np.median(h_all[s:e]))
        H = min(tops, key=lambda t: abs(hm - t))
        if abs(hm - H) <= near:
            floor[s:e] = H - eps
    if not np.isfinite(floor).any():
        return q.copy()
    Rm = _R(q).copy(); x = q[:, :3].copy()
    omega = np.zeros((len(q), 3))
    for t in np.flatnonzero(np.isfinite(floor)):
        r = Rm[t] @ V.T
        base = x[t] @ n + d
        h = r.T @ n + base
        if h.min() >= floor[t]:
            continue
        rl = r[:, h.argmin()]
        ax = np.cross(rl, n); s = np.linalg.norm(ax)
        if s < 1e-9:
            continue
        ax /= s
        lim = floor[t]
        ok = lambda th: ((Rot.from_rotvec(ax * th).as_matrix() @ r).T @ n + base).min() >= lim
        lo, hi = 0.0, max_rot
        if ok(hi):
            for _ in range(30):
                mid = (lo + hi) / 2
                lo, hi = (lo, mid) if ok(mid) else (mid, hi)
            omega[t] = ax * hi
        else:
            ths = np.linspace(0, max_rot, 26)
            mins = [((Rot.from_rotvec(ax * th).as_matrix() @ r).T @ n).min() for th in ths]
            omega[t] = ax * ths[int(np.argmax(mins))]
    omega = gaussian_filter1d(omega, sigma, axis=0, mode="nearest")
    Rm = np.einsum("tij,tjk->tik", Rot.from_rotvec(omega).as_matrix(), Rm)
    h = (np.einsum("tij,vj->tvi", Rm, V) + x[:, None]) @ n + d
    need = np.where(np.isfinite(floor), np.maximum(0.0, floor - h.min(1)), 0.0)
    return _pack(Rm, x + need[:, None] * n)
