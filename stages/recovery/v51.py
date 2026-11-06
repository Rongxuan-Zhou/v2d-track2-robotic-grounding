"""Tier 2 物体轨迹降噪器 v5.1（正式实现）。只用 Tier 2 可得的输入。

来历：v1 静止段积分（0.593）→ v5 支撑面 3 自由度 + 布朗桥（0.658）→ v5.1 支持翻面与换高度（0.693）。
推导与消融见 support_bridge.py / support_bridge_v2.py 与计划 N10、N11。

腐蚀约定：obs_R = D_R·true_R（世界系左乘），obs_x = true_x + D_x，D 为从 0 开始的截断随机游走。
1. 静止段（窗口判据）内观测增量就是漂移增量，积分得 D；运动段不可观测，先保持。
2. 静止段若静置在支撑面上，可得 3 个绝对量：沿地面法向的高度、D 把法向 n 映到哪（倾斜）。
   假设按层级挑：先试"与第 0 帧同面同高"（v5，第 0 帧 d≈0）；过不了门限才在
   {凸包稳定放置姿态} × {地面, support 顶面} 里找，与第二名拉不开差距就跳过。
3. 运动段在两侧静止段的 D 之间线性插值（布朗桥：两端已知的随机游走，中间最优估计）。
偏航与水平位置支撑面约束不到，保留积分值。

输入：p (T,7) 物体位姿（SOMA 世界系），地面 (n, d)，support 顶面列表，稳定姿态列表。
"""
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import trimesh

import add_metrics as A
from t2core import window_rot, window_tr, segments, integrate, apply, correct   # 纯函数，不读数据

M = lambda q: A.quat_wxyz_to_matrix(q)
DEFAULT = dict(ref_tilt=35.0, ref_h=0.05, fb_tilt=33.0, fb_h=0.035, margin=0.5,
               eq_ang=5.0, eq_h=0.01, tilt_cap=30.0, h_cap=0.03,
               st_win=91, st_rot=23.5, st_tr=0.06)
# v5.2 = v5.1 + 放宽的静止判据。只在 3 份全新噪声上选参（st_tune.py），真实 Tier 2 只评估一次：
# 0.693 → 0.734。st_rot 在 26–36° 整段都优于 23.5°，不是孤立尖峰。
V52 = {**DEFAULT, "st_rot": 33.0, "st_win": 61}


def surface_tops(usda: Path) -> list[float]:
    """support usda（只有 Cylinder / Cube，无旋转）→ 各几何体顶面相对地面平面的高度。"""
    tops = []
    for blk in re.split(r"\bdef (?=Cylinder|Cube)", Path(usda).read_text())[1:]:
        z = float(re.search(r"xformOp:translate = \(([^)]*)\)", blk).group(1).split(",")[2])
        if blk.startswith("Cylinder"):
            tops.append(z + float(re.search(r"double height = ([\d.eE+-]+)", blk).group(1)) / 2)
        else:
            size = float(re.search(r"double size = ([\d.eE+-]+)", blk).group(1))
            sc = re.search(r"xformOp:scale = \(([^)]*)\)", blk)
            tops.append(z + size * (float(sc.group(1).split(",")[2]) if sc else 1.0) / 2)
    return tops


@lru_cache(maxsize=None)
def stable_ups(mesh_path: str) -> tuple:
    """凸包稳定放置姿态 → ((物体系朝上单位向量 b, 原点到最低点距离 δ), ...)。"""
    m = trimesh.load(mesh_path, force="scene")
    m = m.dump(concatenate=True) if isinstance(m, trimesh.Scene) else m
    hull = m.convex_hull
    T, _ = hull.compute_stable_poses()
    out = []
    for Tk in T:
        b = Tk[:3, :3].T @ np.array([0, 0, 1.0])
        out.append((b, -float((hull.vertices @ b).min())))
    return tuple(out)


CHI3_P99 = 3.3682   # 三维各向同性高斯模长的 0.99 分位 / σ（sqrt(chi2_3 的 0.99 分位 11.345)）


def static_mask(p, P, cover=False, guard=None):
    """静止判据：以当前帧为中心的 st_win 帧窗口里两两最大转角 < st_rot 且两两最大位移 < st_tr。
    cover=True（候选 A1，计划 agent-doc/plan-tier2-denoise-2026-10-04.md 第 2 版第 4 节）：只要有一个通过判据的窗口包含这一帧
    就算静止，即把通过的窗口中心向两边各扩 st_win // 2 帧。
    guard=(σ_rot 弧度, σ_tr 米)（候选 A1′，同节；E5 不过时改试）：扩出去的帧 k 逐帧检查，从通过的窗口中心 i 起的累计观测转角与位移
    都不超过 |k − i| 步白噪声随机游走的 P99（CHI3_P99·σ·√|k − i|）才算静止。"""
    st = (window_rot(p, P["st_win"]) < P["st_rot"]) & (window_tr(p, P["st_win"]) < P["st_tr"])
    if cover:
        h = P["st_win"] // 2
        idx = np.flatnonzero(st)
        out = np.zeros_like(st)
        if guard is not None:
            Rall = M(p[:, 3:7])
        for i in idx:
            lo, hi = max(0, i - h), min(len(st), i + h + 1)
            if guard is None:
                out[lo:hi] = True
                continue
            L = np.sqrt(np.abs(np.arange(lo, hi) - i))
            ang = np.arccos(np.clip((np.einsum("tii->t", np.einsum("tij,jk->tik", Rall[lo:hi], Rall[i].T)) - 1) / 2, -1, 1))
            dx = np.linalg.norm(p[lo:hi, :3] - p[i, :3], axis=1)
            out[lo:hi] |= (ang <= CHI3_P99 * guard[0] * L) & (dx <= CHI3_P99 * guard[1] * L)
        st = out
    return st


def denoise_v51(p, n, d, tops, ups, P=DEFAULT, mode="tier", bridge=True, truth=None,
                cover=False, carry=False, holdout=None, drop=None, cover_guard=None):
    """mode: "tier" = v5.1；"ref" = v5；"flat" = 所有假设平等竞争；"oracle" = 静止段用 truth 的 3 自由度。
    以下开关默认关闭，关闭时输出与原来逐位相同（stages/recovery/t2eval.py 对每条 episode 与 runs/recovery/tier2_v54 逐位核对）：
      cover   候选 A1：静止判据按窗口覆盖（static_mask）；cover_guard=(σ_rot, σ_tr) 时是 A1′（扩出去的帧过白噪声守门）
      carry   候选 B1：锚点返回 None 的段沿用上一个锚定段的修正（t2core.correct）
      holdout 评价 E2a：(s, e) 帧区间；与它重叠的静止段锚点返回 None，其余照旧
      drop    评价 E2b：(s, e) 帧区间；与它重叠的静止段整个不参与锚定，桥跨过它"""
    P = {**DEFAULT, **P}
    st = static_mask(p, P, cover, cover_guard)
    segs = segments(st, 10)
    overlap = lambda s, e, r: r is not None and s < r[1] and r[0] < e
    if drop is not None:
        segs = [(s, e) for s, e in segs if not overlap(s, e, drop)]
    DR, Dx = integrate(p, st)
    h = lambda x: x @ n + d
    ref = (M(p[:1, 3:7])[0].T @ n, float(h(p[0, :3])))
    poses = [(b, H + dl) for b, dl in ups for H in [0.0] + list(tops)]
    rt, fbt, cap = np.radians(P["ref_tilt"]), np.radians(P["fb_tilt"]), np.radians(P["tilt_cap"])
    eqa = np.radians(P["eq_ang"])
    same = lambda a, b: np.arccos(np.clip(a[0] @ b[0], -1, 1)) < eqa and abs(a[1] - b[1]) < P["eq_h"]

    def target(s, e):
        m = (s + e) // 2; Ro = M(p[m:m+1, 3:7])[0]; ho = h(p[m, :3])
        if mode == "oracle":
            Rt = M(truth[m:m+1, 3:7])[0]
            return ((Ro @ Rt.T) @ n, ho - h(truth[m, :3]), None)
        if mode in ("tier", "ref"):
            z = Ro @ ref[0]; dh = ho - ref[1]
            if np.arccos(np.clip(z @ n, -1, 1)) < rt and abs(dh) < P["ref_h"]:
                return (z, dh, None)
            if mode == "ref":
                return (None, None, None)
        hyps = poses if mode == "tier" else [ref] + poses
        cand = []
        for hy in hyps:
            z = Ro @ hy[0]; th = np.arccos(np.clip(z @ n, -1, 1)); dh = ho - hy[1]
            if th < fbt and abs(dh) < P["fb_h"]:
                cand.append((th / cap + abs(dh) / P["h_cap"], hy, z, dh))
        if not cand:
            return (None, None, None)
        cand.sort(key=lambda c: c[0]); best = cand[0]
        rival = next((c for c in cand[1:] if not same(c[1], best[1])), None)
        if rival is not None and rival[0] - best[0] < P["margin"]:
            return (None, None, None)
        return (best[2], best[3], None)

    if holdout is not None:
        _target = target
        target = lambda s, e: (None, None, None) if overlap(s, e, holdout) else _target(s, e)
    return apply(p, *correct(DR, Dx, segs, target, n, bridge, carry=carry))
