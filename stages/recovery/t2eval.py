"""Tier 2 参考降噪的评价，只用 Tier 2 的数据（计划 agent-doc/plan-tier2-denoise-2026-10-04.md 第 2 版第 3–5 节）。

E1 伪标签：静止判据在伪静止帧上的召回；核心运动帧（离最近的伪静止帧 ≥ 30 帧的伪运动帧）被判成静止的比例；按距离分档的误判。
E2a 去锚留段（主）、E2b 去段（只报）：不含第 0 帧的静置段，逐段让与它重叠的静止段锚点返回 None，或把它们整个去掉，
    量这一段中点修正后的倾斜残差（对网格稳定面取最小）与高度残差，与 v5.4 在同一设置下逐段配对。
E3 合成自洽：伪真值 = v5.4 输出，加净室模型的漂移（每轴每帧 0.649° / 0.846 mm 的白增量、d(0) = I；主集不截断 10 份，
    敏感性集截断在 33.9° / 24.2 mm、5 份），降噪后对伪真值的 ADD-AUC（发布网格按种子 0 采 500 个顶点）。
E4 物理：地面约束之前凸包最低点低于地面 > 1 cm 的帧与最深处；静置段里低于支撑面顶 > 1 cm 的帧（只用高度）；
    物体在动（±5 帧中心位移 > 2 cm）而全身 77 关节都离凸包 > 0.25 m 的帧；运动帧角加速度中位数与白噪声底（只报）。
E5 新增静止的守门：候选判为静止、v5.4 没判的每段连续帧（连上与已有静止帧相接的增量），累计转角与累计位移对
    同长度白噪声随机游走的 P99（三维各向同性高斯的模长 = 3.368·σ·√L）；纯噪声下两项合计约 2% 的段超出。
伪标签、静置段的定义与 cleanroom_params.py 相同（全身离凸包 > 0.25 m 且 ≥ 30 帧；物体中心 ±30 帧位移 > 0.15 m；
静置段最低点离某个面 ≤ 8 cm、高度残差 < 0.1 m）。E4 的“人远”沿用观测位姿算的距离（各方法位姿只差几厘米，对 0.25 m 的判断可忽略）。
配置名：v54（基线）、a1（窗口覆盖）、a1p（A1′：窗口覆盖加白噪声守门，A1 的 E5 不过时替 A1）、b1（沿用修正）、c1（支撑面顶不穿透），
用 + 组合，例如 a1+b1。只报的 E4 附加项“静止段穿顶面”（v5.4 判据的静止段里，最低点中位数离某个顶面 ≤ 8 cm 时低于顶面 > 1 cm 的帧）
是 10/5 小跑 3 条之后加的：登记的“静置段穿顶面”只看人远离物体的段，量不到 ep20 那种陷进桌面的帧。
每个配置都是：v51 降噪 → [C1] → 地面不穿透（与 v5.4 的导出相同）。v54 配置对每条 episode 的输出与 runs/recovery/tier2_v54 逐位比较。
用法（仓库根，服务器宿主）：
  python3 stages/recovery/t2eval.py --configs v54 a1 b1 c1 a1+b1 … [--workers 12] [--e3 10] [--e3-trunc 5] [--episodes 0 1 …]
         [--json results/t2denoise/<名字>.json] [--verdict]
进程开头调 forbid_tier1()，只读 Tier 2 档位目录、runs/recovery/tier2_v54 与 FK 缓存；最后一行 T2EVAL_DONE。测试：stages/recovery/test_t2eval.py。
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):   # 多进程并行，每个进程单线程；要在 import numpy 之前
    os.environ.setdefault(_v, "1")
import argparse
import json
import statistics
import sys
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "libs" / "v2d_infra")); sys.path.insert(0, str(HERE))
from tier_guard import forbid_tier1
forbid_tier1()
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as Rot

import add_metrics as A
from paths import RUNS, TIER2
from soma_fk import col, fk, R_CV
from t2core import to_soma, segments
from v51 import denoise_v51, static_mask, surface_tops, stable_ups
from nopen import enforce, enforce_tops, hull_vertices, min_height

SIG_R, SIG_X = np.radians(0.649), 0.000846          # 净室漂移步长（每轴每帧），results/cleanroom_params.txt
CAP_R, CAP_X = np.radians(33.916), 0.024             # 净室上限（同上），只用于敏感性集
FAR, MINSEG, PSEUDO_W, PSEUDO_D, NEAR_SURF, HRES_MAX = 0.25, 30, 30, 0.15, 0.08, 0.10   # 同 cleanroom_params.py
CORE_GAP, MOVE_W, MOVE_DX, PEN = 30, 5, 0.02, 0.01
CHI3_P99 = 3.3682                                    # sqrt(chi2_3 的 0.99 分位 11.345)
CR = json.loads((RUNS / "recovery" / "cleanroom_params.json").read_text())["params"]
PARAMS = {k: CR[k] for k in ("ref_tilt", "ref_h", "fb_tilt", "fb_h", "tilt_cap", "h_cap", "st_win", "st_rot", "st_tr",
                             "margin", "eq_ang", "eq_h")}
NOPEN_MAX_ROT = np.radians(CR["nopen_max_rot_deg"])
META = {json.loads(l)["episode_index"]: json.loads(l) for l in open(TIER2 / "meta" / "episodes_metadata.jsonl")}
CANDS = ("a1", "a1p", "b1", "c1")


def fill_invisible(p, vis):
    out = p.copy(); first = int(np.argmax(vis)); out[:first] = p[first]; last = first
    for i in range(first, len(p)):
        if vis[i]: last = i
        else: out[i] = p[last]
    return out


def pack(Rm, x):
    q = Rot.from_matrix(Rm).as_quat()
    out = np.empty((len(x), 7)); out[:, :3] = x; out[:, 3:7] = q[:, [3, 0, 1, 2]]
    return out


def parse_cfg(name):
    parts = set(name.split("+")) - {"v54"}
    assert parts <= set(CANDS) and not {"a1", "a1p"} <= parts, name
    return parts


def run_config(parts, p, ep, holdout=None, drop=None):
    """返回 (终输出, 地面约束之前的输出)，都在 SOMA 世界系。"""
    q = denoise_v51(p, ep["n"], ep["d"], ep["tops"], ep["ups"], PARAMS, cover=bool(parts & {"a1", "a1p"}), carry="b1" in parts,
                    holdout=holdout, drop=drop, cover_guard=(SIG_R, SIG_X) if "a1p" in parts else None)
    if "c1" in parts:
        q = enforce_tops(q, p, ep["V"], ep["n"], ep["d"], ep["tops"], PARAMS, max_rot=NOPEN_MAX_ROT)
    return enforce(q, ep["V"], ep["n"], ep["d"], mode="rot_s", max_rot=NOPEN_MAX_ROT), q


def seg_resid(q, s, e, top, ep):
    mid = (s + e) // 2
    R = A.quat_wxyz_to_matrix(q[mid, 3:7])
    tilts = [float(np.degrees(np.arccos(np.clip((R @ b) @ ep["n"], -1, 1)))) for b, _ in ep["ups"]]
    k = int(np.argmin(tilts))
    return tilts[k], 1000.0 * float(q[mid, :3] @ ep["n"] + ep["d"] - (top + ep["ups"][k][1]))


def load(ei):
    m = META[ei]
    t = pq.read_table(TIER2 / "data" / "chunk-000" / f"episode_{ei:06d}.parquet")
    vis = np.asarray(col(t, "observation.object.visible")).astype(bool).ravel()
    p = to_soma(fill_invisible(col(t, "observation.object.pose").astype(np.float64), vis))
    g = json.load(open(TIER2 / m["ground_plane"]))["plane"]; s_ = np.linalg.norm(g[:3])
    n = R_CV @ (np.array(g[:3]) / s_); d = g[3] / s_
    if np.median(p[:, :3] @ n + d) < 0:
        n, d = -n, -d
    mesh = str(TIER2 / m["mesh"])
    ep = dict(ei=ei, obj=m["object"], p=p, n=n, d=d, tops=surface_tops(TIER2 / m["support_surface"]), ups=stable_ups(mesh),
              V=hull_vertices(mesh), verts=A.sample_mesh_vertices(mesh, 500, 0))
    Rm = Rot.from_quat(p[:, [4, 5, 6, 3]]).as_matrix()
    W = np.einsum("tij,vj->tvi", Rm, ep["V"]) + p[:, None, :3]
    J = fk(t, flex=True)
    dist = np.concatenate([np.linalg.norm(J[i:i + 25, :, None, :] - W[i:i + 25, None, :, :], axis=-1).min(axis=(1, 2))
                           for i in range(0, len(p), 25)])                     # 分块：顶点多的物体整张量会占几 GB
    T = len(p)
    st_lab = np.zeros(T, bool)
    for s, e in segments(dist > FAR, MINSEG):
        st_lab[s:e] = True
    x = p[:, :3]
    disp = np.array([np.linalg.norm(x[max(0, i - PSEUDO_W):min(T, i + PSEUDO_W + 1)] - x[i], axis=1).max() for i in range(T)])
    mv = (disp > PSEUDO_D) & ~st_lab
    idx = np.flatnonzero(st_lab)
    gap = np.full(T, 10**9) if len(idx) == 0 else np.abs(np.arange(T)[:, None] - idx[None, :]).min(1)
    tops_all = [0.0] + list(ep["tops"])
    rest = []
    for s, e in segments(st_lab, MINSEG):
        low = (W[s:e] @ n + d).min(1)
        top = min(tops_all, key=lambda H: abs(np.median(low) - H))
        if abs(np.median(low) - top) > NEAR_SURF:
            continue
        if abs(seg_resid(p, s, e, top, ep)[1]) / 1000.0 >= HRES_MAX:
            continue
        rest.append(dict(s=int(s), e=int(e), has0=bool(s == 0), top=float(top)))
    ep.update(st_lab=st_lab, mv=mv, gap=gap, far=dist > FAR, rest=rest)
    return ep


def synth(T, ei, k, trunc):
    rng = np.random.default_rng([ei, k, int(trunc)])
    DR = np.empty((T, 3, 3)); Dx = np.zeros((T, 3)); DR[0] = np.eye(3)
    for t in range(1, T):
        R = Rot.from_rotvec(rng.normal(0, SIG_R, 3)).as_matrix() @ DR[t - 1]
        xx = Dx[t - 1] + rng.normal(0, SIG_X, 3)
        if trunc:
            rv = Rot.from_matrix(R).as_rotvec(); a = np.linalg.norm(rv)
            if a > CAP_R:
                R = Rot.from_rotvec(rv * CAP_R / a).as_matrix()
            nx = np.linalg.norm(xx)
            if nx > CAP_X:
                xx = xx * CAP_X / nx
        DR[t] = R; Dx[t] = xx
    return DR, Dx


def e1(ep, parts):
    st = static_mask(ep["p"], PARAMS, cover=bool(parts & {"a1", "a1p"}), guard=(SIG_R, SIG_X) if "a1p" in parts else None)
    out = dict(st_lab=int(ep["st_lab"].sum()), st_hit=int((st & ep["st_lab"]).sum()))
    for name, lo, hi in (("fp_0_10", 0, 10), ("fp_10_30", 10, 30), ("fp_core", CORE_GAP, 10**9 + 1)):
        mask = ep["mv"] & (ep["gap"] >= lo) & (ep["gap"] < hi)
        out[name] = [int((st & mask).sum()), int(mask.sum())]
    return out, st


def e4(ep, q, pre):
    n, d, V = ep["n"], ep["d"], ep["V"]
    hp = min_height(pre, V, n, d)
    hq = min_height(q, V, n, d)
    top_pen = 0
    for r in ep["rest"]:
        if r["top"] > 0:
            top_pen += int((hq[r["s"]:r["e"]] < r["top"] - PEN).sum())
    x = q[:, :3]; T = len(x)
    lo, hi = np.clip(np.arange(T) - MOVE_W, 0, T - 1), np.clip(np.arange(T) + MOVE_W, 0, T - 1)
    moving = np.linalg.norm(x[hi] - x[lo], axis=1) > MOVE_DX
    R = A.quat_wxyz_to_matrix(q[:, 3:7])
    w = Rot.from_matrix(np.einsum("tij,tkj->tik", R[1:], R[:-1])).as_rotvec()
    dw = np.degrees(np.linalg.norm(np.diff(w, axis=0), axis=1))
    mvf = ep["mv"][2:]
    top_pen_static = 0                                   # 只报（10/5 小跑后加）：v5.4 判据的静止段里贴着顶面的帧
    tops = [H for H in ep["tops"] if H > 0]
    for s, e in segments(static_mask(ep["p"], PARAMS), 10):
        hm = float(np.median(hq[s:e]))
        if tops:
            H = min(tops, key=lambda t: abs(hm - t))
            if abs(hm - H) <= NEAR_SURF:
                top_pen_static += int((hq[s:e] < H - PEN).sum())
    return dict(ground_pre=int((hp < -PEN).sum()), ground_pre_depth_mm=float(max(0.0, -1000 * hp.min())),
                top_pen=int(top_pen), top_pen_static=int(top_pen_static), second_actor=int((moving & ep["far"]).sum()),
                angacc_med=float(np.median(dw[mvf])) if mvf.any() else None)


def e5(ep, st54, stx):
    p = ep["p"]; T = len(p)
    R = A.quat_wxyz_to_matrix(p[:, 3:7])
    rows = []
    for s, e in segments(stx & ~st54, 1):
        a = s - 1 if s > 0 and st54[s - 1] else s
        b = e if e < T and st54[e] else e - 1
        L = b - a
        if L <= 0:
            continue
        rot = float(np.linalg.norm(Rot.from_matrix(R[b] @ R[a].T).as_rotvec()))
        tr = float(np.linalg.norm(p[b, :3] - p[a, :3]))
        rows.append(dict(s=int(s), e=int(e), L=int(L), rot_deg=float(np.degrees(rot)), tr_mm=1000 * tr,
                         flag=bool(rot > CHI3_P99 * SIG_R * np.sqrt(L) or tr > CHI3_P99 * SIG_X * np.sqrt(L))))
    return rows


def episode_worker(job):
    ei, cfgs, n_main, n_trunc = job
    ep = load(ei)
    T = len(ep["p"])
    out = {"ei": ei, "obj": ep["obj"], "T": T, "rest": ep["rest"], "cfg": {}}
    q54, _ = run_config(set(), ep["p"], ep)
    stored = np.load(RUNS / "recovery" / "tier2_v54" / f"episode_{ei:06d}.npz")["object_pose"]
    q54_cv = to_soma(q54)
    out["v54_reproduced"] = bool(np.array_equal(q54_cv, stored))
    out["v54_maxdiff"] = float(np.abs(q54_cv - stored).max())
    st54 = static_mask(ep["p"], PARAMS)
    gts = []
    for k in range(n_main + n_trunc):
        trunc = k >= n_main
        DR, Dx = synth(T, ei, k, trunc)
        Rg = A.quat_wxyz_to_matrix(q54[:, 3:7])
        gts.append((trunc, pack(np.einsum("tij,tjk->tik", DR, Rg), q54[:, :3] + Dx)))
    for name in cfgs:
        parts = parse_cfg(name)
        q, pre = run_config(parts, ep["p"], ep)
        r1, st = e1(ep, parts)
        res = {"e1": r1, "e4": e4(ep, q, pre), "e5": e5(ep, st54, st) if parts & {"a1", "a1p"} else [], "e2": [], "e3_main": [], "e3_trunc": []}
        for r in ep["rest"]:
            if r["has0"]:
                continue
            qa, _ = run_config(parts, ep["p"], ep, holdout=(r["s"], r["e"]))
            qb, _ = run_config(parts, ep["p"], ep, drop=(r["s"], r["e"]))
            ta, ha = seg_resid(qa, r["s"], r["e"], r["top"], ep)
            tb, hb = seg_resid(qb, r["s"], r["e"], r["top"], ep)
            res["e2"].append(dict(s=r["s"], e=r["e"], tilt_a=ta, h_a=ha, tilt_b=tb, h_b=hb))
        for trunc, obs in gts:
            est, _ = run_config(parts, obs, ep)
            auc = A.auc_from_add(A.add_per_frame(ep["verts"], q54, est))[0]
            res["e3_trunc" if trunc else "e3_main"].append(float(auc))
        out["cfg"][name] = res
    return out


def noise_floor_angacc(n=200000, seed=7):
    rng = np.random.default_rng(seed)
    w = rng.normal(0, np.degrees(SIG_R), (n + 1, 3))
    return float(np.median(np.linalg.norm(np.diff(w, axis=0), axis=1)))


def summarize(eps, cfgs):
    S = {}
    for name in cfgs:
        e1s = [e["cfg"][name]["e1"] for e in eps]
        st_lab, st_hit = sum(x["st_lab"] for x in e1s), sum(x["st_hit"] for x in e1s)
        fp = {k: [sum(x[k][0] for x in e1s), sum(x[k][1] for x in e1s)] for k in ("fp_0_10", "fp_10_30", "fp_core")}
        e4s = [e["cfg"][name]["e4"] for e in eps]
        e2 = [dict(ei=e["ei"], **r) for e in eps for r in e["cfg"][name]["e2"]]
        e5 = [dict(ei=e["ei"], **r) for e in eps for r in e["cfg"][name]["e5"]]
        S[name] = dict(
            recall=st_hit / max(st_lab, 1), fp={k: v[0] / max(v[1], 1) for k, v in fp.items()}, fp_counts=fp,
            ground_pre=sum(x["ground_pre"] for x in e4s), ground_pre_depth_mm=max(x["ground_pre_depth_mm"] for x in e4s),
            top_pen=sum(x["top_pen"] for x in e4s), top_pen_static=sum(x["top_pen_static"] for x in e4s),
            second_actor=sum(x["second_actor"] for x in e4s),
            angacc_med=(statistics.median(v) if (v := [x["angacc_med"] for x in e4s if x["angacc_med"] is not None]) else float("nan")),
            e2=e2, e5=e5, e5_flag_rate=(sum(r["flag"] for r in e5) / len(e5)) if e5 else 0.0,
            e3_main={e["ei"]: statistics.mean(e["cfg"][name]["e3_main"]) for e in eps if e["cfg"][name]["e3_main"]},
            e3_trunc={e["ei"]: statistics.mean(e["cfg"][name]["e3_trunc"]) for e in eps if e["cfg"][name]["e3_trunc"]})
        S[name]["e3_main_mean"] = statistics.mean(S[name]["e3_main"].values()) if S[name]["e3_main"] else None
        S[name]["e3_trunc_mean"] = statistics.mean(S[name]["e3_trunc"].values()) if S[name]["e3_trunc"] else None
    return S


def paired(S, name, key):
    base = {(r["ei"], r["s"]): r[key] for r in S["v54"]["e2"]}
    return [r[key] - base[(r["ei"], r["s"])] for r in S[name]["e2"]]


def checks(S, name):
    """第 5 节：共同判据 G1、G3、G4，与各候选自己的判据。返回 {判据: (过否, 说明)}。"""
    b, x = S["v54"], S[name]
    parts = parse_cfg(name)
    out = {}
    out["G1"] = (x["fp"]["fp_core"] <= b["fp"]["fp_core"] + 0.005,
                 f"核心运动帧误判 {x['fp']['fp_core']:.4f}（v5.4 {b['fp']['fp_core']:.4f}，要 ≤ +0.005）")
    worst = min(x["e3_main"][k] - b["e3_main"][k] for k in b["e3_main"])
    out["G3"] = (x["e3_main_mean"] >= b["e3_main_mean"] - 0.003 and worst >= -0.02,
                 f"E3 主集平均 {x['e3_main_mean']:.4f}（v5.4 {b['e3_main_mean']:.4f}，要 ≥ −0.003）；逐条最差 {worst:+.4f}（要 ≥ −0.02）")
    out["G4"] = (x["ground_pre"] <= 1.1 * b["ground_pre"] and x["second_actor"] <= 1.1 * b["second_actor"],
                 f"约束前穿地帧 {x['ground_pre']}（v5.4 {b['ground_pre']}）；第二施动者帧 {x['second_actor']}（v5.4 {b['second_actor']}）；要 ≤ 1.1 倍")
    dt = paired(S, name, "tilt_a")
    med = statistics.median(dt) if dt else 0.0
    frac = sum(v < 0 for v in dt) / len(dt) if dt else 0.0
    if parts & {"a1", "a1p", "b1"}:
        out["G2"] = (frac >= 0.6 and med < 0, f"E2a 倾斜残差变低的段 {frac:.0%}（要 ≥ 60%），配对差中位 {med:+.2f}°（要 < 0），共 {len(dt)} 段")
    if parts & {"a1", "a1p"}:
        out["E5"] = (x["e5_flag_rate"] <= 0.03, f"新增静止段超出白噪声 P99 的比例 {x['e5_flag_rate']:.1%}（{sum(r['flag'] for r in x['e5'])} / {len(x['e5'])}，要 ≤ 3%）")
    if "c1" in parts:
        out["C1"] = (x["top_pen"] <= 0.5 * b["top_pen"] and med <= 0.5,
                     f"静置段穿支撑面顶帧 {x['top_pen']}（v5.4 {b['top_pen']}，要 ≤ 一半）；E2a 配对差中位 {med:+.2f}°（要 ≤ +0.5°）")
    return out


def verdict(S):
    lines = []
    a = "a1"
    if "a1" in S and not checks(S, "a1")["E5"][0]:
        a = "a1p"
        lines.append("A1 的 E5 不过，按第 4 节改用 A1′（a1p）替它" + ("" if "a1p" in S else "，但 a1p 没有评"))
    singles = [c for c in (a, "b1", "c1") if c in S]
    passed = [c for c in singles if all(ok for ok, _ in checks(S, c).values())]
    lines.append(f"单独通过的候选：{passed if passed else '无'}")
    if not passed:
        return "v54", lines
    combo = "+".join(passed)
    order = [combo] + ["+".join(c) for r in range(len(passed) - 1, 0, -1) for c in combinations(passed, r)]
    for name in order:
        if name not in S:
            lines.append(f"  {name} 没有评，跳过")
            continue
        ok = all(v for v, _ in checks(S, name).values())
        lines.append(f"  {name}：{'过' if ok else '不过'}，E3 主集平均 {S[name]['e3_main_mean']:.4f}")
    full = [n for n in order if n in S and all(v for v, _ in checks(S, n).values())]
    if combo in full:
        return combo, lines
    drop1 = [n for n in full if n.count("+") == combo.count("+") - 1]
    if drop1:
        return max(drop1, key=lambda n: S[n]["e3_main_mean"]), lines
    return max(passed, key=lambda n: S[n]["e3_main_mean"]), lines


def report(S, cfgs, eps):
    print(f"v5.4 复现（开关全关重算 vs runs/recovery/tier2_v54）：{sum(e['v54_reproduced'] for e in eps)} / {len(eps)} 逐位相同；"
          f"最大差 {max(e['v54_maxdiff'] for e in eps):.3g}")
    print(f"白噪声底：运动帧角加速度中位 {noise_floor_angacc():.3f}°/帧²（净室步长 0.649°/轴/帧）")
    print(f"\n{'配置':<12}{'召回':>7}{'误判0-10':>9}{'误判10-30':>10}{'误判核心':>9}{'约束前穿地':>10}{'最深mm':>8}{'穿顶面':>7}{'静止穿顶(报)':>12}{'第二施动者':>10}"
          f"{'角加速度':>9}{'E3主集':>9}{'E3截断':>9}{'E5超出':>8}")
    for name in cfgs:
        x = S[name]
        print(f"{name:<12}{x['recall']:>7.3f}{x['fp']['fp_0_10']:>9.3f}{x['fp']['fp_10_30']:>10.3f}{x['fp']['fp_core']:>9.4f}"
              f"{x['ground_pre']:>10d}{x['ground_pre_depth_mm']:>8.1f}{x['top_pen']:>7d}{x['top_pen_static']:>12d}{x['second_actor']:>10d}{x['angacc_med']:>9.3f}"
              f"{x['e3_main_mean'] if x['e3_main_mean'] is not None else float('nan'):>9.4f}"
              f"{x['e3_trunc_mean'] if x['e3_trunc_mean'] is not None else float('nan'):>9.4f}"
              f"{x['e5_flag_rate']:>8.1%}")
    print("\nE2（不含第 0 帧的静置段；倾斜 °、高度 mm）：")
    hdr = "".join(f"{n + ' a':>18}{n + ' b':>18}" for n in cfgs)
    print(f"  {'ep':<4}{'段':<11}{hdr}")
    for i, r in enumerate(S["v54"]["e2"] if "v54" in S else []):
        cells = "".join(f"{S[n]['e2'][i]['tilt_a']:>9.2f}/{S[n]['e2'][i]['h_a']:>7.1f} {S[n]['e2'][i]['tilt_b']:>9.2f}/{S[n]['e2'][i]['h_b']:>7.1f}"
                        for n in cfgs)
        print(f"  {r['ei']:<4}{r['s']:>4}–{r['e']:<6}{cells}")
    if "v54" in S:
        for name in cfgs:
            if name == "v54":
                continue
            print(f"\n== {name} 对 v5.4")
            for k, (ok, why) in checks(S, name).items():
                print(f"  {k} {'过' if ok else '不过'}：{why}")
            ex = [r for r in S[name]["e5"] if r["flag"]]
            if ex:
                print("  E5 超出的段：" + "，".join(f"ep{r['ei']} {r['s']}–{r['e']}（{r['rot_deg']:.1f}°/{r['tr_mm']:.0f} mm，L={r['L']}）" for r in ex))
            print("  E3 逐条（主集，候选 − v5.4）：" + "，".join(f"ep{k} {S[name]['e3_main'][k] - S['v54']['e3_main'][k]:+.3f}"
                                                         for k in sorted(S["v54"]["e3_main"])))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--e3", type=int, default=10)
    ap.add_argument("--e3-trunc", type=int, default=5)
    ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--json")
    ap.add_argument("--verdict", action="store_true")
    a = ap.parse_args(argv)
    cfgs = list(dict.fromkeys(["v54"] + a.configs))
    for c in cfgs:
        parse_cfg(c)
    eis = a.episodes if a.episodes else sorted(META)
    print(f"净室：forbid_tier1() 已调；配置 {cfgs}；episode {len(eis)} 条；E3 主集 {a.e3} 份、截断 {a.e3_trunc} 份；进程 {a.workers}")
    jobs = [(ei, cfgs, a.e3, a.e3_trunc) for ei in eis]
    with Pool(a.workers, initializer=forbid_tier1) as pool:
        eps = sorted(pool.map(episode_worker, jobs, chunksize=1), key=lambda e: e["ei"])
    S = summarize(eps, cfgs)
    report(S, cfgs, eps)
    if a.verdict:
        pick, lines = verdict(S)
        print("\n第 5 节组合规则：")
        for ln in lines:
            print("  " + ln)
        print(f"T2EVAL_PICK {pick}")
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps({"configs": cfgs, "episodes": eps, "summary": S}, ensure_ascii=False, default=float))
    print("T2EVAL_DONE")


if __name__ == "__main__":
    main()
