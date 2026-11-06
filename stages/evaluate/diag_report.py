"""逐帧诊断报告（只用 numpy）：ADD-AUC 在哪一段丢的、朝向误差怎么变、物体有没有被转动、手的情况、失败的世界。

输入 diag_rollout.py 存的 npz 与它的评测 JSON、同一张卡上原样 fc_eval.sh 的评测 JSON，可选另一张卡上的参照评测 JSON（只描述）。
计划与事先登记见 agent-doc/plan-day1b-2026-09-27.md（第 2 版）。任何一道闸不过：只打印数字、不给分类，退出码 3。
  G0 形状：T、W 与评测 JSON 一致，物体刚体 1 个
  G1 同卡：包装后的评测与原样评测的指标逐位相同；不同则要求 chord_sr 的 |z| < 3 且 |ΔADD-AUC| < 0.005 × √(4096 / 世界数)（注明）。
     容差随世界数缩放：4096 世界时就是登记的 0.005；同一冒烟 policy 两次不包装的 256 世界评测就差 0.0057（results/playback/repeat_eval_smoke_w256.txt）
  G2 由 npz 的逐帧 ADD 重算的全程 ADD-AUC 等于评测 JSON（|Δ| < 1e-6）
  G3 按位姿算的物体朝向误差与评测器自己的 object_orientation_rad 最大差 < 0.1°（查 xyzw 顺序与帧对齐）
  G4 已知答案：由参考算出的“增量 > 0.5° 的首末帧”“首次超过 1.5 rad 的帧”等于 --expect-ref（来自 static_baseline_*.txt）
  G5 按判据逐世界重算的成功率等于评测报告
  G6 几何自检（可选 --predict-frame 帧）：任务原生判据下成功的每个世界，该帧物体离自身初始朝向 ≥
     参考在该帧离初始朝向 − 1.5 rad − 该世界第 0 帧的朝向误差 − 0.1°（三角不等式，逐世界）

转动进度：取参考在关注段内逐帧世界系增量 log(R_t R_{t-1}^T) 之和的方向为主轴，把参考与实际的增量都投影到主轴上带符号累加。
零均值抖动相消、不受四元数正负号影响、主轴不必竖直。单帧增量 > 30° 记为跳变。

用法：python diag_report.py <npz> --diag-json D --same-card-json E [--cross-card-json F] --segments 209 420
        --expect-ref 209 420 269 [--predict-frame 359] [--frame-scale 1] [--svg out.svg] [--title 名字]
  --phases F0 F1 …（可选，升序）：Q2b 按这些帧分阶段记新增的落后量（回放速度对照登记的是第 209、269、314、420 帧及其按时间换算）。
  --frame-scale：Q1、Q2 按关键帧列数时另列的 ep20 帧（100、269、300、359、360）按此倍数换算；按真实时长回放（30 → 50 fps）给 5/3。只影响打印。
"""
import argparse
import json
import sys

import numpy as np

AUC_THRESHOLDS = np.arange(0.01, 0.10, 0.01)  # 同评测器 metrics.ADD_THRESHOLDS_M
GATE_DEG = float(np.degrees(1.5))


def auc(samples: np.ndarray) -> float:
    """评测器 add_auc 的同一定义：各阈值下的准确率对 [0,1] 上的等距轴做梯形积分。"""
    s = np.asarray(samples, np.float64).ravel()
    return float(np.trapezoid([np.mean(s < t) for t in AUC_THRESHOLDS], x=np.linspace(0, 1, AUC_THRESHOLDS.size)))


def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def rotvec(R: np.ndarray) -> np.ndarray:
    ang = np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1))
    v = np.stack([R[..., 2, 1] - R[..., 1, 2], R[..., 0, 2] - R[..., 2, 0], R[..., 1, 0] - R[..., 0, 1]], -1)
    s = np.sin(ang)
    return v * np.where(s > 1e-9, ang / (2 * np.where(s > 1e-9, s, 1)), 0.5)[..., None]


def angle_deg(R: np.ndarray) -> np.ndarray:
    return np.degrees(np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))


def world_increments(R: np.ndarray) -> np.ndarray:
    return rotvec(np.einsum("t...ij,t...kj->t...ik", R[1:], R[:-1]))


def classify(net: float, peak: float, net_ref: float) -> str:
    """事先登记的 Q2 分类（名字不暗示“跟上了”）。"""
    if net_ref != 0 and net / net_ref >= 0.5:
        return "净转过半"
    if peak >= 30:
        return "转过但没保持或没跟上"
    if peak < 10:
        return "基本没转"
    return "其他"


def pct(x, qs=(25, 50, 75)):
    return np.percentile(np.asarray(x, np.float64), qs, axis=0)


def svg_chart(title, T, s0, s1, panels) -> str:
    """panels: [(标题, 单位, [(名字, y(T,), 颜色, 虚线?)], 带 (lo,hi) 或 None, 水平线或 None)]。"""
    Wd, left, right, ph, gap, top = 900, 70, 20, 130, 46, 40
    H = top + len(panels) * (ph + gap) + 10

    def x(f):
        return left + (Wd - left - right) * f / max(T - 1, 1)

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {Wd} {H}" font-family="sans-serif" font-size="12">',
           f'<rect x="0" y="0" width="{Wd}" height="{H}" fill="#ffffff"/>',
           f'<text x="{left}" y="22" font-size="14" font-weight="700" fill="#111111">{title}</text>']
    for k, (ptitle, unit, series, band, hline) in enumerate(panels):
        y0 = top + k * (ph + gap) + 18
        vals = [np.asarray(s[1], float) for s in series] + ([np.asarray(band[0]), np.asarray(band[1])] if band else [])
        if hline is not None:
            vals.append(np.array([hline]))
        lo, hi = float(np.nanmin(np.concatenate(vals))), float(np.nanmax(np.concatenate(vals)))
        hi = hi if hi - lo > 1e-9 else lo + 1
        pad = 0.05 * (hi - lo)
        lo, hi = lo - pad, hi + pad

        def y(v, y0=y0, lo=lo, hi=hi):  # 绑定本面板的参数
            return y0 + ph - ph * (v - lo) / (hi - lo)

        out.append(f'<text x="{left}" y="{y0 - 6}" font-weight="700" fill="#111111">{ptitle}</text>')
        out.append(f'<rect x="{x(s0):.1f}" y="{y0}" width="{x(s1) - x(s0):.1f}" height="{ph}" fill="#000000" fill-opacity="0.06"/>')
        out.append(f'<rect x="{left}" y="{y0}" width="{Wd - left - right}" height="{ph}" fill="none" stroke="#bbbbbb"/>')
        for t in np.linspace(lo + pad, hi - pad, 4):
            out.append(f'<text x="{left - 6}" y="{y(t) + 4:.1f}" text-anchor="end" fill="#555555">{t:.0f}{unit}</text>')
        if band is not None:
            pts = [f"{x(f):.1f},{y(v):.1f}" for f, v in enumerate(band[1])]
            pts += [f"{x(f):.1f},{y(v):.1f}" for f, v in reversed(list(enumerate(band[0])))]
            out.append(f'<polygon points="{" ".join(pts)}" fill="#2e7d32" fill-opacity="0.18" stroke="none"/>')
        if hline is not None:
            out.append(f'<line x1="{left}" x2="{Wd - right}" y1="{y(hline):.1f}" y2="{y(hline):.1f}" stroke="#b3261e" stroke-dasharray="5 4"/>')
        for i, (name, ys, color, dashed) in enumerate(series):
            pts = " ".join(f"{x(f):.1f},{y(v):.1f}" for f, v in enumerate(ys))
            dash = ' stroke-dasharray="4 3"' if dashed else ""  # 容器是 Python 3.11：f-string 表达式里不能有反斜杠
            out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.6"{dash}/>')
            out.append(f'<text x="{Wd - right - 6}" y="{y0 + 14 + 14 * i}" text-anchor="end" fill="{color}">{name}</text>')
        if k == len(panels) - 1:
            for f in range(0, T, 100):
                out.append(f'<text x="{x(f):.1f}" y="{y0 + ph + 16}" text-anchor="middle" fill="#555555">{f}</text>')
            out.append(f'<text x="{(x(s0) + x(s1)) / 2:.1f}" y="{y0 + ph + 32}" text-anchor="middle" fill="#555555">帧（灰底为第 {s0}–{s1} 帧）</text>')
    out.append("</svg>")
    return "\n".join(out)


def main() -> int:  # 报告按登记的条目顺序写，拆开反而难对照
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--diag-json", required=True)
    ap.add_argument("--same-card-json", required=True)
    ap.add_argument("--cross-card-json")
    ap.add_argument("--segments", type=int, nargs=2, required=True)
    ap.add_argument("--expect-ref", type=int, nargs=3, required=True, metavar=("FIRST_MOVE", "LAST_MOVE", "FIRST_GATE"))
    ap.add_argument("--predict-frame", type=int, help="G6 几何自检所用的帧（参考离初始朝向接近最大的那帧，ep20 为 359）")
    ap.add_argument("--phases", type=int, nargs="+", help="Q2b 分阶段的边界帧（升序）")
    ap.add_argument("--frame-scale", type=float, default=1.0, help="Q1、Q2 另列关键帧的换算倍数（真实时长回放给 5/3），只影响打印")
    ap.add_argument("--svg")
    ap.add_argument("--title", default="")
    a = ap.parse_args()
    s0, s1 = a.segments
    if a.phases and (len(a.phases) < 2 or any(x >= y for x, y in zip(a.phases, a.phases[1:]))):
        raise SystemExit(f"--phases 要至少两个升序的帧，收到 {a.phases}")

    def load_json(path):
        with open(path) as f:
            return json.load(f)

    d = np.load(a.npz, allow_pickle=False)
    diag, same = load_json(a.diag_json), load_json(a.same_card_json)
    cross = load_json(a.cross_card_json) if a.cross_card_json else None
    ach = d["achieved_pose_w"].astype(np.float64)
    refp = d["reference_pose_w"].astype(np.float64)
    err = d["tracking_error"].astype(np.float64)
    add, add_static = d["add"].astype(np.float64), d["add_static"].astype(np.float64)
    names = [str(n) for n in d["tracking_error_names"]]
    crit = json.loads(str(d["criteria_json"]))
    T, W, B = ach.shape[:3]
    if a.phases and a.phases[-1] >= T:
        raise SystemExit(f"--phases 的末帧 {a.phases[-1]} 超出 rollout 的 {T} 帧")
    col = {n: i for i, n in enumerate(names)}
    gates = []
    print(f"policy {d['policy']}；世界 {W}，帧 {T}；关注段 {s0}–{s1}；上游文件 md5 {d['upstream_md5']}；包装调用 {d['wrapper_calls']}")

    # ── 闸 ───────────────────────────────────────────────────────────
    gates.append(("G0 形状", T == diag["step_count"] and W == diag["world_count"] and B == 1,
                  f"T {T}/{diag['step_count']}，W {W}/{diag['world_count']}，B {B}"))
    m1, m2 = diag["metrics"], same["metrics"]
    numeric = [k for k, v in m1.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    identical = (all(m1[k] == m2.get(k) for k in numeric) and diag["checkpoint_sha256"] == same["checkpoint_sha256"]
                 and diag["world_count"] == same["world_count"])
    if identical:
        gates.append(("G1 同卡：包装前后逐位相同", True, f"{len(numeric)} 项指标全等"))
    else:
        p1, p2, n1, n2 = m1["chord_sr"], m2["chord_sr"], diag["world_count"], same["world_count"]
        p = (p1 * n1 + p2 * n2) / (n1 + n2)
        se = np.sqrt(max(p * (1 - p), 1e-12) * (1 / n1 + 1 / n2))
        z = (p1 - p2) / se
        dadd = abs(m1["add_auc"] - m2["add_auc"])
        tol = 0.005 * np.sqrt(4096 / min(n1, n2))   # ADD-AUC 是世界平均，评测不逐位可复现带来的差按 1/√世界数 缩放
        gates.append((f"G1 同卡：不逐位相同，改用 |z| < 3 与 |ΔADD-AUC| < {tol:.4f}（0.005 × √(4096/世界数)）",
                      abs(z) < 3 and dadd < tol and diag["checkpoint_sha256"] == same["checkpoint_sha256"],
                      f"chord_sr {p1:.4f} vs {p2:.4f}（z {z:.2f}），ADD-AUC 差 {dadd:.5f}"))
    auc_full = auc(add)
    gates.append(("G2 重算全程 ADD-AUC", abs(auc_full - m1["add_auc"]) < 1e-6, f"重算 {auc_full:.6f} vs JSON {m1['add_auc']:.6f}"))
    Ra, Rr = quat_xyzw_to_matrix(ach[:, :, 0, 3:7]), quat_xyzw_to_matrix(refp[:, 0, 3:7])
    ori = angle_deg(np.einsum("twij,tkj->twik", Ra, Rr))                                   # (T,W) °
    env_ori = np.degrees(err[..., col["object_orientation_rad"]])
    dmax = float(np.max(np.abs(env_ori - ori)))
    gates.append(("G3 朝向误差对账", dmax < 0.1, f"最大差 {dmax:.4f}°"))
    ang0_ref = angle_deg(np.einsum("tij,kj->tik", Rr, Rr[0]))
    inc_ref_deg = angle_deg(np.einsum("tij,tkj->tik", Rr[1:], Rr[:-1]))
    moving = np.where(inc_ref_deg > 0.5)[0] + 1
    cross_idx = np.where(ang0_ref > GATE_DEG)[0]
    got = (int(moving.min()) if moving.size else -1, int(moving.max()) if moving.size else -1,
           int(cross_idx[0]) if cross_idx.size else -1)
    gates.append(("G4 参考的已知答案", got == tuple(a.expect_ref), f"算得 {got}，应为 {tuple(a.expect_ref)}"))
    thr, act = crit["thresholds"], crit["active_after_step"]
    step = np.arange(T)[:, None]
    cause_fail, first_fail, first_cause = {}, np.full(W, T), np.array([""] * W, dtype=object)
    for cname, idx in crit["causes"]:
        f = np.zeros(W, dtype=bool)
        for i in idx:
            if thr[i] is None:
                continue
            ex = (step > act[i]) & (err[..., i] > thr[i])
            f |= ex.any(0)
            fe = np.where(ex.any(0), ex.argmax(0), T)
            first_cause = np.where(fe < first_fail, cname, first_cause)
            first_fail = np.minimum(first_fail, fe)
        cause_fail[cname] = f
    invalid = np.zeros(W, dtype=bool)
    invalid[d["non_finite_worlds"]] = True
    reached = d["completion_truncated"].astype(bool) & np.isclose(d["completion_progress"], 1.0, atol=1e-6)
    success = reached & ~invalid & ~np.logical_or.reduce(list(cause_fail.values()))
    rec, rep = float(success.mean()), float(d["reported_success_rate"])
    gates.append(("G5 逐世界成功率重算", abs(rec - rep) < 1e-9, f"重算 {rec:.4f} vs 报告 {rep:.4f}；门槛 {dict(zip(crit['error_names'], thr))}"))
    if a.predict_frame is not None:
        # 三角不等式：成功的世界在第 pf 帧（门槛生效后）朝向误差 ≤ 1.5 rad，所以它离自身初始朝向
        # ≥ 参考在该帧离初始朝向的角度 − 1.5 rad − 它第 0 帧的朝向误差；再留 0.1° 数值余量。逐世界判
        pf = int(a.predict_frame)
        turned = angle_deg(np.einsum("wij,wkj->wik", Ra[pf], Ra[0]))                       # (W,) 第 pf 帧离自身初始朝向
        bound = ang0_ref[pf] - GATE_DEG - ori[0] - 0.1                                       # (W,)
        slack = turned - bound
        ok = bool((slack[success] >= 0).all())  # 没有成功的世界时不适用（记为通过并注明）
        worst = slack[success].min() if success.any() else float("nan")
        detail = (f"参考该帧离初始 {ang0_ref[pf]:.1f}°；成功世界 {int(success.sum())} 个，转离初始朝向最小 "
                  f"{turned[success].min() if success.any() else float('nan'):.1f}°，比各自下界多出的最小余量 {worst:.2f}°")
        if not success.any():
            detail = "没有成功的世界，不适用"
        gates.append((f"G6 几何自检：成功世界第 {pf} 帧转离初始朝向 ≥ 各自下界", ok, detail))
    valid = all(ok for _, ok, _ in gates)
    print("== 闸")
    for name, ok, detail in gates:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}：{detail}")
    if cross is not None:
        p1, p2, n1, n2 = same["metrics"]["chord_sr"], cross["metrics"]["chord_sr"], same["world_count"], cross["world_count"]
        p = (p1 * n1 + p2 * n2) / (n1 + n2)
        z = (p1 - p2) / np.sqrt(max(p * (1 - p), 1e-12) * (1 / n1 + 1 / n2))
        print(f"  （描述）跨卡：本卡原样 ADD-AUC {same['metrics']['add_auc']:.4f} vs 参照 {cross['metrics']['add_auc']:.4f}；"
              f"chord_sr {p1:.4f} vs {p2:.4f}（z {z:.2f}）；参照 {n2} 世界")
    distinct = len(np.unique(np.round(ach[-1, :, 0, :], 5), axis=0))
    print(f"  （描述）最后一帧物体位姿去重后的不同轨迹 {distinct} 条（有效样本量的上限）")
    print(f"  VALIDITY {'PASS' if valid else 'FAIL'}")

    # ── Q0 ADD-AUC 分段 ───────────────────────────────────────────────
    segs = [(f"0–{s0 - 1}", 0, s0), (f"{s0}–{s1}", s0, s1 + 1), (f"{s1 + 1}–{T - 1}", s1 + 1, T)]
    pos = np.linalg.norm(ach[:, :, 0, :3] - refp[:, None, 0, :3], axis=-1) * 100             # (T,W) cm
    pos_static = np.linalg.norm(refp[:, 0, :3] - refp[0, 0, :3], axis=-1) * 100             # (T,) cm
    auc_static_full = auc(add_static)
    print(f"== Q0 ADD-AUC 分段（全程：policy {auc_full:.4f}，物体不动 {auc_static_full:.4f}，差 {auc_full - auc_static_full:+.4f}；"
          "贡献 = 段帧数/总帧数 × 段内 ADD-AUC，各段相加即全程）")
    tot_gap = auc_full - auc_static_full
    seg_gaps = []
    for label, f0, f1 in segs:
        n = f1 - f0
        ap_, as_ = auc(add[f0:f1]), auc(add_static[f0:f1])
        cp, cs = n / T * ap_, n / T * as_
        seg_gaps.append(cp - cs)
        share = (cp - cs) / tot_gap if abs(tot_gap) > 1e-12 else float("nan")
        print(f"  第 {label:<9} 帧 {n:>3}：段内 policy {ap_:.4f} / 不动 {as_:.4f}；贡献 {cp:.4f} / {cs:.4f}，差 {cp - cs:+.4f}"
              f"（占全程差 {share:.0%}）；位置误差均值 policy {pos[f0:f1].mean():.1f} cm / 不动 {pos_static[f0:f1].mean():.1f} cm；"
              f"ADD 中位 policy {np.median(add[f0:f1]) * 100:.1f} cm / 不动 {np.median(add_static[f0:f1]) * 100:.1f} cm")

    # ── Q1 朝向误差 ───────────────────────────────────────────────────
    ep20_frames = {int(round(f * a.frame_scale)) for f in (100, 269, 300, 359, 360)}
    kf = sorted({0, s0, (s0 + s1) // 2, s1, T - 1} | {f for f in ep20_frames if f < T})
    med = np.median(ori, 1)
    obj_ori_i = col["object_orientation_rad"]
    obj_act = act[obj_ori_i]
    ex = (step > obj_act) & (ori > GATE_DEG)
    first = np.where(ex.any(0), ex.argmax(0), -1)
    crossed = first >= 0
    print("== Q1 物体朝向误差（°）")
    print("  世界中位：" + "  ".join(f"{k}:{med[k]:.1f}" for k in kf))
    print("  分段平均：" + "；".join(f"{lbl} {ori[f0:f1].mean():.1f}" for lbl, f0, f1 in segs))
    if crossed.any():
        q25, q50, q75 = pct(first[crossed])
        print(f"  逐世界首次超过 1.5 rad：{crossed.mean():.1%} 的世界越过，首次越线帧四分位 {q25:.0f}/{q50:.0f}/{q75:.0f}")
    else:
        print("  逐世界首次超过 1.5 rad：没有世界越过")
    mc = np.where(med > GATE_DEG)[0]
    print(f"  （口径：世界中位曲线）首次超过 1.5 rad 在第 {int(mc[0]) if mc.size else '—'} 帧")

    # ── Q2 转动 ───────────────────────────────────────────────────────
    inc_r, inc_a = world_increments(Rr), world_increments(Ra)             # (T-1,3), (T-1,W,3)
    axis = inc_r[s0:s1].sum(0)
    axis /= np.linalg.norm(axis)
    phi_r = np.degrees(np.concatenate([[0.0], np.cumsum(inc_r @ axis)]))
    phi_a = np.degrees(np.concatenate([np.zeros((1, W)), np.cumsum(inc_a @ axis, 0)]))
    net_r = phi_r[s1] - phi_r[s0]
    sgn = np.sign(net_r) if net_r != 0 else 1.0
    net_a = phi_a[s1] - phi_a[s0]
    net_end = phi_a[-1] - phi_a[s0]
    peak = (sgn * (phi_a[s0:s1 + 1] - phi_a[s0])).max(0)
    ratio = net_a / net_r if net_r != 0 else np.full(W, np.nan)
    need = 1 - GATE_DEG / abs(net_r) if abs(net_r) > GATE_DEG else 0.0
    jumps = int((angle_deg(np.einsum("twij,twkj->twik", Ra[1:], Ra[:-1])) > 30).sum())  # 用转角而非 rotvec 范数：近 180° 也数得到
    unsigned_r = float(np.degrees(np.linalg.norm(inc_r[s0:s1], axis=-1).sum()))
    print("== Q2 绕参考主轴的转动")
    print(f"  参考：第 {s0}–{s1} 帧净转 {net_r:.1f}°（增量绝对值之和 {unsigned_r:.1f}°，含抖动）；主轴与竖直方向夹角 "
          f"{np.degrees(np.arccos(min(1.0, abs(axis[2])))):.1f}°；第 {s1} 帧不越 1.5 rad 所需净转比约 {need:.0%}")
    for name, v in ((f"净转动比（第 {s0}–{s1} 帧）", ratio), (f"净转动（°，第 {s0} 帧到末帧）", net_end), ("段内峰值进度（°）", peak)):
        q25, q50, q75 = pct(v)
        print(f"  {name}：四分位 {q25:.2f}/{q50:.2f}/{q75:.2f}")
    print(f"  净转动比达到所需比值的世界 {np.mean(ratio >= need):.1%}；实际单帧增量 > 30° 的跳变 {jumps} 次")
    print("  转动进度（°，世界中位 / 参考）：" + "  ".join(f"{k}:{np.median(phi_a[k]):.0f}/{phi_r[k]:.0f}" for k in kf))
    if valid:
        cls = [classify(n_, p_, net_r) for n_, p_ in zip(net_a, peak)]
        print("  分类（事先登记）：" + "；".join(f"{c} {cls.count(c)}" for c in ("净转过半", "转过但没保持或没跟上", "基本没转", "其他")))
    else:
        print("  分类：闸没过，不给")

    # ── Q2b 落后量（回放速度对照的主指标）──────────────────────────────
    # 从第 0 帧起算、绕同一主轴：关注段开始前提前转的部分也算进去（净转动比只从关注段起点算，会漏掉它）
    lag = sgn * (phi_r[:, None] - phi_a)                                   # (T,W) °，正 = 落后于参考
    in_window = crossed & (first >= s0) & (first <= s1)
    fps = diag.get("sim_control_fps")
    print("== Q2b 落后量（绕参考主轴、从第 0 帧起算，°；正 = 落后于参考，负 = 提前转了）")
    print(f"  世界中位：第 {s0} 帧 {np.median(lag[s0]):+.1f}；第 {s1} 帧 {np.median(lag[s1]):+.1f}；末帧 {np.median(lag[-1]):+.1f}")
    q25, q50, q75 = pct(lag[s1])
    print(f"  第 {s1} 帧落后量四分位 {q25:.1f}/{q50:.1f}/{q75:.1f}；第 {s0}–{s1} 帧内首次越过 1.5 rad 的世界 {in_window.mean():.1%}")
    phase_lag = []
    for f0, f1 in zip(a.phases or [], (a.phases or [])[1:]):
        acc = lag[f1] - lag[f0]                                            # (W,) 这一阶段新增的落后量
        dr = sgn * (phi_r[f1] - phi_r[f0])
        speed = f"，{abs(dr) / (f1 - f0) * fps:.0f}°/s" if fps else ""
        q25, q50, q75 = pct(acc)
        phase_lag.append(float(q50))
        print(f"  阶段 {f0}–{f1}：参考转 {dr:.1f}°（{abs(dr) / (f1 - f0):.2f}°/帧{speed}），policy 世界中位转 {np.median(sgn * (phi_a[f1] - phi_a[f0])):.1f}°；"
              f"新增落后量四分位 {q25:+.1f}/{q50:+.1f}/{q75:+.1f}")
    half = np.array([c == "净转过半" for c in cls]) if valid else np.zeros(W, dtype=bool)
    if half.any():
        print(f"  净转过半的 {int(half.sum())} 个世界：第 {s1} 帧落后量中位 {np.median(lag[s1][half]):+.1f}")

    # ── Q3 手 ─────────────────────────────────────────────────────────
    palm = err[..., col["palm_position_m"]] * 100
    dist = d["ref_palm_mesh_dist"].astype(np.float64) * 100
    pnames = [str(n) for n in d["ref_palm_names"]]
    sides = [str(s) for s in d["contact_sides"]]
    contact = d["contact_active"].astype(bool)
    print("== Q3 手（只报数）：palm 位置误差取两手中较大者（同评测器，门槛 0.15 m）；参考手到参考物体网格顶点的最近距离（palm 坐标帧，非手掌表面）")
    for label, f0, f1 in segs:
        cs = "，".join(f"{s} 接触 {contact[f0:f1, i].mean():.0%}" for i, s in enumerate(sides))
        ds = "，".join(f"{n} {np.median(dist[f0:f1, i]):.1f} cm" for i, n in enumerate(pnames))
        print(f"  第 {label:<9}：palm 误差中位 {np.median(palm[f0:f1]):.1f} cm；参考手到物体 {ds}；参考接触标记 {cs}")

    print(f"  距离所用顶点：{d['palm_dist_vertex_source']}（trimesh 与评测器顶点一致：{bool(d['trimesh_matches_evaluator'])}）")
    contact_dist = {}
    for i, sname in enumerate(sides):
        k = next((j for j, pn in enumerate(pnames) if pn.startswith(sname)), None)
        m = contact[:, i]
        if k is None or not m.any():
            print(f"  参考 {sname} 手接触帧 {int(m.sum())}：没有对应的 palm 帧或没有接触帧")
            continue
        contact_dist[sname] = float(np.median(dist[m, k]))
        print(f"  参考 {sname} 手的接触帧（{int(m.sum())} 帧）上，{pnames[k]} 到物体距离中位 {contact_dist[sname]:.1f} cm")

    # ── Q4 失败的世界 ─────────────────────────────────────────────────
    fail = ~success
    print(f"== Q4 任务原生判据：成功 {int(success.sum())}，失败 {int(fail.sum())}")
    if fail.any():
        q25, q50, q75 = pct(first_fail[fail])
        causes = first_cause[fail].tolist()
        print(f"  失败世界首次越线帧四分位 {q25:.0f}/{q50:.0f}/{q75:.0f}；首个越线的原因："
              + "；".join(f"{c} {causes.count(c)}" for c in dict.fromkeys(causes)))

    static_share = (seg_gaps[0] + seg_gaps[2]) / tot_gap if abs(tot_gap) > 1e-12 else float("nan")
    summary = {"worlds": W, "valid": valid, "add_auc": auc_full, "static_add_auc": auc_static_full, "segment_gap": seg_gaps,
               "non_manipulation_gap_share": static_share, "median_net_ratio": float(np.median(ratio)), "required_ratio": need,
               "palm_error_median_cm": [float(np.median(palm[f0:f1])) for _, f0, f1 in segs],
               "contact_palm_mesh_dist_cm": contact_dist, "success": int(success.sum()),
               "frames": T, "segments": [s0, s1], "ref_net_deg": float(net_r),
               "lag_deg": {"s0": float(np.median(lag[s0])), "s1": float(np.median(lag[s1])), "last": float(np.median(lag[-1]))},
               "phase_bounds": list(a.phases or []), "phase_lag_deg": phase_lag, "window_cross_share": float(in_window.mean()),
               "half_rotated": int(half.sum()), "half_rotated_lag_s1_deg": float(np.median(lag[s1][half])) if half.any() else None,
               "first_fail_causes": {c or "未越线": int(n) for c, n in zip(*np.unique(first_cause[~success].astype(str), return_counts=True))}}
    print("summary_json " + json.dumps(summary, ensure_ascii=False))

    if a.svg:
        lo, mid, hi = np.percentile(add * 100, [25, 50, 75], axis=1)       # 各帧在世界间的四分位
        lo_p, mid_p, hi_p = np.percentile(phi_a, [25, 50, 75], axis=1)
        lo_o, mid_o, hi_o = np.percentile(ori, [25, 50, 75], axis=1)
        mid_h = np.median(palm, 1)
        svg = svg_chart(a.title or str(d["policy"]), T, s0, s1, [
            ("物体 ADD（cm；绿：policy 世界中位与四分位；黑虚线：物体不动）", "", [("不动", add_static * 100, "#222222", True),
                                                                   ("policy 中位", mid, "#2e7d32", False)], (lo, hi), None),
            ("绕参考主轴的转动进度（°）", "°", [("参考", phi_r, "#222222", True), ("policy 中位", mid_p, "#2e7d32", False)], (lo_p, hi_p), None),
            ("物体朝向误差（°；红虚线：1.5 rad）", "°", [("中位", mid_o, "#2e7d32", False)], (lo_o, hi_o), GATE_DEG),
            ("palm 误差与参考手到物体的距离（cm）", "", [("palm 误差中位", mid_h, "#2e7d32", False)]
             + [(f"参考 {n} 到物体", dist[:, i], "#7a4f01", True) for i, n in enumerate(pnames)], None, None),
        ])
        with open(a.svg, "w") as f:
            f.write(svg)
        print(f"svg: {a.svg}")
    return 0 if valid else 3


if __name__ == "__main__":
    sys.exit(main())
