"""MP-SR 为 0 的序列诊断：失败出在接近、抓取还是跟踪（只用 numpy；放在与评测器同一个 FlashCHORD 容器里跑，numpy 版本一致）。

计划 agent-doc/plan-diag-mpsr0-2026-10-02.md 第 3.2 节与第 4 节（判据执行前登记）。输入：diag_rollout.py --palms 存的 npz、这次评测的 JSON、
本机已有的同一 checkpoint 4096 世界评测 JSON（本机评测链算的）。
闸（任何一道不过：只打印数字、不分类，退出码 3）：
  V0 形状：T、W 与评测 JSON 一致，物体刚体 1 个，没有非有限世界，手掌数据在、各手能与参考接触标记、参考手掌对上，同步闸最大差 < 1e-6 m；
     手掌坐标帧闸（第 2 版，按审查 M1）：由 npz 重算的“两只手掌到参考手掌距离的较大者”与评测器自己每步记的 tracking_error 的 palm_position_m
     最大差 < 1e-4 m（评测器先按当前帧打分、再推进帧号，所以第 t 步对参考第 t 帧；另报错一帧的对齐差多少，只报）
  V1 同卡：两份 JSON 的 checkpoint_sha256 相同，且指标逐位相同，或 chord_sr |z| < 3 且 |ΔADD-AUC| < 0.005
  V2 重算：按评测器 metrics.py 的公式由 npz 重算的 MP-SR、SP-SR 与这次 JSON 的 maniptrans_sr、spider_sr_uncentered 相差 < 1e-9；
     由 npz 逐帧 ADD 重算的 ADD-AUC 与 JSON 相差 < 1e-6
参考分段（只用参考）：主动手 = 参考接触标记里有接触帧的手；t_c = 主动手最早的接触帧；t_last = 最晚的接触帧；
  开动帧 t_m = t_c 之后第一个参考物体相对 t_c 移动 > 2 cm 或转过 > 5° 的帧（没有就记“不动”）。
逐世界：物体位置、朝向误差（同评测器）；d_rob = 机器人手掌到实际物体表面的距离，d_ref = 参考手掌到参考物体表面的距离（都用评测器的 500 个顶点）。
分类：MP-SR 成功的世界记“成功”（身上若有下面的标记记作误报，供对照检查）；失败的世界按顺序先中先停：
  A1 物体被提前碰动：[t_c, t_c+10) 帧物体位置误差中位 > 3 cm 或朝向误差中位 > 30°
  A2 手没到位：t_c 时在接触的某只手，[t_c, t_c+10) 帧 d_rob − d_ref 的中位 > 5 cm
  G1 手没贴住：开动窗口 [t_m, t_m+30)（参考物体不动时用 [t_c, t_c+30)）里，“在接触的每只手都有 d_rob ≤ d_ref + 3 cm”的帧不到一半
  G2 物体没跟上（只在参考物体开动时查）：窗口首末参考移动 ≥ 5 cm 而实际不到一半，或参考转过 ≥ 15° 而实际不到一半
  K  跟踪失败：以上都不中
结论（第 4 节）：失败世界里占比最大的一组（接近 = A1+A2，抓取 = G1+G2，跟踪 = K）就是主要失败环节；前两组相差不到 10 个百分点记“混合”。
对照（--role control）：成功世界里带接近或抓取标记的比例 ≤ 10% 记 CONTROL_OK，否则 CONTROL_FAIL（退出码 4，分类门限不可信）。
另报：各段对时间平均误差的贡献，接触段里手贴着物体的帧比例，灵敏度（A2 用 2.5、10 cm，G1 用 1.5、6 cm），每类挑 3 个世界号供 viser 复看；
  第 2 版另加（按审查 M2，只报）：A2 世界的手何时到位（相对 t_c），A1 世界在第 0–9 帧与 t_c 前 10 帧的位置误差，K 世界里接触段贴合不到一半的比例，
  失败世界里位置不过、朝向不过、两样都不过各多少，G1 实际数了几帧。
用法：python /repo/stages/evaluate/diag_phase.py <npz> --diag-json D --same-card-json E [--role main|control]
"""
import argparse
import json
import sys

import numpy as np

MP_POS, MP_DEG = 0.03, 30.0          # metrics.MANIPTRANS_*
SP_POS, SP_RAD = 0.1, 0.5            # metrics.SPIDER_*
AUC_T = np.arange(0.01, 0.10, 0.01)  # metrics.ADD_THRESHOLDS_M
A_WIN, G_WIN = 10, 30
A_POS, A_DEG, A_HAND = 0.03, 30.0, 0.05
G_HAND, G_FRAC, G_POS, G_DEG, G_RATIO = 0.03, 0.5, 0.05, 15.0, 0.5
ONSET_POS, ONSET_DEG = 0.02, 5.0
CONTROL_MAX_FALSE = 0.10
GROUPS = {"接近": ("A1", "A2"), "抓取": ("G1", "G2"), "跟踪": ("K",)}


def geodesic(a, b):
    """同评测器 metrics.quaternion_geodesic_angle：不先归一化，sin 半角形式。"""
    dot = np.clip(np.abs(np.sum(a * b, axis=-1)), 0.0, 1.0)
    return 2.0 * np.arctan2(np.sqrt(np.maximum(0.0, 1.0 - dot * dot)), dot)


def rotmat(q):
    """同评测器 metrics._rotation_matrix（xyzw）。"""
    x, y, z, w = (q[..., i] for i in range(4))
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1).reshape(*q.shape[:-1], 3, 3)


def surface_dist(palm, pose, verts):
    """palm (..., 3)、pose (..., 7) 前面的维度一致；verts (V, 3) 是物体系顶点 → 手掌到物体表面（顶点）的最近距离 (...)"""
    world = np.einsum("...ij,vj->...vi", rotmat(pose[..., 3:]), verts) + pose[..., None, :3]
    return np.sqrt(((world - palm[..., None, :]) ** 2).sum(-1).min(-1))


def auc(add):
    x = np.asarray(add, np.float64)
    return float(np.trapezoid([np.mean(x < t) for t in AUC_T], x=np.linspace(0, 1, AUC_T.size)))


def palm_frame_check(d, sides, rsides, ref_idx):
    """重算 max_h ‖robot_palm[t,w,h] − ref_palm[t,h]‖，与评测器记的 tracking_error 的 palm_position_m 比。返回 {对齐: 最大差}：
    0 是第 t 步对参考第 t 帧（判据用这一个），+1、−1 是错一帧（只报，用来说明对齐选对了）。"""
    names = [str(n) for n in d["tracking_error_names"]]
    if "palm_position_m" not in names:
        return {}
    rec = d["tracking_error"][..., names.index("palm_position_m")].astype(np.float64)        # (T,W)
    palm = d["robot_palm_pos"].astype(np.float64)                                            # (T,W,H,3)
    ref = d["ref_palm_pos"].astype(np.float64)                                               # (T,F,3)
    order = [ref_idx[s] for s in rsides]
    out = {}
    for shift in (0, 1, -1):
        T = palm.shape[0]
        lo, hi = max(0, -shift), min(T, T - shift)
        dist = np.linalg.norm(palm[lo:hi] - ref[lo + shift:hi + shift][:, None][:, :, order], axis=-1).max(-1)
        out[shift] = float(np.abs(dist - rec[lo:hi]).max())
    return out


def same_card(m1, m2, n1, n2):
    numeric = [k for k, v in m1.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if all(m1[k] == m2.get(k) for k in numeric) and n1 == n2:
        return True, f"{len(numeric)} 项指标逐位相同"
    p1, p2 = m1["chord_sr"], m2["chord_sr"]
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = np.sqrt(max(p * (1 - p), 1e-12) * (1 / n1 + 1 / n2))
    z, dauc = (p1 - p2) / se, m1["add_auc"] - m2["add_auc"]
    return abs(z) < 3 and abs(dauc) < 0.005, f"不逐位相同；chord_sr {p1:.4f} vs {p2:.4f}（|z| {abs(z):.2f}），ADD-AUC 差 {dauc:+.5f}"


def classify(pos_err, ori_err, d_rob, d_ref, contact, active, t_c, t_m, ach, refp, a_hand=A_HAND, g_hand=G_HAND):
    """返回各标记的布尔数组 (W,)。d_rob[t, w, k]、d_ref[t, k] 按 active 的顺序；contact[t, k] 同。"""
    T, W = pos_err.shape
    wa = slice(t_c, min(T, t_c + A_WIN))
    a1 = (np.median(pos_err[wa], axis=0) > A_POS) | (np.degrees(np.median(ori_err[wa], axis=0)) > A_DEG)
    at_c = [k for k in range(len(active)) if contact[t_c, k]]
    a2 = np.zeros(W, bool)
    for k in at_c:
        a2 |= np.median(d_rob[wa, :, k] - d_ref[wa, k][:, None], axis=0) > a_hand
    g0 = t_m if t_m is not None else t_c
    wg = range(g0, min(T, g0 + G_WIN))
    ok_frames, counted = np.zeros(W), 0
    for t in wg:
        ks = [k for k in range(len(active)) if contact[t, k]]
        if not ks:
            continue
        counted += 1
        ok_frames += np.all([d_rob[t, :, k] <= d_ref[t, k] + g_hand for k in ks], axis=0)
    g1 = (ok_frames / counted < G_FRAC) if counted else np.zeros(W, bool)
    info = {"g_window": (wg.start, wg.stop - 1), "g_counted": counted}
    g2 = np.zeros(W, bool)
    if t_m is not None:
        e = wg[-1]
        dp_ref = float(np.linalg.norm(refp[e, :3] - refp[t_m, :3]))
        da_ref = float(geodesic(refp[e, 3:], refp[t_m, 3:]))
        dp_ach = np.linalg.norm(ach[e, :, :3] - ach[t_m, :, :3], axis=-1)
        da_ach = geodesic(ach[e, :, 3:], ach[t_m, :, 3:])
        if dp_ref >= G_POS:
            g2 |= dp_ach < G_RATIO * dp_ref
        if np.degrees(da_ref) >= G_DEG:
            g2 |= da_ach < G_RATIO * da_ref
    return {"A1": a1, "A2": a2, "G1": g1, "G2": g2, "_info": info}


def categories(flags, mp_ok):
    cat = np.full(mp_ok.shape, "K", dtype=object)
    for name in ("G2", "G1", "A2", "A1"):   # 倒着覆盖，结果就是“先中先停”
        cat[flags[name]] = name
    cat[mp_ok] = "S"
    return cat


def group_shares(cat):
    fail = cat != "S"
    n = int(fail.sum())
    return n, {g: (int(np.isin(cat[fail], names).sum()) / n if n else float("nan")) for g, names in GROUPS.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--diag-json", required=True)
    ap.add_argument("--same-card-json", required=True)
    ap.add_argument("--role", choices=("main", "control"), default="main")
    a = ap.parse_args()
    d = np.load(a.npz, allow_pickle=False)
    diag, same = json.load(open(a.diag_json)), json.load(open(a.same_card_json))
    ach4 = d["achieved_pose_w"].astype(np.float64)
    refp3 = d["reference_pose_w"].astype(np.float64)
    T, W, B = ach4.shape[:3]
    print(f"policy {d['policy']}；世界 {W}，帧 {T}；角色 {a.role}")

    gates = []
    has_palms = "robot_palm_pos" in d.files
    sides = [str(s) for s in d["contact_sides"]] if "contact_sides" in d.files else []
    rsides = [str(s) for s in d["robot_palm_sides"]] if has_palms else []
    rnames = [str(s) for s in d["ref_palm_names"]]
    ref_idx = {s: next((i for i, n in enumerate(rnames) if n.startswith(f"{s}_")), None) for s in sides}
    sync = float(d["palm_object_sync_max_abs_m"]) if has_palms else float("nan")
    frames_ok = has_palms and all(s in ref_idx and ref_idx[s] is not None for s in rsides)
    pfc = palm_frame_check(d, sides, rsides, ref_idx) if frames_ok else {}
    v0 = (T == diag["step_count"] and W == diag["world_count"] and B == 1 and len(d["non_finite_worlds"]) == 0 and has_palms
          and d["robot_palm_pos"].shape == (T, W, len(rsides), 3) and all(s in rsides for s in sides)
          and all(i is not None for i in ref_idx.values()) and np.isfinite(sync) and sync < 1e-6 and d["reference_pose_w"].dtype == np.float64
          and frames_ok and 0 in pfc and pfc[0] < 1e-4)
    gates.append(("V0 形状与同步", v0, f"T {T}/{diag['step_count']}，W {W}/{diag['world_count']}，B {B}，非有限世界 {len(d['non_finite_worlds'])}，"
                  f"手掌 {rsides}，参考接触 {sides}，参考手掌 {rnames}，同步最大差 {sync:.3g} m，参考位姿 {d['reference_pose_w'].dtype}；"
                  f"手掌坐标帧闸：与评测器 palm_position_m 最大差 {pfc.get(0, float('nan')):.3g} m（错一帧时 +1 {pfc.get(1, float('nan')):.3g}、−1 {pfc.get(-1, float('nan')):.3g}，只报）"))
    sc_ok, sc_note = same_card(diag["metrics"], same["metrics"], diag["world_count"], same["world_count"])
    sc_ok = sc_ok and diag.get("checkpoint_sha256") == same.get("checkpoint_sha256")
    gates.append(("V1 同卡", sc_ok, sc_note + ("" if diag.get("checkpoint_sha256") == same.get("checkpoint_sha256") else "；checkpoint_sha256 不同")))
    ach, refp = ach4, refp3[:, None]   # 同评测器 _aligned_poses：(T,W,B,7) 对 (T,1,B,7)
    position = np.linalg.norm(ach[..., :3] - refp[..., :3], axis=-1).mean(axis=0)   # (W,B)，同 _per_world_pose_error
    orientation = geodesic(ach[..., 3:], refp[..., 3:]).mean(axis=0)
    sp = (position.mean(-1) <= SP_POS) & (orientation.mean(-1) <= SP_RAD)
    mp_ok = (position[:, 0] < MP_POS) & (np.degrees(orientation[:, 0]) < MP_DEG)
    m = diag["metrics"]
    add_auc = auc(d["add"])
    v2 = abs(mp_ok.mean() - m["maniptrans_sr"]) < 1e-9 and abs(sp.mean() - m["spider_sr_uncentered"]) < 1e-9 and abs(add_auc - m["add_auc"]) < 1e-6
    gates.append(("V2 重算", v2, f"MP-SR {mp_ok.mean():.6f} vs {m['maniptrans_sr']:.6f}，SP-SR {sp.mean():.6f} vs {m['spider_sr_uncentered']:.6f}，"
                  f"ADD-AUC {add_auc:.6f} vs {m['add_auc']:.6f}"))
    print("== 闸")
    for name, ok, note in gates:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}：{note}")
    valid = all(ok for _, ok, _ in gates)
    print(f"  VALIDITY {'PASS' if valid else 'FAIL'}")
    if not valid:
        print("闸没过：只打印上面的数字，不分类")
        return 3

    contact_all = np.asarray(d["contact_active"], bool)
    active = [s for s in sides if contact_all[:, sides.index(s)].any()]
    if not active:
        print("参考里没有接触帧：接近、抓取的判据用不了，不分类")
        return 3
    contact = np.stack([contact_all[:, sides.index(s)] for s in active], -1)
    firsts = [int(np.flatnonzero(contact[:, k])[0]) for k in range(len(active))]
    lasts = [int(np.flatnonzero(contact[:, k])[-1]) for k in range(len(active))]
    t_c, t_last = min(firsts), max(lasts)
    p_ref, q_ref = refp3[:, 0, :3], refp3[:, 0, 3:]
    moved = [(np.linalg.norm(p_ref[t] - p_ref[t_c]) > ONSET_POS) or (np.degrees(geodesic(q_ref[t], q_ref[t_c])) > ONSET_DEG)
             for t in range(t_c, t_last + 1)]
    t_m = t_c + int(np.argmax(moved)) if any(moved) else None
    print("== 参考分段（只用参考）")
    for k, s in enumerate(active):
        print(f"  主动手 {s}：接触 {int(contact[:, k].sum())} 帧，第 {firsts[k]}–{lasts[k]} 帧")
    seg = lambda lo, hi: f"{lo}–{hi}" if hi >= lo else "无"   # noqa: E731
    print(f"  t_c {t_c}，t_last {t_last}，开动帧 t_m {t_m if t_m is not None else '—（参考物体在接触段里不动）'}；段：接近 {seg(0, t_c - 1)}，接触 {t_c}–{t_last}，之后 {seg(t_last + 1, T - 1)}")

    a0 = ach4[:, :, 0]
    pos_err = np.linalg.norm(a0[..., :3] - p_ref[:, None], axis=-1)            # (T,W)
    ori_err = geodesic(a0[..., 3:], q_ref[:, None])                            # (T,W)
    verts = d["object_vertices_o"][0].astype(np.float64)
    palm = d["robot_palm_pos"].astype(np.float64)
    ref_palm = d["ref_palm_pos"].astype(np.float64)
    d_ref = np.stack([surface_dist(ref_palm[:, ref_idx[s]], refp3[:, 0], verts) for s in active], -1)   # (T,K)
    g0 = t_m if t_m is not None else t_c
    t_hi = min(T, max(t_last + 1, g0 + G_WIN, t_c + A_WIN))
    d_rob = np.full((T, W, len(active)), np.nan)
    for t in range(t_c, t_hi):
        for k, s in enumerate(active):
            d_rob[t, :, k] = surface_dist(palm[t, :, rsides.index(s)], a0[t], verts)

    flags = classify(pos_err, ori_err, d_rob, d_ref, contact, active, t_c, t_m, a0, refp3[:, 0])
    cat = categories(flags, mp_ok)
    nfail, shares = group_shares(cat)
    print("== 逐世界分类（MP-SR 成功的先归成功；失败的按 A1 → A2 → G1 → G2 → K 先中先停）")
    for c in ("S", "A1", "A2", "G1", "G2", "K"):
        n = int((cat == c).sum())
        print(f"  {c:2s} {n:5d}（{n / W:.1%}）")
    order = sorted(shares.items(), key=lambda kv: -kv[1]) if nfail else []
    if nfail:
        print("  失败世界 " + str(nfail) + " 个，各组占比：" + "，".join(f"{g} {v:.1%}" for g, v in shares.items()))
        top, second = order[0], order[1]
        verdict = "混合（" + "、".join(g for g, _ in order[:2]) + "）" if top[1] - second[1] < 0.10 else top[0]
        print(f"  主要失败环节（第 4 节的规则）：{verdict}")
    else:
        verdict = "没有失败世界"
        print("  没有失败世界")

    gi = flags["_info"]
    print(f"  G1 的窗口第 {gi['g_window'][0]}–{gi['g_window'][1]} 帧，其中有手在接触、实际数了 {gi['g_counted']} 帧"
          + ("（0 帧：G1 不判）" if gi["g_counted"] == 0 else ""))
    print("== 读分类时要分清的几件事（第 2 版加，只报）")
    fail_w = cat != "S"
    pos_bad = position[:, 0] >= MP_POS
    ori_bad = np.degrees(orientation[:, 0]) >= MP_DEG
    if fail_w.any():
        nfw = int(fail_w.sum())
        print(f"  失败世界 {nfw} 个：只有位置不过 {int((fail_w & pos_bad & ~ori_bad).sum())}，只有朝向不过 {int((fail_w & ~pos_bad & ori_bad).sum())}，"
              f"两样都不过 {int((fail_w & pos_bad & ori_bad).sum())}（对称物体的朝向项要单独看）")
    a2w = np.flatnonzero(cat == "A2")
    if a2w.size:
        at_c = [k for k in range(len(active)) if contact[t_c, k]]
        arrive = []
        for w in a2w:
            okt = np.all([d_rob[t_c:t_hi, w, k] <= d_ref[t_c:t_hi, k] + G_HAND for k in at_c], axis=0)
            arrive.append(int(np.argmax(okt)) if okt.any() else None)
        got = [x for x in arrive if x is not None]
        q = np.percentile(got, [25, 50, 75]).round(1).tolist() if got else []
        print(f"  A2 世界 {a2w.size} 个：手到位（d_rob ≤ d_ref + 3 cm）相对 t_c 的帧数四分位 {q}；30 帧内到位 {sum(x is not None and x < G_WIN for x in arrive)}，"
              f"更晚才到 {sum(x is not None and x >= G_WIN for x in arrive)}，到第 {t_hi - 1} 帧都没到 {sum(x is None for x in arrive)}")
    a1w = np.flatnonzero(cat == "A1")
    if a1w.size:
        e0 = np.median(pos_err[0:min(10, T)][:, a1w]) * 100
        e1 = np.median(pos_err[max(0, t_c - 10):t_c][:, a1w]) * 100 if t_c > 0 else float("nan")
        print(f"  A1 世界 {a1w.size} 个：位置误差中位，第 0–9 帧 {e0:.1f} cm，t_c 前 10 帧 {e1:.1f} cm（前者已经大说明一开始就偏了，不是手碰的）")
    kw = np.flatnonzero(cat == "K")
    if kw.size:
        rk = []
        for k_ in range(len(active)):
            fr = contact[:, k_].copy(); fr[t_hi:] = False
            if fr.any():
                rk.append((d_rob[fr][:, kw, k_] <= d_ref[fr, k_][:, None] + G_HAND).mean(0))
        if rk:
            low = (np.min(rk, axis=0) < 0.5)
            print(f"  K 世界 {kw.size} 个：接触段里某只主动手贴合不到一半的 {int(low.sum())} 个（{low.mean():.0%}，开动窗口之后才松手的会落在这里）")
    print("== 各段对时间平均误差的贡献（失败世界的中位；贡献 = 段内误差之和 / 总帧数，各段相加即全程平均）")
    fail = cat != "S"
    phases = {"接近": slice(0, t_c), "接触": slice(t_c, t_last + 1), "之后": slice(t_last + 1, T)}
    for name, sl in phases.items():
        if fail.any():
            cp = pos_err[sl][:, fail].sum(0) / T
            co = np.degrees(ori_err[sl][:, fail].sum(0) / T)
            sh = cp / np.maximum(position[fail, 0], 1e-12)
            print(f"  {name}（{sl.stop - sl.start} 帧）：位置 {np.median(cp) * 100:.2f} cm（占全程 {np.median(sh):.0%}），朝向 {np.median(co):.1f}°")
    print("== 接触段里手贴着物体的帧比例（d_rob ≤ d_ref + 3 cm，只数参考里这只手在接触的帧；世界中位，成功 / 失败）")
    for k, s in enumerate(active):
        fr = contact[:, k].copy()
        fr[t_hi:] = False
        if fr.any():
            okf = (d_rob[fr, :, k] <= d_ref[fr, k][:, None] + G_HAND).mean(0)
            print(f"  {s}：成功 {np.median(okf[~fail]) if (~fail).any() else float('nan'):.0%}，失败 {np.median(okf[fail]) if fail.any() else float('nan'):.0%}"
                  f"（第 {t_c}–{t_hi - 1} 帧里的 {int(fr.sum())} 帧）")
    print("== 灵敏度（只报，不改结论的门限）")
    sens = {}
    for label, kw in (("A2 2.5 cm", {"a_hand": 0.025}), ("A2 10 cm", {"a_hand": 0.10}), ("G1 1.5 cm", {"g_hand": 0.015}), ("G1 6 cm", {"g_hand": 0.06})):
        n2, sh2 = group_shares(categories(classify(pos_err, ori_err, d_rob, d_ref, contact, active, t_c, t_m, a0, refp3[:, 0], **kw), mp_ok))
        sens[label] = sh2
        print(f"  {label}：" + "，".join(f"{g} {v:.1%}" for g, v in sh2.items()))
    print("== 每类挑 3 个世界（世界平均位置误差最接近这一类的中位）")
    picks = {}
    for c in ("S", "A1", "A2", "G1", "G2", "K"):
        idx = np.flatnonzero(cat == c)
        if idx.size:
            med = np.median(position[idx, 0])
            sel = idx[np.lexsort((idx, np.abs(position[idx, 0] - med)))][:3]
            picks[c] = [int(i) for i in sel]
            print(f"  {c}：世界 {picks[c]}（平均位置误差 {', '.join(f'{position[i, 0] * 100:.1f} cm' for i in sel)}）")
    rc = 0
    false_rate = None
    if a.role == "control":
        succ = cat == "S"
        any_flag = flags["A1"] | flags["A2"] | flags["G1"] | flags["G2"]
        false_rate = float((succ & any_flag).sum() / succ.sum()) if succ.any() else float("nan")
        ok = succ.any() and false_rate <= CONTROL_MAX_FALSE
        print(f"== 对照：成功世界 {int(succ.sum())} 个，其中带接近或抓取标记 {int((succ & any_flag).sum())} 个（{false_rate:.1%}，上限 {CONTROL_MAX_FALSE:.0%}）")
        print("CONTROL_OK" if ok else "CONTROL_FAIL")
        rc = 0 if ok else 4
    summary = {"worlds": W, "frames": T, "role": a.role, "valid": True, "t_c": t_c, "t_m": t_m, "t_last": t_last, "active": active,
               "counts": {c: int((cat == c).sum()) for c in ("S", "A1", "A2", "G1", "G2", "K")}, "fail_worlds": nfail,
               "group_shares": shares, "verdict": verdict, "sensitivity": sens, "picks": picks, "control_false_rate": false_rate,
               "mp_sr": float(mp_ok.mean()), "sp_sr": float(sp.mean()), "add_auc": add_auc}
    print("summary_json " + json.dumps(summary, ensure_ascii=False))
    return rc


if __name__ == "__main__":
    sys.exit(main())
