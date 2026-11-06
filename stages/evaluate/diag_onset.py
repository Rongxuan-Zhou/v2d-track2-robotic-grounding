"""举起再掉的失败世界：掉之前手贴没贴住（只用 numpy；与 diag_phase.py 同一容器里跑，读 diag_rollout.py --palms 存的 npz；只报不判）。
计划 agent-doc/plan-t2dg-2026-10-20.md 第 2 节 D1b（第 2 版，按复核 MAJOR 1 加）：diag_phase.py 的 G1/G2/K 在“抬起后才掉”的集上按掉的时刻切分，
不按原因；这里按每个失败世界自己的离开时刻看。判据（结果出来之前登记）：
  失败世界 = MP-SR 不成功的世界（同评测器与 diag_phase.py 的重算）。离开帧 t_o = 物体位置误差第一次 > --onset-m（默认 0.10 m）的帧；没有就记“没离开”（精度不够的失败）。
  看窗 [t_o − --pre, t_o)（默认 10 帧），只数参考里至少一只主动手在接触的帧：
    窗里没有这样的帧 → “参考无接触时离开”（碰掉、推走一类）；
    这些帧里“每只在接触的手都有 d_rob ≤ d_ref + --g-hand（默认 3 cm）”的帧不少于一半 → “贴住后掉”（滑脱一类）；否则 → “没贴住”。
  d_rob、d_ref 同 diag_phase.py：机器人手掌、参考手掌到实际、参考物体网格顶点的最近距离。
另报（只报）：t_o 相对开动帧 t_m 与最后接触帧 t_last 的分布；G2 的位移投到参考位移方向上的比值（窗 [t_m, t_m+30)）；npz 的 upstream_md5。
用法（容器内）：python /repo/stages/evaluate/diag_onset.py <npz> [--onset-m 0.10] [--pre 10] [--g-hand 0.03]   最后一行 DIAG_ONSET_DONE
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_phase import (
    G_WIN,
    MP_DEG,
    MP_POS,
    ONSET_DEG,
    ONSET_POS,
    geodesic,
    surface_dist,
)

KINDS = ("参考无接触时离开", "没贴住", "贴住后掉", "没离开")


def analyze(d, onset_m=0.10, pre=10, g_hand=0.03) -> dict:
    ach = d["achieved_pose_w"][:, :, 0].astype(np.float64)        # (T,W,7)
    refp = d["reference_pose_w"][:, 0].astype(np.float64)          # (T,7)
    T, W = ach.shape[:2]
    pos_err = np.linalg.norm(ach[..., :3] - refp[:, None, :3], axis=-1)   # (T,W)
    ori_err = geodesic(ach[..., 3:], refp[:, None, 3:])
    mp_ok = (pos_err.mean(0) < MP_POS) & (np.degrees(ori_err.mean(0)) < MP_DEG)
    sides = [str(s) for s in d["contact_sides"]]
    rsides = [str(s) for s in d["robot_palm_sides"]]
    rnames = [str(s) for s in d["ref_palm_names"]]
    ref_idx = {s: next(i for i, n in enumerate(rnames) if n.startswith(f"{s}_")) for s in sides}
    c_all = np.asarray(d["contact_active"], bool)
    active = [s for s in sides if c_all[:, sides.index(s)].any()]
    contact = np.stack([c_all[:, sides.index(s)] for s in active], -1)        # (T,K)
    t_c = min(int(np.flatnonzero(contact[:, k])[0]) for k in range(len(active)))
    t_last = max(int(np.flatnonzero(contact[:, k])[-1]) for k in range(len(active)))
    moved = [(np.linalg.norm(refp[t, :3] - refp[t_c, :3]) > ONSET_POS) or (np.degrees(geodesic(refp[t, 3:], refp[t_c, 3:])) > ONSET_DEG)
             for t in range(t_c, t_last + 1)]
    t_m = t_c + int(np.argmax(moved)) if any(moved) else None
    verts = d["object_vertices_o"][0].astype(np.float64)
    palm = d["robot_palm_pos"].astype(np.float64)
    ref_palm = d["ref_palm_pos"].astype(np.float64)
    d_ref = np.stack([surface_dist(ref_palm[:, ref_idx[s]], refp, verts) for s in active], -1)   # (T,K)
    over = pos_err > onset_m
    t_o = np.where(over.any(0), over.argmax(0), -1)
    fail = ~mp_ok
    kind = np.full(W, "", dtype=object)
    kind[fail & (t_o < 0)] = "没离开"
    need = fail & (t_o >= 0)
    lo_all = np.maximum(t_o - pre, 0)
    frames = sorted({t for w in np.flatnonzero(need) for t in range(lo_all[w], t_o[w])})
    att = {}   # (t) -> (W,) 这一帧在接触的手是否都贴住；这一帧参考没有手接触时不存
    for t in frames:
        ks = [k for k in range(len(active)) if contact[t, k]]
        if not ks:
            continue
        ok = np.ones(W, bool)
        for k in ks:
            ok &= surface_dist(palm[t, :, rsides.index(active[k])], ach[t], verts) <= d_ref[t, k] + g_hand
        att[t] = ok
    for w in np.flatnonzero(need):
        ts = [t for t in range(lo_all[w], t_o[w]) if t in att]
        if not ts:
            kind[w] = "参考无接触时离开"
        else:
            kind[w] = "贴住后掉" if np.mean([att[t][w] for t in ts]) >= 0.5 else "没贴住"
    proj = None
    if t_m is not None:
        e = min(T - 1, t_m + G_WIN - 1)
        dref = refp[e, :3] - refp[t_m, :3]
        n = np.linalg.norm(dref)
        if n > 1e-6:
            proj = ((ach[e, :, :3] - ach[t_m, :, :3]) @ (dref / n)) / n     # (W,)
    return {"T": T, "W": W, "mp_ok": mp_ok, "fail": fail, "kind": kind, "t_o": t_o, "t_c": t_c, "t_m": t_m, "t_last": t_last,
            "active": active, "proj": proj}


def q(x, qs=(10, 25, 50, 75, 90)) -> str:
    return " / ".join(f"{np.percentile(x, p):.0f}" for p in qs) if len(x) else "（无）"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--onset-m", type=float, default=0.10)
    ap.add_argument("--pre", type=int, default=10)
    ap.add_argument("--g-hand", type=float, default=0.03)
    a = ap.parse_args()
    d = np.load(a.npz, allow_pickle=False)
    r = analyze(d, a.onset_m, a.pre, a.g_hand)
    print(f"npz {a.npz}；policy {d['policy']}；世界 {r['W']}、帧 {r['T']}；upstream_md5 {d['upstream_md5'] if 'upstream_md5' in d.files else '（没有）'}")
    print(f"参考：主动手 {r['active']}，t_c {r['t_c']}，开动帧 t_m {r['t_m']}，最后接触帧 t_last {r['t_last']}")
    nf = int(r["fail"].sum())
    print(f"MP-SR 成功 {int(r['mp_ok'].sum())}/{r['W']}；失败 {nf}")
    if nf:
        for k in KINDS:
            n = int((r["kind"] == k).sum())
            print(f"  {k}：{n}（失败世界的 {n / nf:.1%}）")
        hit = r["t_o"][r["fail"] & (r["t_o"] >= 0)]
        print(f"  离开帧 t_o 分位 10/25/50/75/90：{q(hit)}")
        if r["t_m"] is not None:
            print(f"  t_o − t_m 分位：{q(hit - r['t_m'])}；t_o 早于 t_m 的 {int((hit < r['t_m']).sum())}、晚于 t_last 的 {int((hit > r['t_last']).sum())}")
        n1, n2 = int((r["kind"] == "没贴住").sum()), int((r["kind"] == "贴住后掉").sum())
        n0 = int((r["kind"] == "参考无接触时离开").sum())
        top = max((("参考无接触时离开", n0), ("没贴住", n1), ("贴住后掉", n2)), key=lambda kv: kv[1])
        rest = sorted((n0, n1, n2), reverse=True)
        verdict = top[0] if (rest[0] - rest[1]) / nf >= 0.10 else "混合"
        print(f"  D1b（登记的读法，三类里占比最大且领先第二名 10 个百分点以上）：{verdict}")
    if r["proj"] is not None:
        p = r["proj"][r["fail"]]
        print(f"  只报：G 窗 [t_m, t_m+30) 内实际位移投到参考位移方向上 ÷ 参考位移，失败世界分位 10/50/90："
              f"{np.percentile(p, 10):.2f} / {np.percentile(p, 50):.2f} / {np.percentile(p, 90):.2f}" if p.size else "  只报：没有失败世界")
    print("DIAG_ONSET_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
