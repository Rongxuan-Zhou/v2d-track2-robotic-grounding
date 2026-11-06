"""诊断 rollout 里 Dex3 手指合了多少（只用 numpy；读 diag_rollout.py --palms --fingers 存的 npz；容器内跑；只报不判）。
起因：agent-doc/experiments/2026-10-30-t1hard-diag.md：参考手指全程是 0，策略只能靠残差合手；要分开"策略没让手合"（目标角也接近 0）
与"让合了但合不上"（目标角大、实际角小，被物体挡住或力矩不够）。
做法：
  - 失败世界 = 物体位置误差第一次 > --onset-m 的世界，t_o 为那一帧（同 diag_onset.py 的读法，不重算 MP-SR）；
  - 每只手的合手量 = 7 个手指关节角绝对值之和（rad）；目标量 = 7 个 PD 目标角绝对值之和；差 = 7 个 |目标 − 实际| 之和；
  - 报：参考里这只手接触的帧上，所有世界的合手量与目标量的 10/50/90 分位；失败世界在 t_o − 20、t_o − 10、t_o − 5、t_o、t_o + 5 帧的三量中位，
    以及同一帧机器人手掌到实际物体网格的最近距离中位（diag_phase.surface_dist）与参考手掌到参考物体的最近距离；
  - 另报每个手指关节在接触帧上的实际角中位与目标角中位（看是哪根手指在动）。
用法（容器内）：python /repo/stages/evaluate/finger_facts.py <npz> [--onset-m 0.10]   最后一行 FINGER_FACTS_DONE
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_phase import surface_dist

OFFSETS = (-20, -10, -5, 0, 5)


def q3(x) -> str:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return "（无）" if x.size == 0 else " / ".join(f"{np.percentile(x, p):.2f}" for p in (10, 50, 90))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--onset-m", type=float, default=0.10)
    a = ap.parse_args()
    d = np.load(a.npz, allow_pickle=False)
    for k in ("robot_finger_q", "robot_finger_target", "robot_palm_pos"):
        if k not in d:
            print(f"npz 里没有 {k}：要用 diag_rollout.py --palms --fingers 存")
            return 2
    ach = d["achieved_pose_w"][:, :, 0].astype(np.float64)    # (T,W,7)
    refp = d["reference_pose_w"][:, 0].astype(np.float64)      # (T,7)
    T, W = ach.shape[:2]
    bad = set(int(x) for x in d["non_finite_worlds"])
    ok = np.array([w not in bad for w in range(W)])
    names = [str(x) for x in d["robot_finger_names"]]
    fq = d["robot_finger_q"].astype(np.float64)                 # (T,W,14)
    ft = d["robot_finger_target"].astype(np.float64)
    sides = [str(s) for s in d["contact_sides"]]
    contact = np.asarray(d["contact_active"], bool)              # (T,S)
    rsides = [str(s) for s in d["robot_palm_sides"]]
    rnames = [str(s) for s in d["ref_palm_names"]]
    palm = d["robot_palm_pos"].astype(np.float64)                # (T,W,H,3)
    ref_palm = d["ref_palm_pos"].astype(np.float64)              # (T,H',3)
    verts = d["object_vertices_o"][0].astype(np.float64)
    pos_err = np.linalg.norm(ach[..., :3] - refp[:, None, :3], axis=-1)
    over = pos_err > a.onset_m
    t_o = np.where(over.any(0), over.argmax(0), -1)
    fail = ok & (t_o >= 0)
    print(f"{a.npz}：T={T} W={W}（非有限 {len(bad)}）；物体误差超过 {a.onset_m:.2f} m 的世界 {int(fail.sum())}，t_o 分位 10/50/90 {q3(t_o[fail])}")
    for s in sides:
        cols = [i for i, n in enumerate(names) if n.startswith(f"{s}_")]
        if len(cols) != 7:
            print(f"{s}：手指关节 {len(cols)} 个（应为 7），跳过")
            continue
        ci = sides.index(s)
        cf = np.flatnonzero(contact[:, ci])
        print(f"== {s} 手：参考接触 {cf.size} 帧" + (f"（第 {cf[0]}–{cf[-1]} 帧）" if cf.size else ""))
        close = np.abs(fq[:, :, cols]).sum(-1)                   # (T,W)
        targ = np.abs(ft[:, :, cols]).sum(-1)
        gap = np.abs(ft[:, :, cols] - fq[:, :, cols]).sum(-1)
        if cf.size:
            print(f"  接触帧、全部世界：合手量 {q3(close[cf][:, ok])} rad；目标量 {q3(targ[cf][:, ok])} rad；|目标−实际| {q3(gap[cf][:, ok])} rad")
            for j in cols:
                print(f"    {names[j]}：实际中位 {np.median(fq[cf][:, ok, j]):+.3f}，目标中位 {np.median(ft[cf][:, ok, j]):+.3f} rad")
        if s in rsides and fail.any():
            ri = rsides.index(s)
            rj = next((i for i, n in enumerate(rnames) if n.startswith(f"{s}_")), None)
            print("  失败世界，相对 t_o\t合手量\t目标量\t|目标−实际|\t机器人手掌→物体 m\t参考手掌→参考物体 m\t参考接触")
            for off in OFFSETS:
                ws = [w for w in np.flatnonzero(fail) if 0 <= t_o[w] + off < T]
                if not ws:
                    continue
                ts = np.array([t_o[w] + off for w in ws])
                wa = np.array(ws)
                dr = [float(surface_dist(palm[t, w:w + 1, ri], ach[t, w:w + 1], verts)[0]) for t, w in zip(ts, wa)]
                tm = int(np.median(ts))
                dref = float(surface_dist(ref_palm[tm:tm + 1, rj], refp[tm:tm + 1], verts)[0]) if rj is not None else float("nan")
                print(f"  {off:+d}\t{np.median(close[ts, wa]):.2f}\t{np.median(targ[ts, wa]):.2f}\t{np.median(gap[ts, wa]):.2f}"
                      f"\t{np.median(dr):.3f}\t{dref:.3f}（第 {tm} 帧）\t{np.mean(contact[ts, ci]):.2f}")
    print("FINGER_FACTS_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
