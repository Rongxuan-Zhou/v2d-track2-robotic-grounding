"""诊断 rollout 里机器人手掌离参考手掌多远（只用 numpy；读 diag_rollout.py --palms 存的 npz；容器内跑；只报不判）。
起因：agent-doc/plan-t1palm-2026-10-30.md 复核 MAJOR 1：手掌位置奖励（objective.palm_position_std）比的是机器人手掌对参考手掌，
诊断（finger_facts.py）报的是手掌到物体表面；这里补量前者。
报：每只手，参考接触帧上全部世界的 |机器人手掌 − 参考手掌| 的 10/50/90 分位（m）；失败世界（物体误差首次 > --onset-m）在离开前 −40、−20、−10、0 帧的中位；
另报开动帧之前（参考接触到开动之间）的中位，看接近段手是否已经落后。
用法（容器内）：python /repo/stages/evaluate/palm_ref_gap.py <npz> [--onset-m 0.10]   最后一行 PALM_REF_GAP_DONE
"""
import argparse
import sys

import numpy as np


def q3(x) -> str:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return "（无）" if x.size == 0 else " / ".join(f"{np.percentile(x, p):.3f}" for p in (10, 50, 90))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--onset-m", type=float, default=0.10)
    a = ap.parse_args()
    d = np.load(a.npz, allow_pickle=False)
    ach = d["achieved_pose_w"][:, :, 0].astype(np.float64)
    refp = d["reference_pose_w"][:, 0].astype(np.float64)
    T, W = ach.shape[:2]
    palm = d["robot_palm_pos"].astype(np.float64)          # (T,W,H,3)
    ref_palm = d["ref_palm_pos"].astype(np.float64)        # (T,H',3)
    rsides = [str(s) for s in d["robot_palm_sides"]]
    rnames = [str(s) for s in d["ref_palm_names"]]
    sides = [str(s) for s in d["contact_sides"]]
    contact = np.asarray(d["contact_active"], bool)
    err = np.linalg.norm(ach[..., :3] - refp[:, None, :3], axis=-1)
    over = err > a.onset_m
    t_o = np.where(over.any(0), over.argmax(0), -1)
    fail = t_o >= 0
    print(f"{a.npz}：T={T} W={W}；失败世界 {int(fail.sum())}，t_o 中位 {int(np.median(t_o[fail])) if fail.any() else -1}")
    for s in rsides:
        ri = rsides.index(s)
        rj = next(i for i, n in enumerate(rnames) if n.startswith(f"{s}_"))
        gap = np.linalg.norm(palm[:, :, ri] - ref_palm[:, None, rj], axis=-1)   # (T,W)
        cf = np.flatnonzero(contact[:, sides.index(s)]) if s in sides else np.array([], int)
        print(f"== {s} 手：参考接触 {cf.size} 帧")
        if cf.size:
            print(f"  接触帧、全部世界 |机器人手掌 − 参考手掌| m：{q3(gap[cf])}")
            pre = cf[cf < int(np.median(t_o[fail]))] if fail.any() else cf
            print(f"  接触帧里早于 t_o 中位的帧：{q3(gap[pre])}（{pre.size} 帧）")
        for off in (-40, -20, -10, 0):
            ws = [w for w in np.flatnonzero(fail) if 0 <= t_o[w] + off < T]
            if ws:
                print(f"  失败世界 t_o{off:+d}：中位 {np.median([gap[t_o[w] + off, w] for w in ws]):.3f} m")
    print("PALM_REF_GAP_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
