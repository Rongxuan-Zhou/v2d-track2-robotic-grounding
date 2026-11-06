"""参考里被拿着的物体相对手掌有没有“打滑”（训练机上经 scripts/dev/in-flash-chord 在容器里跑，只用 CPU；只读参考 parquet，只报不判）。
起因：agent-doc/plan-t3hand-2026-10-27.md 复核 BLOCKER 1 与导出误差（results/ml40/t3hand/rollout_error_*.txt）：5 集跟丢的“重建”类都跟过了首次接触，
丢在参考物体快动或反向的那一段。被拿着的刚体应随手一起动：物体在手掌系里的位置应近似不变。这里逐帧算：
  rel = R_palmᵀ (p_obj − p_palm)（物体第一个刚体的位置在较近那只手掌系里）；滑移速度 = |Δrel| × fps；物体速度 = |Δp_obj| × fps；手掌速度。
只在该手 hand_contact_active 的帧上统计。读 ee_pose_w（手掌位姿，四元数 wxyz）、object_body_position / object_body_wxyz、hand_contact_active、hand_sides。
报：接触帧里滑移速度的中位、95% 分位、> --fast 的帧数；每段连续接触（≥ 15 帧）里 rel 偏离该段中位的位移（中位 / 95% / 最大，即“把物体刚性挂到手上要挪多远”）；
--range a:b:s 里每 s 帧一行（帧、接触、物体速度、手掌速度、滑移速度、rel 相对接触起点的位移）。
用法（容器内）：python /repo/stages/retarget/ref_hand_object_slip.py [--fps 30] [--fast 0.3] [--range a:b:s] <motion parquet>...   最后一行 REF_SLIP_DONE
"""
import argparse
import sys

import numpy as np
import pyarrow.parquet as pq


def quat_mat(q, order="wxyz"):
    q = np.asarray(q, dtype=float)
    if order == "wxyz":
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    else:
        x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z); R[..., 0, 1] = 2 * (x * y - z * w); R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w); R[..., 1, 1] = 1 - 2 * (x * x + z * z); R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w); R[..., 2, 1] = 2 * (y * z + x * w); R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--fast", type=float, default=0.3)
    ap.add_argument("--range", default=None)
    ap.add_argument("parquet", nargs="+")
    a = ap.parse_args()
    for path in a.parquet:
        row = pq.read_table(path).to_pylist()[0]
        sides = list(row["hand_sides"])
        contact = {s: np.asarray(c, dtype=bool) for s, c in zip(sides, row["hand_contact_active"])}
        ee = np.asarray(row["ee_pose_w"], dtype=float)
        ee_names = list(row["ee_link_names"])
        op = np.asarray(row["object_body_position"], dtype=float)[:, 0]
        T = len(op)
        firsts = {s: int(np.argmax(c)) for s, c in contact.items() if c.any()}
        if not firsts:
            print(f"{path}：没有接触标记")
            continue
        side = min(firsts, key=firsts.get)
        k = ee_names.index(f"{side}_hand_palm_link")
        pp, pR = ee[:, k, :3], quat_mat(ee[:, k, 3:], "wxyz")
        rel = np.einsum("nji,nj->ni", pR, op - pp)
        slip = np.r_[0.0, np.linalg.norm(np.diff(rel, axis=0), axis=1)] * a.fps
        vo = np.r_[0.0, np.linalg.norm(np.diff(op, axis=0), axis=1)] * a.fps
        vp = np.r_[0.0, np.linalg.norm(np.diff(pp, axis=0), axis=1)] * a.fps
        c = contact[side]
        cs = slip[c]
        print(f"{path}：{T} 帧；手 {side}，接触 {int(c.sum())} 帧（首次 {firsts[side]}）；接触帧滑移速度 中位 {np.median(cs):.3f} m/s、95% {np.percentile(cs, 95):.3f} m/s、"
              f"> {a.fast} m/s 的 {int((cs > a.fast).sum())} 帧；接触帧物体速度中位 {np.median(vo[c]):.3f} m/s、最大 {vo[c].max():.3f} m/s")
        m = np.r_[False, c, False]
        edges = np.flatnonzero(m[1:] != m[:-1])
        for s0_, s1_ in zip(edges[::2], edges[1::2]):
            if s1_ - s0_ < 15:
                continue
            dev = np.linalg.norm(rel[s0_:s1_] - np.median(rel[s0_:s1_], axis=0), axis=1)
            print(f"  接触段 {s0_}–{s1_}（{s1_ - s0_} 帧）：rel 偏离段中位 中位 {np.median(dev):.3f} m、95% {np.percentile(dev, 95):.3f} m、最大 {dev.max():.3f} m")
        if a.range:
            s0, s1, st = (int(x) for x in a.range.split(":"))
            t0 = firsts[side]
            print("  帧\t接触\t物体 m/s\t手掌 m/s\t滑移 m/s\trel 相对接触起点 m")
            for f in range(s0, min(s1, T), st):
                print(f"  {f}\t{int(c[f])}\t{vo[f]:.2f}\t{vp[f]:.2f}\t{slip[f]:.2f}\t{np.linalg.norm(rel[f] - rel[t0]):.3f}")
    print("REF_SLIP_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
