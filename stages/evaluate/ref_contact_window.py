"""一份重定向参考在一段帧里的接触标记与物体位移（训练机上经 scripts/dev/in-flash-chord 在容器里跑，只用 CPU，只读参考 parquet；只报不判）。
起因：agent-doc/plan-t3g5-2026-10-21.md 10/21 08:4x 的诊断——ep16 导出那一局里仿真的瓶子第 93 帧就被碰动，参考的瓶子第 129 帧才动；
管线总览记着 ep16 第 105–125 帧手离瓶约 10 cm。这里看参考在这段里两只手的 hand_contact_active 与物体相对第 0 帧的平移、高度，
判断机器人是不是在参考没有接触标记的帧里就碰到了物体。
读的列：hand_sides、hand_contact_active（2×T）、object_body_position（T × 部件 × 3，取第 0 个部件）。
--summary：每份参考只打一行汇总——第一次接触帧、物体第一次平移 > 3 cm 的帧、第一次接触时物体已平移多少，以及全程“没有一只手在接触、物体却在动”
（相邻帧平移 > --speed-cm，默认 0.5 cm/帧）的帧数与这些帧里的平移合计。门槛只用于描述，不进任何方法。
用法（容器内）：python /repo/stages/evaluate/ref_contact_window.py [--summary] [--frames 80:150:5] <参考 parquet>...   最后一行 REF_CONTACT_WINDOW_DONE
"""
import argparse

import numpy as np
import pyarrow.parquet as pq


def segments(flags: np.ndarray) -> list[tuple[int, int]]:
    """布尔序列里为真的连续段（含两端的帧号）。"""
    out, start = [], None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        elif not f and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(flags) - 1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="0:200:10", help="起:止:步（止不含）")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--speed-cm", type=float, default=0.5)
    ap.add_argument("parquet", nargs="+")
    a = ap.parse_args()
    lo, hi, st = (int(x) for x in a.frames.split(":"))
    for path in a.parquet:
        t = pq.read_table(path, columns=["hand_sides", "hand_contact_active", "object_body_position"]).to_pydict()
        sides = [str(s) for s in t["hand_sides"][0]]
        c = np.asarray(t["hand_contact_active"][0], dtype=np.float64)
        pos = np.asarray(t["object_body_position"][0], dtype=np.float64)[:, 0, :]
        n = len(pos)
        if c.shape != (len(sides), n):
            raise SystemExit(f"{path}：hand_contact_active 形状 {c.shape} 与 {len(sides)} 只手、{n} 帧对不上")
        d = np.linalg.norm(pos - pos[0], axis=1)
        dz = pos[:, 2] - pos[0, 2]
        anyc = (c > 0.5).any(axis=0)
        step = np.r_[0.0, np.linalg.norm(np.diff(pos, axis=0), axis=1)]
        free = (~anyc) & (step > a.speed_cm / 100.0)
        if a.summary:
            fc = int(np.argmax(anyc)) if anyc.any() else None
            fm = next((i for i in range(n) if d[i] > 0.03), None)
            at = f"{d[fc]:.3f}" if fc is not None else "-"
            print(f"{path}\t帧 {n}\t首次接触 {fc}\t物体首次平移>3cm {fm}\t首次接触时已平移 m {at}\t无接触而在动的帧 {int(free.sum())}\t这些帧的平移合计 m {step[free].sum():.3f}")
            continue
        print(f"{path}：{n} 帧；手 {sides}")
        for k, s in enumerate(sides):
            segs = segments(c[k] > 0.5)
            print(f"  {s} 的接触段：{' '.join(f'{x}–{y}' for x, y in segs) or '无'}")
        first_move = next((i for i in range(n) if d[i] > 0.03), None)
        print(f"  参考物体第一次平移 > 3 cm：第 {first_move} 帧")
        print("  帧\t" + "\t".join(f"{s}接触" for s in sides) + "\t物体平移 m\t高度变化 m")
        for i in range(max(0, lo), min(hi, n), st):
            print(f"  {i}\t" + "\t".join(str(int(c[k, i])) for k in range(len(sides))) + f"\t{d[i]:.3f}\t{dz[i]:+.3f}")
    print("REF_CONTACT_WINDOW_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
