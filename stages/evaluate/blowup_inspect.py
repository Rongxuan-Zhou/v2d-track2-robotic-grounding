"""看取证 rollout（blowup_rollout.py 的 npz）里崩掉的世界在崩之前几帧的物体与手掌位置（要 numpy；ml40 宿主机没有，放 FlashCHORD 容器里不带 GPU 跑）。

计划 agent-doc/plan-ep7-eval-and-guards-2026-10-02.md 的 A 部分（A-4 只描述）。对每个崩掉的世界（first_bad_frame >= 0），打印第一次出问题之前 --before 帧到之后 --after 帧：
  物体位置与它离参考物体位置的距离、物体底部相对桌面的高度（--table-top，按参考的世界系；不给就不报）、两只手掌的位置与它们离物体的距离、
  评测器自己记的逐步跟踪误差（tracking_error，按名字）。另报这个世界第一次出问题的那一帧，物体与手掌里哪一个先出问题、是 NaN、inf 还是超远值。
只读 npz，不写任何东西。

用法：python stages/evaluate/blowup_inspect.py <npz> [--before 12] [--after 2] [--max-worlds 5] [--table-top 0.7821]
"""
import argparse

import numpy as np


def kind(v: np.ndarray, far: float) -> str:
    if np.isnan(v).any():
        return "NaN"
    if np.isinf(v).any():
        return "inf"
    if (np.abs(v) > far).any():
        return f"超过 {far:g} m"
    return "有限"


def fmt(v) -> str:
    v = np.asarray(v, np.float64).ravel()
    return "(" + ", ".join("nan" if np.isnan(x) else ("inf" if np.isposinf(x) else ("-inf" if np.isneginf(x) else f"{x:.4g}")) for x in v) + ")"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--before", type=int, default=12)
    ap.add_argument("--after", type=int, default=2)
    ap.add_argument("--max-worlds", type=int, default=5)
    ap.add_argument("--table-top", type=float, default=None, help="桌面顶面的 z（参考的世界系）；ep7 的桌面是 0.7771 + 0.005")
    a = ap.parse_args()
    z = np.load(a.npz)
    obj, palm, first = z["object_pos"], z["palm_pos"], z["first_bad_frame"]
    ref = z["reference_pose_w"]
    far = float(z["far_m"])
    names = [str(n) for n in z["tracking_error_names"]]
    te = z["tracking_error"]
    sides = [str(s) for s in z["palm_sides"]]
    T, W = obj.shape[:2]
    bad = np.flatnonzero(first >= 0)
    print(f"{a.npz}：T={T} W={W}，崩掉的世界 {len(bad)} 个；评测器记的非有限世界 {z['evaluator_non_finite_worlds'].tolist()}；"
          f"被判终止的世界 {int(np.asarray(z['terminated']).sum())} 个；评分报错：{str(z['score_error']) or '无'}")
    print(f"tracking_error 的列：{names}；手掌顺序：{sides}")
    for w in bad[: a.max_worlds]:
        f0 = int(first[w])
        o_bad = kind(obj[f0, w], far)
        p_bad = [kind(palm[f0, w, h], far) for h in range(palm.shape[2])]
        print(f"\n== 世界 {w}：第一次出问题在第 {f0} 帧（共 {T} 帧）；那一帧物体 {o_bad}，手掌 {dict(zip(sides, p_bad))}")
        prev = [t for t in range(f0) if not np.isfinite(obj[t, w]).all() or not np.isfinite(palm[t, w]).all()]
        if prev:
            print(f"  （之前已有非有限帧：{prev[:5]}…）")
        for t in range(max(0, f0 - a.before), min(T, f0 + a.after + 1)):
            o = obj[t, w, 0].astype(np.float64)
            r = ref[t, 0, :3].astype(np.float64)
            d_ref = float(np.linalg.norm(o - r)) if np.isfinite(o).all() else float("nan")
            hp = []
            for h in range(palm.shape[2]):
                p = palm[t, w, h].astype(np.float64)
                d = float(np.linalg.norm(p - o)) if np.isfinite(p).all() and np.isfinite(o).all() else float("nan")
                hp.append(f"{sides[h]} {fmt(p)} 离物体 {d:.3f} m")
            height = f"，物体中心离桌面 {o[2] - a.table_top:+.3f} m" if a.table_top is not None and np.isfinite(o[2]) else ""
            errs = "，".join(f"{n} {float(te[t, w, i]):.4g}" for i, n in enumerate(names))
            mark = " ←" if t == f0 else ""
            print(f"  第 {t} 帧{mark}：物体 {fmt(o)}，离参考 {d_ref:.3f} m{height}；{'；'.join(hp)}；{errs}")
    if len(bad) > a.max_worlds:
        print(f"\n（另有 {len(bad) - a.max_worlds} 个世界没打印）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
