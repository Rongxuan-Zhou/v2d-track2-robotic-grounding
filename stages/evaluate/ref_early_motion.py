"""事后诊断（计划外，只报）：参考物体在开头几帧自己动了多少，以及第一次接触在第几帧（容器里跑，要 numpy、pyarrow）。只读、不占 GPU。
起因：逐帧终止分析里 Tier 3 best_hm 的 6 集在冻结期刚结束（第 51 帧）所有世界都已越线，早于第一次接触；要分清是参考自己在动（重建问题），
还是仿真里的物体被碰或自己滚（计划 agent-doc/plan-frame-onset-2026-10-14.md 执行记录）。参考 parquet 取自 policy 内嵌配置（同 fail_onset.py）。
一次只收同一档；t2、t3 先装 Tier 1 禁读。
打印：每条 run 物体（第 0 个刚体）从第 0 帧到第 25、51、100 帧的位置位移（cm）与转角（度），第 0–51 帧内相邻帧转角的最大值（度），
参考第一次接触的帧（两侧任一侧）。
用法：python stages/evaluate/ref_early_motion.py <run> [<run> ...] [--steps 349999104] [--root .]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import fail_onset as fon  # noqa: E402
from libs.v2d_infra import tier_guard  # noqa: E402

FRAMES = (25, 51, 100)


def angle_deg(q0: np.ndarray, q: np.ndarray) -> np.ndarray:
    """两组单位四元数之间的转角（度），符号无关。"""
    d = np.abs(np.sum(q0 * q, axis=-1) / (np.linalg.norm(q0, axis=-1) * np.linalg.norm(q, axis=-1)))
    return np.degrees(2 * np.arccos(np.clip(d, 0.0, 1.0)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--steps", type=int, default=349999104)
    ap.add_argument("--root", default=".")
    a = ap.parse_args(argv)
    tiers = {r.removeprefix("smoke-").split("-", 1)[0] for r in a.runs}   # 冒烟 run 也收（计划 agent-doc/plan-t3g5-2026-10-21.md：用冒烟的 policy 提前看参考开头）
    if len(tiers) != 1 or not tiers <= {"t1", "t2", "t3"}:
        print(f"ref_early_motion: 一次只收同一档的 run，这次是 {sorted(tiers)}", file=sys.stderr)
        return 2
    if tiers != {"t1"}:
        tier_guard.forbid_tier1()
    import pyarrow.parquet as pq

    root = Path(a.root)
    for run in a.runs:
        path = fon.ref_parquet(root, run, a.steps)
        t = pq.read_table(path, columns=["object_body_names", "object_body_position", "object_body_wxyz"]).to_pydict()
        nb = len(t["object_body_names"][0])
        pos = np.asarray(t["object_body_position"][0], dtype=np.float64).reshape(-1, nb, 3)[:, 0]
        q = np.asarray(t["object_body_wxyz"][0], dtype=np.float64).reshape(-1, nb, 4)[:, 0]
        _, contact = fon.ref_contact(path)
        any_c = contact.any(axis=0)
        first = int(np.argmax(any_c)) if any_c.any() else -1
        parts = []
        for f in FRAMES:
            if f < len(pos):
                parts.append(f"第 {f} 帧 {np.linalg.norm(pos[f] - pos[0]) * 100:.1f} cm、{angle_deg(q[0], q[f]):.1f}°")
        step = angle_deg(q[:51], q[1:52]).max() if len(q) > 51 else float("nan")
        print(f"== {run}：从第 0 帧起 {'；'.join(parts)}；第 0–51 帧相邻帧转角最大 {step:.1f}°；参考第一次接触第 {first} 帧")
    print("REF_EARLY_MOTION_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
