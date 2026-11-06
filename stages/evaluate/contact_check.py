"""一次性核对接触的两种取法是否逐帧相同（容器里跑，要 numpy、pyarrow）：fail_onset.py 直接读参考 parquet 的 hand_contact_active，
diag_rollout.py 存的是评测器参考对象的 contact_active(side)（按 start_frame 切片）。计划 agent-doc/plan-frame-onset-2026-10-14.md 第 8 节第 8 条。
两边按侧名配对，逐帧比；全部相同打 CONTACT_SAME，否则打不同的帧数与前几帧并以退出码 1 结束。只读同一条 run 的文件（Tier 1 的 run 只读 Tier 1）。
用法：python stages/evaluate/contact_check.py <diag_rollout 的 npz> <run> [--steps 349999104] [--root .]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import fail_onset as fon  # noqa: E402
from libs.v2d_infra import tier_guard  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("run")
    ap.add_argument("--steps", type=int, default=349999104)
    ap.add_argument("--root", default=".")
    a = ap.parse_args(argv)
    if not a.run.startswith("t1-"):
        tier_guard.forbid_tier1()   # Tier 2、Tier 3 的 run 不许读 Tier 1（代码审查第 13 条）
    z = np.load(a.npz, allow_pickle=False)
    d_sides = [str(s) for s in z["contact_sides"]]
    d_c = np.asarray(z["contact_active"], dtype=bool)          # (T, 侧)
    path = fon.ref_parquet(Path(a.root), a.run, a.steps)
    p_sides, p_c = fon.ref_contact(path)                        # (侧, T)
    print(f"参考 {path}；diag 的侧 {d_sides}、形状 {d_c.shape}；parquet 的侧 {p_sides}、形状 {p_c.shape}")
    if sorted(d_sides) != sorted(p_sides) or d_c.shape[0] != p_c.shape[1]:
        print("CONTACT_DIFF 侧名或帧数对不上")
        return 1
    bad = 0
    for s in d_sides:
        x, y = d_c[:, d_sides.index(s)], p_c[p_sides.index(s)]
        diff = np.flatnonzero(x != y)
        print(f"  {s}：diag 接触 {int(x.sum())} 帧，parquet 接触 {int(y.sum())} 帧，不同 {len(diff)} 帧{('，前几帧 ' + str(diff[:10].tolist())) if len(diff) else ''}")
        bad += len(diff)
    print("CONTACT_SAME" if bad == 0 else f"CONTACT_DIFF {bad}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
