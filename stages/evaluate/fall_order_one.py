"""单条 run 的摔倒时序：调 fall_order.py 的 run_one 与 describe，门与判法都不改（计划 agent-doc/plan-first-place-2026-10-14.md 第 2 节选项丙）。

fall_order.py 的 main() 只收 10/7 试验的清单（每集 base42、base43、var42、var43 四条 run），跟丢的集只有单条 run，所以另写这个驱动。
只读这条 run 自己的文件：runs/flash_chord/<run>/eval_dump/eval_<步数>_w<世界数>.{json,npz}（fc_eval_dump.sh 的输出）与原评测
runs/flash_chord/<run>/eval_<步数>_w<世界数>.json；Tier 2 的 run 只读 Tier 2 的东西。
物体朝向阈值的核对：run 名带 -r3a- 或 -r3o- 的按变体 0.6（fall_order.is_variant），其余按官方 1.5；--ori 0.6 或 1.5 对全部 run 改用给定值。
run_one 的门（误差名、数组形状、npz 与新评 JSON 的 MP-SR、checkpoint sha256、MP-SR 复现、朝向阈值）不过就打印问题、这条不分类；
文件缺、读不了或 JSON 缺键也按门不过记，接着处理下一条。有一条不过退出码 1，全部过退出码 0。MPPE 差只报，超过参考容差不算门不过。
一次只收同一档的 run；t2、t3 先装 Tier 1 禁读（libs/v2d_infra/tier_guard.py，同 onset_terms.py；r3o 计划审查第 6 条）。
用法：python stages/evaluate/fall_order_one.py <run> [<run> ...] [--steps 349999104] [--worlds 4096] [--root .] [--ori 0.6|1.5]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import fall_order as fo  # noqa: E402
from libs.v2d_infra import tier_guard  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--steps", type=int, default=349999104)
    ap.add_argument("--worlds", type=int, default=4096)
    ap.add_argument("--root", default=".")
    ap.add_argument("--ori", type=float, choices=(0.6, 1.5), default=None)
    a = ap.parse_args(argv)
    tiers = {r.split("-", 1)[0] for r in a.runs}
    if len(tiers) != 1 or not tiers <= {"t1", "t2", "t3"}:
        print(f"fall_order_one: 一次只收同一档的 run，这次是 {sorted(tiers)}", file=sys.stderr)
        return 2
    if tiers != {"t1"}:
        tier_guard.forbid_tier1()
    bad = 0
    for run in a.runs:
        is_var = a.ori == 0.6 if a.ori is not None else fo.is_variant(run)
        try:
            r = fo.run_one(a.root, run, a.steps, a.worlds, is_var)
        except (OSError, KeyError, ValueError) as e:
            print(f"== {run}：门不过，不分类：读不了评测文件或缺键（{type(e).__name__}: {e}）")
            bad += 1
            continue
        print(f"== {run}（物体朝向阈值按 {0.6 if is_var else 1.5} 核）：新评 MP-SR {r['mp_new']:.4f}，与原评测差 {r['d_mp']:+.4f}；"
              f"MPPE 差 {r['d_mppe']:+.2f} cm（参考容差 {r['tol']:.2f}，只报）")
        if r["problems"]:
            print("  门不过，不分类：" + "；".join(r["problems"]))
            bad += 1
            continue
        fo.describe([r], run)
    print("FALL_ORDER_ONE_DONE" if not bad else f"FALL_ORDER_ONE_GATE_FAILED {bad} 条")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
