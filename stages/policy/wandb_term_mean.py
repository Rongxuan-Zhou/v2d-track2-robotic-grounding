"""某个 W&B 指标在指定课程段里的均值（训练机上经 scripts/dev/in-flash-chord 在容器里跑，容器里有 wandb；读法复用 wandb_local_history.py 的 rows）。
用途：计划 agent-doc/plan-r3o-2026-10-15.md 2.2 的“只报”项——每条 r3o run 第 7–9 段（curriculum/stage_index 6、7、8，VOC 为 0 的三段）日志里新项
reward_components/object_orientation_weighted_mean_per_env_step 的均值；跟丢集全部低于 0.05 记“新项没有提供信号”，与“没用”分开写。
每个文件报：所选各段的行数与均值（段内全部行），所选段合起来的均值。
用法：python stages/policy/wandb_term_mean.py --key <指标> [--stages 6,7,8] <run-*.wandb>...；最后一行 WANDB_TERM_MEAN_DONE。
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wandb_local_history import numeric, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--key", required=True)
    ap.add_argument("--stages", default="6,7,8")
    ap.add_argument("files", nargs="+", type=Path)
    a = ap.parse_args()
    want = {int(s) for s in a.stages.split(",")}
    for f in a.files:
        per, allv = {}, []
        for r in rows(f):
            st, v = r.get("curriculum/stage_index"), r.get(a.key)
            if numeric(st) and int(st) in want and numeric(v) and math.isfinite(v):
                per.setdefault(int(st), []).append(float(v))
                allv.append(float(v))
        parts = "；".join(f"段 {s + 1}（stage_index {s}）{len(per.get(s, []))} 行，均值 "
                         f"{sum(per[s]) / len(per[s]):.4f}" if per.get(s) else f"段 {s + 1} 没有行" for s in sorted(want))
        mean = f"{sum(allv) / len(allv):.4f}" if allv else "—"
        print(f"{f.parent.name}：{parts}；合起来 {mean}")
    print("WANDB_TERM_MEAN_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
