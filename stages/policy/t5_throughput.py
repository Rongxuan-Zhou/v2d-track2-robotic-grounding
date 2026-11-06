"""T5 冒烟的吞吐判据（计划 agent-doc/plan-ep7-eval-and-guards-2026-10-02.md 第 5 版的 B.4，T5 跑之前登记，b96a530）。在 FlashCHORD 容器里跑（要 wandb）：
  scripts/dev/in-flash-chord "python /repo/stages/policy/t5_throughput.py <新冒烟的 run-*.wandb> <对照冒烟的 run-*.wandb>"
稳态行指环境步数（W&B 的 _step）不小于 1,024,000 的 history 行；两条各取这些行 throughput/env_steps_per_second 的中位数，
新的不低于对照的 95% 为过。另报两条的行数、最小值、最大值。最后一行 T5_THROUGHPUT PASS|FAIL <比值>。
"""
import argparse
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from wandb_local_history import rows

KEY = "throughput/env_steps_per_second"
STEADY_FROM = 1_024_000
RATIO = 0.95


def steady(path: Path) -> list[float]:
    values = []
    for row in rows(path):
        step, value = row.get("_step"), row.get(KEY)
        if isinstance(step, (int, float)) and step >= STEADY_FROM and isinstance(value, (int, float)) and math.isfinite(value):
            values.append(float(value))
    return values


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("new", type=Path)
    ap.add_argument("reference", type=Path)
    a = ap.parse_args()
    medians = {}
    for label, path in (("新", a.new), ("对照", a.reference)):
        values = steady(path)
        if not values:
            print(f"{label} {path.name}：没有环境步数 ≥ {STEADY_FROM:,} 的行")
            return 1
        medians[label] = statistics.median(values)
        print(f"{label} {path.name}：稳态行 {len(values)} 行，{KEY} 中位数 {medians[label]:,.0f}，最小 {min(values):,.0f}，最大 {max(values):,.0f}")
    ratio = medians["新"] / medians["对照"]
    verdict = "PASS" if ratio >= RATIO else "FAIL"
    print(f"新 / 对照 = {ratio:.4f}（判据 ≥ {RATIO}）")
    print(f"T5_THROUGHPUT {verdict} {ratio:.4f}")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
