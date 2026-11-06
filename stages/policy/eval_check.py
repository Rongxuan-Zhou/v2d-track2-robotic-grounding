"""核一份 FlashCHORD 评测 JSON（scripts/evaluate_policy.py 的 metrics_output）：non_finite_worlds 为空，排行榜的四个指标都有限。只用标准库。

第 3 步配方试验闸 G1 用它核“8c778f9 评测器能读变体 A 的 checkpoint”（计划 agent-doc/plan-step3-recipe-trial-2026-10-04.md 第 2 版第 7 节）。
用法：python3 stages/policy/eval_check.py <评测 JSON>；不过时退出码 1。
"""
import json
import math
import sys

KEYS = ("maniptrans_sr", "spider_sr_uncentered", "add_auc", "mppe_cm")


def main(path: str) -> int:
    d = json.load(open(path))
    m = d.get("metrics") or {}
    vals = {k: m.get(k) for k in KEYS}
    finite = all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals.values())
    nf = d.get("non_finite_worlds")
    print(f"{path}：checkpoint_sha256 {str(d.get('checkpoint_sha256'))[:16]}…，world_count {d.get('world_count')}，non_finite_worlds {nf}")
    print("  " + "，".join(f"{k} {v}" for k, v in vals.items()))
    ok = finite and nf == []
    print("EVAL_CHECK " + ("OK" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
