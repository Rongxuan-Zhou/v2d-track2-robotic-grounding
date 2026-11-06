"""训练结束时各 run 的回合怎么结束（训练机上跑，宿主机 python3 即可，只读 W&B 本地汇总；只报不判）。
计划 agent-doc/plan-t12sat-2026-10-20.md 第 2 版（复核 MAJOR 2）：ep4 训练回合 87% 因物体终止、平均 69 步，ep0 85% 走到参考结尾。
读 runs/flash_chord/<run>/wandb/latest-run/files/wandb-summary.json（训练最后一次记录的值），每个 run 一行：回合平均长度、参考进度、
各终止原因的比例（object、palm、pelvis、reference_end）、|动作| 均值（action_scale 10 时 /10 即 |tanh| 均值）、记录时的步数。
用法：python3 stages/evaluate/train_term_table.py <ep>:<run> ...   最后一行 TRAIN_TERM_TABLE_DONE
"""
import json
import sys
from pathlib import Path

KEYS = [("episode/length_mean_per_completed_episode", "长度"), ("episode/reference_progress_mean_per_completed_episode", "进度"),
        ("termination/object_fraction_per_completed_episode", "物体终止"), ("termination/palm_fraction_per_completed_episode", "手掌终止"),
        ("termination/pelvis_fraction_per_completed_episode", "骨盆终止"), ("termination/reference_end_fraction_per_completed_episode", "走到结尾"),
        ("policy/action_abs_mean_per_env_action", "|动作|"), ("_step", "记录步")]


def main() -> int:
    print("ep\trun\t" + "\t".join(n for _, n in KEYS))
    for spec in sys.argv[1:]:
        ep, run = spec.split(":", 1)
        f = Path("runs/flash_chord") / run / "wandb" / "latest-run" / "files" / "wandb-summary.json"
        if not f.is_file():
            print(f"{ep}\t{run}\t（没有 {f}）")
            continue
        d = json.loads(f.read_text())
        vals = []
        for k, _ in KEYS:
            v = d.get(k)
            vals.append("—" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v)))
        print(f"{ep}\t{run}\t" + "\t".join(vals))
    print("TRAIN_TERM_TABLE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
