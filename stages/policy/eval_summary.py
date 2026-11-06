"""打印 FlashCHORD 评测 JSON（scripts/evaluate_policy.py 的 metrics_output）的摘要（只用标准库）。

fc_eval.sh 评测完调用它；也可以直接用于任何一份评测 JSON，例如 fc_eval.sh 出现之前、
9/25 冒烟评测留下的 runs/flash_chord/smoke_t1_ep20_w256/policy_3072000.safetensors.metrics.json。

口径（harness.py）：JSON 里的 chord_sr 被替换为 reference_tracking.success_rate，即 ReconBody 任务原生判据
（pelvis / palm / object 三组门槛，palm 与 object 在 reset_freeze_steps 之后才计）；metrics 里的
termination_cause_fraction 来自旧的物体中心口径（palm 当 wrist、门槛从第 0 帧起算）。两者不可混读，
失败原因以 reference_tracking.failure_cause_fraction 为准。

用法：python3 stages/policy/eval_summary.py <评测 JSON>...
"""
import json
import sys

KEYS = ["add_auc", "chord_sr", "maniptrans_sr", "spider_sr_uncentered", "mppe_cm", "mean_add_m",
        "object_position_error_m", "object_orientation_error_deg", "termination_cause_fraction"]


def summarize(path: str) -> bool:
    with open(path) as f:
        d = json.load(f)
    m = d.get("metrics", {})
    print(f"== {path}")
    print(f"  checkpoint {d.get('checkpoint')}  sha256 {str(d.get('checkpoint_sha256'))[:16]}…")
    print(f"  environment_steps {d.get('environment_steps')}  world_count {d.get('world_count')}  step_count {d.get('step_count')}"
          f"  start_frame {d.get('start_frame')}  reset_mode {d.get('reset_mode')}  non_finite_worlds {d.get('non_finite_worlds')}")
    for k in KEYS:
        if k in m:
            label = f"{k} (legacy object-centric)" if k == "termination_cause_fraction" else k
            print(f"  {label:<46} {m[k]}")
    rt = d.get("reference_tracking") or {}
    print(f"  {'reference_tracking.success_rate (= chord_sr)':<46} {rt.get('success_rate')}")
    print(f"  {'reference_tracking.failure_cause_fraction':<46} {rt.get('failure_cause_fraction')}")
    print(f"  {'reference_tracking.reference_end_fraction':<46} {rt.get('reference_end_fraction')}")
    missing = [k for k in KEYS if k not in m] + ([] if rt else ["reference_tracking"])
    if missing:
        print(f"  MISSING {', '.join(missing)}")
    # 机器可读的一行，供 stages/evaluate/verdict.py 按判据比较
    compact = {k: m.get(k) for k in KEYS if k != "termination_cause_fraction"}
    compact.update(environment_steps=d.get("environment_steps"), world_count=d.get("world_count"),
                   checkpoint_sha256=d.get("checkpoint_sha256"), reference_tracking=rt)
    # 2026-09-28 只加不改：配对核对用（Day 2 的比较脚本核对步数、parquet 与 checkpoint 是否对得上）
    compact.update(step_count=d.get("step_count"), parquet=d.get("parquet"), checkpoint=d.get("checkpoint"))
    print("  summary_json " + json.dumps(compact, sort_keys=True))
    return not missing


if __name__ == "__main__":
    ok = [summarize(p) for p in sys.argv[1:]]
    sys.exit(0 if ok and all(ok) else 1)
