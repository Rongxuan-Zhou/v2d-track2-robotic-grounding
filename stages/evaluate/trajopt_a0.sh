#!/usr/bin/env bash
# trajopt A0（计划 agent-doc/plan-trajopt-2026-10-26.md 第 2 版第 1、2 节）：零残差 checkpoint 按真实时长与逐帧回放各评一次 4096 世界，读骨盆失败比例。
# mf4090 上、仓库根执行，放 tmux。只评测，不训练、不导出、不提交。GPU 上有别的进程就不起（不停任何进程）。
# 用法：bash stages/evaluate/trajopt_a0.sh   日志 runs/logs/trajopt_a0_mf4090.log，结果 results/trajopt/a0_*.txt，最后一行 TRAJOPT_A0_DONE 或 TRAJOPT_A0_FAILED <原因>
set -uo pipefail
cd "$(dirname "$0")/../.."
LOG=runs/logs/trajopt_a0_mf4090.log; mkdir -p runs/logs runs/trajopt results/trajopt
exec >> "$LOG" 2>&1
say() { echo "=== $(date '+%F %T %Z') $*"; }
SRC="$HOME/v2d/fc_src/fc_8c778f9"
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || { say "TRAJOPT_A0_FAILED GPU 上有别的进程，不起"; exit 1; }
say "开始，来历 $(cat .v2d-source-commit 2>/dev/null)"
for ep in 4 25 1; do
  pol="runs/flash_chord/smoke-t2-ep${ep}-w2048-cap4m-r3a-sd-mf4090/policy_409600.safetensors"
  [ -f "$pol" ] || { say "TRAJOPT_A0_FAILED 没有 $pol"; exit 1; }
  for pb in frame real; do
    tol=""; [ "$pb" = real ] && tol="--palm-position-tol 0.015 --palm-orientation-tol 0.086"
    out="runs/trajopt/a0_ep${ep}_${pb}.json"
    V2D_GPU=1 V2D_FC_SRC=$SRC scripts/dev/record "trajopt/a0_ep${ep}_${pb}" scripts/dev/in-flash-chord \
      "cd /workspace && python /repo/stages/evaluate/eval_playback_override.py /repo/$pol /repo/$out --playback $pb $tol --worlds 4096" > /dev/null 2>&1 \
      || { say "ep${ep} ${pb} 评测失败（results/trajopt/a0_ep${ep}_${pb}.txt.failed）"; continue; }
    python3 - "$out" "$ep" "$pb" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); m = {**d, **d.get("metrics", {})}
fc = (m.get("reference_tracking") or {}).get("failure_cause_fraction") or m.get("failure_cause_fraction") or {}
print(f"ep{sys.argv[2]} {sys.argv[3]}: step_count {m.get('step_count')} pelvis {fc.get('pelvis')} object {fc.get('object')} palm {fc.get('palm')} mppe_cm {m.get('mppe_cm')}")
PY
  done
done
say TRAJOPT_A0_DONE
