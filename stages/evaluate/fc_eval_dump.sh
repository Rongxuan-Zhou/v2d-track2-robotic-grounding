#!/usr/bin/env bash
# 照 stages/policy/fc_eval.sh 评一份 policy（8c778f9 副本、4096 世界），另把逐步、逐世界的跟踪误差存成 npz（stages/evaluate/eval_dump_errors.py）。
# 计划 agent-doc/plan-t1-fall-order-2026-10-07.md 第 3 节。训练机上、仓库根执行。
# 用法：V2D_FC_SRC=<8c778f9 副本> stages/evaluate/fc_eval_dump.sh <run> <步数>
#   V2D_EVAL_WORLDS=4096（默认）；小跑时给小的世界数
# 输出 runs/flash_chord/<run>/eval_dump/eval_<步数>_w<世界数>.{json,npz}（不进 git）；日志 runs/logs/<run>.eval_dump_<步数>_w<世界数>.log
# 已有完整结果（JSON 与 npz 都在）就直接打印，不重复评测。
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"; cd "$ROOT"
die() { echo "fc_eval_dump: $*" >&2; exit "${2:-2}"; }
RUN=${1:?用法：fc_eval_dump.sh <run> <步数>}; STEP=${2:?用法：fc_eval_dump.sh <run> <步数>}; W=${V2D_EVAL_WORLDS:-4096}
[[ $RUN =~ ^[a-z0-9][a-z0-9_-]{2,59}$ && $STEP =~ ^[0-9]+$ && $W =~ ^[1-9][0-9]*$ ]] || die "run、步数或世界数非法"
[ -n "${V2D_FC_SRC:-}" ] && head -1 "$V2D_FC_SRC/SOURCE_V2D.txt" 2>/dev/null | grep -q 8c778f905c \
  || die "V2D_FC_SRC 必须是 8c778f9 副本（现在是 ${V2D_FC_SRC:-未设}）"
O="runs/flash_chord/$RUN"; P="$O/policy_${STEP}.safetensors"; [ -f "$P" ] || die "没有 $P"
[ -f "$O/NONFINITE" ] && die "$RUN 有 NONFINITE，不评" 12
D="$O/eval_dump"; J="$D/eval_${STEP}_w${W}.json"; N="$D/eval_${STEP}_w${W}.npz"; LOG="runs/logs/$RUN.eval_dump_${STEP}_w${W}.log"
if [ -s "$J" ] && [ -s "$N" ]; then echo "fc_eval_dump: 已有结果 $J $N"; exit 0; fi
others="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" || die "nvidia-smi 出错，按 GPU 被占处理，拒绝评测" 6
others="$(printf '%s' "$others" | tr -d ' ')"
[ -z "$others" ] || die "GPU 上有别的进程（PID ${others//$'\n'/,}），拒绝评测" 6
mkdir -p "$D" runs/logs
echo "=== $(date '+%F %T %Z') fc_eval_dump $RUN policy_$STEP world_count=$W ===" >> "$LOG"
V2D_GPU=1 V2D_CONTAINER_NAME="fc-evaldump-$RUN" scripts/dev/in-flash-chord \
  "cd /workspace && V2D_DUMP_OUT=/repo/$N python /repo/stages/evaluate/eval_dump_errors.py $(printf '%q ' "evaluation.checkpoint=/repo/$P" \
   "evaluation.world_count=$W" "evaluation.metrics_output=/repo/$J" "hydra.run.dir=/repo/$O/hydra/evaldump-$STEP-w$W")" \
  >> "$LOG" 2>&1
rc=$?
[ "$rc" = 0 ] && [ -s "$J" ] && [ -s "$N" ] || die "评测失败（退出码 $rc），见 $LOG" 3
echo "fc_eval_dump: $RUN policy_$STEP（$W 世界）→ $J $N"
