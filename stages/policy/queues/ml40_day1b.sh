#!/usr/bin/env bash
# Day 1b（在 ml40 上执行，放 tmux，不依赖 Mac 在线）。计划见 agent-doc/plan-day1b-2026-09-27.md（第 3 版）。
#   1. 镜像指纹缓存挪开，让冒烟重新计算 base11 与依赖 freeze
#   2. 冒烟（queues/ml40_day1b_smoke.txt，W&B 记到 v2d-selftest）→ 冒烟判据全过才继续
#   3. 正式满配 snack_box（queues/ml40_day1b.txt，V2D_MAX_RETRY=0：失败即停，不自动续训）
#   4. 4096 世界评测与输入清单（经 scripts/dev/record）
# 用法（仓库根）：tmux new-session -d -s v2d-day1b stages/policy/queues/ml40_day1b.sh
# 日志 runs/logs/ml40_day1b.log，最后一行 DAY1B_DONE 或 DAY1B_FAILED。
# 正式 run 已有 checkpoint 却没有 DONE（上次中途失败）时拒绝继续：续训不恢复 replay，要从 0 重来就先移走该目录；
# 确要续训就设 V2D_DAY1B_ALLOW_RESUME=1，按登记该 run 标为偏离。
set -Eeuo pipefail   # -E：函数里的失败也触发 ERR trap
cd "$(dirname "$0")/../../.."
mkdir -p runs/logs
exec >> runs/logs/ml40_day1b.log 2>&1
say() { echo "=== $(date '+%F %T %Z') $*"; }
fail() { say "$*"; say DAY1B_FAILED; exit 1; }
trap 'say "DAY1B_FAILED（第 $LINENO 行命令失败）"' ERR
SMOKE=smoke-snackbox-w2048-cap10m-ml40; RUN=ctrl-snackbox-w2048-cap10m-ml40b   # 第 2 次尝试，见 queues/ml40_day1b.txt
DAY1_BASE11=e2934912858820b8; DAY1_FREEZE=df0cf35711a68fb3   # Day 1 满配 ep20 的 run_record.jsonl
PEAK_MAX_MIB=44000                                           # 沿用 Day 1 冒烟判据
say "start, source $(cat .v2d-source-commit 2>/dev/null || echo unknown)"

# 0. 正式 run 不得处于“有 checkpoint、没有 DONE”的状态（否则 fc_run 会悄悄续训）
R=runs/flash_chord/$RUN
if [ ! -f "$R/DONE" ] && compgen -G "$R/state_*.safetensors" > /dev/null && [ "${V2D_DAY1B_ALLOW_RESUME:-0}" != 1 ]; then
  fail "$R 有 checkpoint 却没有 DONE：从 0 重来就先移走该目录；确要续训设 V2D_DAY1B_ALLOW_RESUME=1（标为偏离）"
fi

# 1. 指纹缓存挪开（Day 1 那份只挪一次，重跑不覆盖）
IMG="$(docker image inspect -f '{{.Id}}' flash_chord:latest)"
C="runs/cache/image-${IMG#sha256:}.json"
if [ -f "$C" ] && [ ! -f "$C.day1" ]; then mv "$C" "$C.day1"; say "指纹缓存挪到 $C.day1，冒烟时重新计算"; fi

# 2. 冒烟与判据（检查本身在 Day 1 已知通过的冒烟日志上核对过，见 results/ml40/day1/smoke_check_day1_known_good.txt）
S=runs/flash_chord/$SMOKE
[ -f "$S/DONE" ] || stages/policy/fc_queue.sh stages/policy/queues/ml40_day1b_smoke.txt
stages/policy/smoke_check.sh "$SMOKE" 4096000 "$PEAK_MAX_MIB" "$DAY1_BASE11" "$DAY1_FREEZE" || fail "冒烟判据没过（原因见上一行）"

# 3. 正式训练
stages/policy/fc_queue.sh stages/policy/queues/ml40_day1b.txt
[ -f "$R/DONE" ] || fail "正式训练没有 DONE"
segs="$(grep -c '"event": "start"' "$R/run_record.jsonl")"
if [ "$segs" = 1 ]; then say "正式训练一段跑完"; else say "注意：正式训练有 $segs 段，按登记标为偏离（续训不恢复 replay）"; fi

# 4. 评测与输入清单
scripts/dev/record "day1/eval_$RUN" stages/policy/fc_eval.sh "$RUN" 349999104
scripts/dev/record "day1/policy_inputs_${RUN}_on_ml40" python3 stages/policy/policy_inputs.py "$R/policy_349999104.safetensors"
say DAY1B_DONE
