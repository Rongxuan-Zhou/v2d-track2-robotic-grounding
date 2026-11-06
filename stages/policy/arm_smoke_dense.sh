#!/usr/bin/env bash
# 冒烟密存档一条龙（mf4090 上、仓库根执行，放 tmux；Mac 只负责启动）。推广自 arm_t2sd.sh（计划 agent-doc/plan-t2sd-2026-10-19.md），
# 用于 agent-doc/plan-t2rp-2026-10-19.md：一段里有几条冒烟时，停链标记要等最后一条冒烟开始之后才建（标记会让后面的冒烟也不开）。
# 1. 配置核对、预检（落 results/<名>/）；2. 起链预演，DRYRUN_OK 才起；3. 起链（arm_var_chain.sh <段名>）；
# 4. 训练链日志出现冒烟队列最后一条的开始行之后，建 runs/.chain-stop-<段名>_train_mf4090，正式训练不开；
# 5. 等训练链写出结束行；6. 起评测链（附加表 stages/evaluate/<名>_mf4090.tsv）与导出门（清单 stages/evaluate/exports/cand_t2_<名>_mf4090.tsv）。
# 用法：stages/policy/arm_smoke_dense.sh <段名，如 t2mrp> <名，如 t2rp>   日志 runs/logs/arm_<名>_mf4090.log，最后一行 ARM_DENSE_DONE 或 ARM_DENSE_FAILED <原因>。
set -uo pipefail
cd "$(dirname "$0")/../.."
SEG=${1:?段名}; NM=${2:?名}
# 10/21 推广到 Tier 3（计划 agent-doc/plan-t3static-2026-10-21.md）：段名 t3p 开头也落 mf4090；名的档位与段名一致，冒烟前缀、导出清单跟着档位走
[[ ( $SEG =~ ^t2m[a-z0-9]+$ && $NM =~ ^t2[a-z0-9]+$ ) || ( $SEG =~ ^t3p[a-z0-9]+$ && $NM =~ ^t3[a-z0-9]+$ ) ]] || { echo "arm_smoke_dense: 段名要以 t2m 或 t3p 开头（落 mf4090），名的档位要与段名相同" >&2; exit 2; }
TIER=${SEG:1:1}
LOG="runs/logs/arm_${NM}_mf4090.log"
mkdir -p runs/logs
exec >> "$LOG" 2>&1
say() { echo "=== $(date '+%F %T %Z') $*"; }
fail() { say "ARM_DENSE_FAILED $*"; exit 1; }
stop_fail() { touch "runs/.chain-stop-${SEG}_train_mf4090"; say "建了停链标记（失败分支，正式训练不开）"; fail "$@"; }
Q="stages/policy/queues/mf4090_${SEG}"
T="runs/logs/${SEG}_train_mf4090.log"
SRC="$HOME/v2d/fc_src/fc_8c778f9"
EXTRA="stages/evaluate/${NM}_mf4090.tsv"; LIST="stages/evaluate/exports/cand_t${TIER}_${NM}_mf4090.tsv"
for f in "${Q}_smoke.txt" "${Q}.txt" "${Q}.expect.tsv" "$EXTRA" "$LIST"; do [ -s "$f" ] || fail "没有 $f"; done
LAST="$(grep -v '^#' "${Q}_smoke.txt" | awk '{for (i = 1; i <= NF; i++) if ($i ~ /^smoke-t[0-9]-/) { r = $i } } END { print r }')"
[ -n "$LAST" ] || fail "冒烟队列里找不到 smoke-t${TIER}- 开头的 run"
# 复核第 1 条：已有 run 目录时冒烟会被跳过、开始行不出现，标记就建不上；这种情况一律不起
for r in $(cut -f1,2 "${Q}.expect.tsv" | tail -n +2 | tr '\t' ' '); do
  [ ! -e "runs/flash_chord/$r" ] || fail "runs/flash_chord/$r 已存在，不起（冒烟会被跳过，停链标记建不上）"
done
say "开始，来历 $(cat .v2d-source-commit 2>/dev/null || echo unknown)；最后一条冒烟 ${LAST}"
scripts/dev/record "${NM}/cfgcheck_mf4090" stages/policy/fc_config_check.sh "${Q}_smoke.txt" "${Q}.txt" > /dev/null 2>&1 || fail "配置核对没过（results/${NM}/cfgcheck_mf4090.txt.failed）"
scripts/dev/record "${NM}/precheck_mf4090" stages/policy/precheck_expect.sh "${Q}.expect.tsv" "${Q}_smoke.txt" "${Q}.txt" > /dev/null 2>&1 || fail "预检没过（results/${NM}/precheck_mf4090.txt.failed）"
dry="$(V2D_ARM_DRYRUN=1 bash stages/policy/arm_var_chain.sh "$SEG" 2>&1)"; tail -n 3 <<< "$dry"
[ "$(tail -n 1 <<< "$dry")" = DRYRUN_OK ] || fail "起链预演没过"
arm="$(bash stages/policy/arm_var_chain.sh "$SEG" 2>&1)"; tail -n 3 <<< "$arm"
[ "$(tail -n 1 <<< "$arm")" = ARMED ] || fail "起链没成"
for _ in $(seq 1 720); do
  grep -qE "冒烟 ${LAST}\$" "$T" 2>/dev/null && break
  tail -n 1 "$T" 2>/dev/null | grep -qE "DAY2_TRAIN_(DONE|FAILED)" && stop_fail "最后一条冒烟开始之前训练链就结束了"
  sleep 5
done
grep -qE "冒烟 ${LAST}\$" "$T" || stop_fail "60 分钟内没等到最后一条冒烟开始"
touch "runs/.chain-stop-${SEG}_train_mf4090"
say "最后一条冒烟已开始，建了停链标记 runs/.chain-stop-${SEG}_train_mf4090"
for _ in $(seq 1 360); do tail -n 1 "$T" | grep -qE "DAY2_TRAIN_(DONE|FAILED)" && break; sleep 10; done
say "训练链最后一行：$(tail -n 1 "$T" | sed 's/DAY2_TRAIN_/训练链结束：/')"
TMUX= tmux new-session -d -s "v2d-${NM}-eval" -c "$PWD" "V2D_FC_SRC=$SRC V2D_GPU_WAIT_S=7200 stages/evaluate/day2_eval_chain.sh - $EXTRA ${NM}_eval_mf4090"
sleep 2
TMUX= tmux new-session -d -s "v2d-${NM}-export" -c "$PWD" "V2D_FC_SRC=$SRC V2D_FALLBACK_MAX=0 stages/evaluate/export_after_eval.sh runs/logs/${NM}_eval_mf4090.log $LIST cand_t${TIER}_${NM}_mf4090"
sleep 5
tmux ls 2>/dev/null | grep "$NM"
say ARM_DENSE_DONE
