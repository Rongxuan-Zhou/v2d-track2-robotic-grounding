#!/usr/bin/env bash
# 一段队列的起链（训练机上、仓库根执行，放 tmux）：配置核对 → 预检 → 起链预演（DRYRUN_OK 才起）→ 起链（arm_var_chain.sh <段名>，要 ARMED）。
# 推广自 arm_t12lift.sh 的 arm()；用于 agent-doc/plan-t2term-2026-10-20.md（段 t2tm）。任何一步不过就停，不起链；不停任何进程。
# 用法：stages/policy/arm_seg.sh <段名> <机器>   记录 results/<段名>/，日志 runs/logs/arm_<段名>_<机器>.log，最后一行 ARM_SEG_DONE 或 ARM_SEG_FAILED <原因>
set -uo pipefail
cd "$(dirname "$0")/../.."
SEG=${1:?段名}; MACH=${2:?机器}
LOG="runs/logs/arm_${SEG}_${MACH}.log"
exec >> "$LOG" 2>&1
say() { echo "=== $(date '+%F %T %Z') $*"; }
fail() { say "ARM_SEG_FAILED $*"; exit 1; }
SRCC="$(cat .v2d-source-commit 2>/dev/null || echo unknown)"
case "$SRCC" in *-syncing|*-dirty) fail "来历标记是 ${SRCC}" ;; esac
Q="stages/policy/queues/${MACH}_${SEG}"
say "开始，来历 ${SRCC}；${Q}"
scripts/dev/record "${SEG}/cfgcheck_${MACH}" stages/policy/fc_config_check.sh "${Q}_smoke.txt" "${Q}.txt" > /dev/null 2>&1 || fail "配置核对没过（results/${SEG}/cfgcheck_${MACH}.txt.failed）"
scripts/dev/record "${SEG}/precheck_${MACH}" stages/policy/precheck_expect.sh "${Q}.expect.tsv" "${Q}_smoke.txt" "${Q}.txt" > /dev/null 2>&1 || fail "预检没过（results/${SEG}/precheck_${MACH}.txt.failed）"
dry="$(V2D_ARM_DRYRUN=1 bash stages/policy/arm_var_chain.sh "$SEG" 2>&1)"; tail -n 3 <<< "$dry"
[ "$(tail -n 1 <<< "$dry")" = DRYRUN_OK ] || fail "起链预演没过"
arm="$(bash stages/policy/arm_var_chain.sh "$SEG" 2>&1)"; tail -n 3 <<< "$arm"
[ "$(tail -n 1 <<< "$arm")" = ARMED ] || fail "起链没成"
say "ARM_SEG_DONE"
