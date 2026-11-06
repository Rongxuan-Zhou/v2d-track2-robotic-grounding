#!/usr/bin/env bash
# t2fc 一条龙（ml40 上、仓库根执行，放 tmux；计划 agent-doc/plan-t2fc-2026-10-20.md 第 1 节第 4–6 步）：
# 1. 等诊断链日志 runs/logs/t2dg_diag_ml40.log 的最后一行变成 DIAG2_DONE / DIAG2_FAILED（最多 4 小时）；
# 2. 补两条对照的评测 dump 与离开帧（stages/evaluate/t2fc_readout.sh）；3. 起链预演（要 DRYRUN_OK）、起链（要 ARMED）；
# 4. 挂读数等待：评测链 runs/logs/t2fc_eval_ml40.log 结束后，对两条新 run 跑 t2fc_readout.sh。
# 日志 runs/logs/arm_t2fc_ml40.log，最后一行 ARM_T2FC_DONE 或 ARM_T2FC_FAILED <原因>。任何一步不过就停，不起链。
set -uo pipefail
cd "$(dirname "$0")/../.."
LOG=runs/logs/arm_t2fc_ml40.log
exec >> "$LOG" 2>&1
say() { echo "=== $(date '+%F %T %Z') $*"; }
fail() { say "ARM_T2FC_FAILED $*"; exit 1; }
M2=runs/motion/tier2/whole_body/soma; M8=runs/motion/tier2_fc8/whole_body/soma
S25=sequence_id=2026-03-20_11-59-57_cyan_water_bottle_rotate_01/robot_name=g1/data.parquet
S4=sequence_id=2026-04-24_11-40-52_baseball_bat_rotate_05/robot_name=g1/data.parquet
say "开始，来历 $(cat .v2d-source-commit 2>/dev/null || echo unknown)"
for _ in $(seq 1 240); do grep -qE "DIAG2_(DONE|FAILED)" < <(tail -n 1 runs/logs/t2dg_diag_ml40.log 2>/dev/null) && break; sleep 60; done
grep -qE "DIAG2_(DONE|FAILED)" < <(tail -n 1 runs/logs/t2dg_diag_ml40.log) || fail "诊断链 4 小时内没结束"
say "诊断链结束：$(tail -n 1 runs/logs/t2dg_diag_ml40.log | sed -E 's/DIAG2_(DONE|FAILED)/诊断链结束/')"
out="$(stages/evaluate/t2fc_readout.sh "t2-ep25-w2048-cap10m-r3a-ml40:$M2/$S25" "t2-ep4-w2048-cap10m-r3a-ml40:$M2/$S4")"
echo "$out" | sed 's/T2FC_READOUT_DONE/对照读数完成/'
[[ $(tail -n 1 <<< "$out") == *"失败 0）" ]] || fail "对照的评测 dump 没做全"
dry="$(V2D_ARM_DRYRUN=1 bash stages/policy/arm_var_chain.sh t2fc 2>&1)"; tail -n 3 <<< "$dry"
[ "$(tail -n 1 <<< "$dry")" = DRYRUN_OK ] || fail "起链预演没过"
arm="$(bash stages/policy/arm_var_chain.sh t2fc 2>&1)"; tail -n 3 <<< "$arm"
[ "$(tail -n 1 <<< "$arm")" = ARMED ] || fail "起链没成"
V2D_POLL_S=60 V2D_MAX_POLLS=720 setsid nohup stages/policy/run_after_log.sh runs/logs/t2fc_eval_ml40.log "DAY2_EVAL_(DONE|FAILED)" \
  t2fc/readout_new_ml40 stages/evaluate/t2fc_readout.sh "t2-ep25-w2048-cap10m-r3a-fc8-ml40:$M8/$S25" "t2-ep4-w2048-cap10m-r3a-fc8-ml40:$M8/$S4" \
  > runs/logs/after_t2fc_ml40.log 2>&1 < /dev/null &
sleep 3
say "读数等待已挂：$(head -n 1 runs/logs/after_t2fc_ml40.log | cut -c1-160)"
say ARM_T2FC_DONE
