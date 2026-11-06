#!/usr/bin/env bash
# t12lift 一条龙（ml40 上、仓库根执行，放 tmux；计划 agent-doc/plan-t12lift-2026-10-20.md 第 2 版第 1 节）。
#   arm_t12lift.sh t2：写 Tier 2 ep25、ep28 与 Tier 1 ep25 的保底参考（hands_off.py --arm-frame-scan，挂 v040_p185_nfg）→ check_inputs、支撑面 md5 →
#                      t2ho 配置核对、预检、预演、起链 → 挂等待：t2ho 的导出链结束后跑“arm_t12lift.sh t1”。
#   arm_t12lift.sh t1：t1ho 配置核对、预检、预演、起链。
# 日志 runs/logs/arm_t12lift_ml40.log，每段最后一行 ARM_T12LIFT_<段>_DONE 或 ARM_T12LIFT_<段>_FAILED <原因>。任何一步不过就停，不起链；不停任何进程。
set -uo pipefail
cd "$(dirname "$0")/../.."
STEP=${1:?用法：arm_t12lift.sh t2|t1}
LOG=runs/logs/arm_t12lift_ml40.log
exec >> "$LOG" 2>&1
say() { echo "=== $(date '+%F %T %Z') $*"; }
fail() { say "ARM_T12LIFT_${STEP}_FAILED $*"; exit 1; }
SRC=/lab_data/rxzhou/v2d/fc_src/v040_p185_nfg
S25=sequence_id=2026-03-20_11-59-57_cyan_water_bottle_rotate_01
S28=sequence_id=2026-05-19_11-27-07_squeegee_sweep_away_from_you_03_
arm() {   # arm <段>
  local Q=stages/policy/queues/ml40_$1
  scripts/dev/record "t12lift/cfgcheck_$1_ml40" stages/policy/fc_config_check.sh "${Q}_smoke.txt" "${Q}.txt" > /dev/null 2>&1 || fail "$1 配置核对没过"
  scripts/dev/record "t12lift/precheck_$1_ml40" stages/policy/precheck_expect.sh "${Q}.expect.tsv" "${Q}_smoke.txt" "${Q}.txt" > /dev/null 2>&1 || fail "$1 预检没过"
  local dry; dry="$(V2D_ARM_DRYRUN=1 bash stages/policy/arm_var_chain.sh "$1" 2>&1)"; tail -n 3 <<< "$dry"
  [ "$(tail -n 1 <<< "$dry")" = DRYRUN_OK ] || fail "$1 起链预演没过"
  local out; out="$(bash stages/policy/arm_var_chain.sh "$1" 2>&1)"; tail -n 3 <<< "$out"
  [ "$(tail -n 1 <<< "$out")" = ARMED ] || fail "$1 起链没成"
}
say "开始 ${STEP}，来历 $(cat .v2d-source-commit 2>/dev/null || echo unknown)"
if [ "$STEP" = t2 ]; then
  for spec in "tier2:$S25" "tier2:$S28" "tier1:$S25"; do
    d=${spec%%:*}; s=${spec#*:}
    [ -e "runs/motion/${d}_ho/whole_body/soma/$s" ] && { say "${d}_ho/$s 已在，不重写"; continue; }
    V2D_ALLOW_FC_SRC=1 V2D_FC_SRC=$SRC scripts/dev/record "t12lift/write_${d}_${s#sequence_id=}" scripts/dev/in-flash-chord \
      "python /repo/stages/retarget/hands_off.py /repo/runs/motion/$d/whole_body/soma/$s/robot_name=g1/data.parquet /repo/runs/motion/${d}_ho --meta-src /repo/runs/motion_src/$d --meta-dst /repo/runs/motion_src/${d}_ho --arm-frame-scan" \
      > /dev/null 2>&1 || fail "写 ${d}_ho/$s 没成（results/t12lift/write_*.failed）"
    say "写好 ${d}_ho/$s：$(grep -h '固定为' "results/t12lift/write_${d}_${s#sequence_id=}.txt" | cut -c1-120 | sed 's/.*固定为/固定为/')"
  done
  pq="$(ls runs/motion/tier2_ho/whole_body/soma/*/robot_name=g1/data.parquet runs/motion/tier1_ho/whole_body/soma/*/robot_name=g1/data.parquet | sed 's,^,/repo/,' | tr '\n' ' ')"
  scripts/dev/record t12lift/check_inputs_ml40 scripts/dev/in-flash-chord "python /repo/stages/policy/check_inputs.py $pq" > /dev/null 2>&1 || fail "check_inputs 没过"
  for d in tier2 tier1; do
    for f in runs/motion/${d}_ho/whole_body/reconstructed_stage/*; do
      cmp -s "$f" "runs/motion/$d/whole_body/reconstructed_stage/$(basename "$f")" || fail "支撑面 $f 与 $d 那份不同"
    done
  done
  say "check_inputs 过；支撑面与原档位逐字节相同"
  arm t2ho
  V2D_POLL_S=60 V2D_MAX_POLLS=900 setsid nohup stages/policy/run_after_log.sh runs/logs/cand_t2_t2ho_ml40.log "EXPORT_CHAIN_(DONE|FAILED)" \
    t12lift/arm_t1_after_ml40 bash stages/policy/arm_t12lift.sh t1 > runs/logs/after_t12lift_ml40.log 2>&1 < /dev/null &
  sleep 3
  say "t1ho 的等待已挂：$(head -n 1 runs/logs/after_t12lift_ml40.log | cut -c1-160)"
elif [ "$STEP" = t1 ]; then
  arm t1ho
else
  fail "未知的段 $STEP"
fi
say "ARM_T12LIFT_${STEP}_DONE"
