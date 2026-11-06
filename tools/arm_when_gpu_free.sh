#!/usr/bin/env bash
# 等 GPU 连续空闲一段时间再起一段链（训练机上、仓库根执行，setsid 后台跑）。计划 agent-doc/plan-t2auc3-2026-11-02.md 执行记录 11/03 02:2x：
# l40n 上另一位成员在跑她的任务，arm_var_chain.sh 见到 GPU 上有别的进程就拒绝起链；我们不停她的进程、也不在她两个任务的空档里硬塞
# （1000 万 buffer 约 40 GB，L40 46 GB，挤进去她下一个任务会爆显存）。所以等 nvidia-smi 连续 <分钟> 次（每 60 秒一次）都没有计算进程，
# 才跑 V2D_ARM_DRYRUN=1 arm_var_chain.sh <段名>，DRYRUN_OK 后再正式起；中途见到任何计算进程就从头计数。
# V2D_AFTER_SEG=<下一段>：起链成功后用 runs/tools/arm_after.sh（V2D_AFTER_INDEPENDENT=1）把下一段接上，参数同 plan-t2auc3 第 1 节。
# V2D_MAX_WAIT_H（默认 30）：总共最多等这么多小时。最后一行 ARM_WHEN_FREE_ARMED <段名> 或 ARM_WHEN_FREE_STOPPED <原因>。
# 用法：setsid nohup bash tools/arm_when_gpu_free.sh <段名> <机器> <空闲分钟> > runs/logs/arm_when_free_<段名>_<机器>.log 2>&1 &
set -uo pipefail
SEG=${1:?段名}; M=${2:?机器}; N=${3:?空闲分钟}
say() { echo "=== $(date '+%F %T %Z') $*"; }
stop() { say "ARM_WHEN_FREE_STOPPED $*"; exit 1; }
[[ $N =~ ^[0-9]+$ ]] && [ "$N" -ge 1 ] || stop "空闲分钟要是正整数：$N"
say "等 GPU 连续 $N 分钟没有计算进程再起 $SEG（$M）；来历 $(cat .v2d-source-commit 2>/dev/null)"
idle=0; t0=$SECONDS; maxs=$(( ${V2D_MAX_WAIT_H:-30} * 3600 ))
while [ "$idle" -lt "$N" ]; do
  [ $((SECONDS - t0)) -lt "$maxs" ] || stop "等了 ${V2D_MAX_WAIT_H:-30} 小时 GPU 还没空"
  [ -e "runs/.chain-stop-arm_when_free_$SEG" ] && stop "见到 runs/.chain-stop-arm_when_free_$SEG"
  apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')"
  if [ -z "$apps" ]; then idle=$((idle + 1)); else [ "$idle" -gt 0 ] && say "GPU 上又有计算进程，从头计数"; idle=0; fi
  [ "$idle" -ge "$N" ] && break
  sleep 60
done
say "GPU 已连续 $N 分钟空闲，预演 $SEG"
out="$(V2D_ARM_DRYRUN=1 stages/policy/arm_var_chain.sh "$SEG" 2>&1)"; rc=$?; printf '%s\n' "$out" | tail -3
[ $rc -eq 0 ] && [ "$(printf '%s\n' "$out" | tail -n 1)" = DRYRUN_OK ] || stop "预演没过（退出码 $rc）"
stages/policy/arm_var_chain.sh "$SEG" > "runs/logs/arm_${SEG}_${M}.log" 2>&1; rc=$?; tail -2 "runs/logs/arm_${SEG}_${M}.log"
[ $rc -eq 0 ] && [ "$(tail -n 1 "runs/logs/arm_${SEG}_${M}.log")" = ARMED ] || stop "起链没成（退出码 $rc）"
if [ -n "${V2D_AFTER_SEG:-}" ]; then
  L=runs/logs; n="$(grep -cv -e '^[[:space:]]*#' -e '^[[:space:]]*$' "stages/policy/queues/${M}_${SEG}.txt")"
  V2D_AFTER_INDEPENDENT=1 setsid nohup bash runs/tools/arm_after.sh "$V2D_AFTER_SEG" "$L/${SEG}_train_${M}.log" "$L/${SEG}_eval_${M}.log" \
    "$L/cand_t2_${SEG}_${M}.log" "$n" > "$L/arm_after_${V2D_AFTER_SEG}_${M}.log" 2>&1 < /dev/null &
  say "接力 $V2D_AFTER_SEG 已挂（上一段 $n 条，日志 $L/arm_after_${V2D_AFTER_SEG}_${M}.log）"
fi
say "ARM_WHEN_FREE_ARMED $SEG"
