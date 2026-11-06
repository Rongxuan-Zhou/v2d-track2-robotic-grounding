#!/usr/bin/env bash
# 让卡给另一位成员的一次任务，等它结束再把同名段重新起起来（训练机上、仓库根执行，setsid 后台跑）。计划 agent-doc/plan-t3auc1-2026-11-03.md 执行记录 11/03 12:0x MST：
# rxzhou 让 ml40 先给另一位成员的 E1 重建实验跑；t3qr3o2c 用停链标记 runs/.chain-stop-<段名>_train_<机器> 停在冒烟之后（冒烟已 DONE，正式 run 没起）。
# 她的脚本等我们的 .chain-active-* 与 flash_chord 容器连续 30 分钟都没有才上 GPU，所以这里不能按“GPU 空了几分钟”起链（她等卡的那 30 分钟 GPU 也是空的）。
# 只读看她的进程和日志，不写她的目录、不动她的进程。步骤：
#   1. 等本段三条链（训练、评测、导出门）的 .chain-active-* 都消失
#   2. 每 60 秒看一次：<对方进程模式> 在 <对方用户> 名下连续 2 次都找不到（E1_DONE、E1_FAILED 或她那边 8 小时超时都算结束），记下 <对方日志> 最后一行
#   3. 把这次停下的段的日志和停链标记移到 runs/logs/yield_<段名>_<时间>/（不删），arm_var_chain.sh 才肯用同一个段名再起
#   4. GPU 连续 V2D_IDLE_MIN（默认 5）分钟没有计算进程，V2D_ARM_DRYRUN=1 预演，DRYRUN_OK 后正式起；中途见到计算进程从头计数
# V2D_MAX_WAIT_H（默认 10）：总共最多等这么多小时。最后一行 YIELD_REARMED <段名> 或 YIELD_STOPPED <原因>。
# 用法：setsid nohup bash runs/tools/arm_after_yield.sh <段名> <机器> <对方用户> <对方进程模式> <对方日志> > runs/logs/arm_after_yield_<段名>_<机器>.log 2>&1 < /dev/null &
set -uo pipefail
SEG=${1:?段名}; M=${2:?机器}; OU=${3:?对方用户}; OPAT=${4:?对方进程模式}; OLOG=${5:?对方日志}
say() { echo "=== $(date '+%F %T %Z') $*"; }
stop() { say "YIELD_STOPPED $*"; exit 1; }
t0=$SECONDS; maxs=$(( ${V2D_MAX_WAIT_H:-10} * 3600 ))
tick() {
  [ $((SECONDS - t0)) -lt "$maxs" ] || stop "等了 ${V2D_MAX_WAIT_H:-10} 小时"
  [ -e "runs/.chain-stop-arm_after_yield_$SEG" ] && stop "见到 runs/.chain-stop-arm_after_yield_$SEG"
  sleep 60
}
xl=$(ls stages/evaluate/exports/cand_t*_"${SEG}"_"$M".tsv 2>/dev/null | head -1)
[[ $xl =~ /cand_t([123])_ ]] || stop "找不到 stages/evaluate/exports/cand_t*_${SEG}_${M}.tsv"
T=${BASH_REMATCH[1]}
LT=${SEG}_train_$M; LE=${SEG}_eval_$M; LX=cand_t${T}_${SEG}_$M
say "等本段的链结束，再等 $OU 名下的 $OPAT 结束（日志 $OLOG），然后重起 $SEG（$M）；来历 $(cat .v2d-source-commit 2>/dev/null)"
while compgen -G "runs/.chain-active-${SEG}_*" > /dev/null || [ -e "runs/.chain-active-${LX}-gate" ]; do tick; done
say "本段的链已结束。训练：$(tail -1 "runs/logs/$LT.log" 2>/dev/null)；评测：$(tail -1 "runs/logs/$LE.log" 2>/dev/null | cut -c1-120)"
[ ! -e "runs/exports/$LX.tsv" ] || stop "已有 runs/exports/$LX.tsv，这一段导出过，不重起"
gone=0
while [ "$gone" -lt 2 ]; do
  if pgrep -u "$OU" -f "$OPAT" > /dev/null; then gone=0; else gone=$((gone + 1)); fi
  [ "$gone" -ge 2 ] && break
  tick
done
say "对方任务已结束，日志最后一行：$(tail -1 "$OLOG" 2>/dev/null | cut -c1-200)"
Y="runs/logs/yield_${SEG}_$(date +%m%d_%H%M)"; mkdir -p "$Y" || stop "建不了 $Y"
for f in "runs/logs/$LT.log" "runs/logs/$LE.log" "runs/logs/${LX}_gate.log" "runs/logs/arm_${SEG}_${M}.log" "runs/.chain-stop-$LT"; do
  [ -e "$f" ] && { mv "$f" "$Y/" || stop "移不走 $f"; }
done
say "旧日志与停链标记已移到 $Y：$(ls "$Y" | tr '\n' ' ')"
idle=0
while [ "$idle" -lt "${V2D_IDLE_MIN:-5}" ]; do
  if [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')" ]; then idle=$((idle + 1)); else idle=0; fi
  [ "$idle" -ge "${V2D_IDLE_MIN:-5}" ] && break
  tick
done
say "GPU 已连续 ${V2D_IDLE_MIN:-5} 分钟空闲，预演 $SEG"
out="$(V2D_ARM_DRYRUN=1 stages/policy/arm_var_chain.sh "$SEG" 2>&1)"; rc=$?; printf '%s\n' "$out" | tail -3
[ $rc -eq 0 ] && [ "$(printf '%s\n' "$out" | tail -n 1)" = DRYRUN_OK ] || stop "预演没过（退出码 $rc）"
stages/policy/arm_var_chain.sh "$SEG" > "runs/logs/arm_${SEG}_${M}.log" 2>&1; rc=$?; tail -2 "runs/logs/arm_${SEG}_${M}.log"
[ $rc -eq 0 ] && [ "$(tail -n 1 "runs/logs/arm_${SEG}_${M}.log")" = ARMED ] || stop "起链没成（退出码 $rc）"
say "YIELD_REARMED $SEG"
