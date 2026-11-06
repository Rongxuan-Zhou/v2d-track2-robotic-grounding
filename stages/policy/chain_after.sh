#!/usr/bin/env bash
# 等上一段评完、上一段训练够格，再开下一段训练链（训练机上、仓库根执行，放 tmux）。计划 agent-doc/plan-t2-batch-2026-10-01.md 第 4 节。
# 用法：stages/policy/chain_after.sh <上一段训练日志> <上一段评测日志> <至少完成几条> <期望表> <冒烟队列> <正式队列> <冒烟显存上限 MiB> <日志名>
#   后五个参数原样交给 stages/policy/day2_train_chain.sh。环境变量（V2D_PRUNE_SMOKE_STATE、V2D_MIN_FREE_MB 等）照样传下去。
# 1. 每分钟看一次上一段评测日志的最后一行，变成 DAY2_EVAL_DONE / DAY2_EVAL_FAILED 才往下走（上一段评测链在等待和运行期间
#    一直留着 runs/.chain-active-*，挡住同步）；
# 2. 上一段训练日志最后一行是 DAY2_TRAIN_DONE，且其中“完成”的条数 ≥ 门槛，才 exec 训练链；
# 3. 否则在 runs/logs/<日志名>.log 写原因与 DAY2_TRAIN_FAILED（下一段的评测链按这一行结束、什么也不评），退出码 1。
#    “完成”不够有几种来路：发散、失败、显存不足（训练链遇到退出码 4 直接 break）、冒烟没过。几种情形都不该接着烧下一段的 GPU。
# 4. 可选的 V2D_GATE_CMD：门槛过了之后在仓库根用 bash -c 跑它，输出与退出码记进下一段的日志；退出码 0 才开，非 0 照第 3 条处理。
#    第 3 步配方试验（计划 agent-doc/plan-step3-recipe-trial-2026-10-04.md 第 3 节）用它跑 trial_verdict.py --midcheck。
set -uo pipefail
cd "$(dirname "$0")/../.."
A_TRAIN=${1:?用法见文件头}; A_EVAL=${2:?}; MIN=${3:?}; shift 3
[ $# -eq 5 ] || { echo "chain_after: 训练链要 5 个参数，给了 $#" >&2; exit 2; }
[[ $MIN =~ ^[0-9]+$ ]] || { echo "chain_after: 门槛必须是整数：$MIN" >&2; exit 2; }
NAME=$5; B_LOG="runs/logs/$NAME.log"; mkdir -p runs/logs
until grep -qE "DAY2_EVAL_(DONE|FAILED)" < <(tail -1 "$A_EVAL" 2>/dev/null); do sleep "${V2D_POLL_S:-60}"; done
last="$(tail -1 "$A_TRAIN" 2>/dev/null)"
done_n="$(sed -n 's/.*DAY2_TRAIN_DONE（完成 \([0-9][0-9]*\)、.*/\1/p' <<< "$last")"
if [ -n "$done_n" ] && [ "$done_n" -ge "$MIN" ]; then
  if [ -n "${V2D_GATE_CMD:-}" ]; then
    gate_out="$(bash -c "$V2D_GATE_CMD" 2>&1)"; grc=$?
    { echo "=== $(date '+%F %T %Z') chain_after：判定命令 ${V2D_GATE_CMD} 退出码 ${grc}"; sed 's/^/    /' <<< "$gate_out"; } >> "$B_LOG"
    if [ "$grc" != 0 ]; then
      { echo "=== $(date '+%F %T %Z') chain_after：判定不过，不开 ${NAME}（上一段完成 ${done_n} 条）"
        echo "=== $(date '+%F %T %Z') DAY2_TRAIN_FAILED"; } >> "$B_LOG"
      exit 1
    fi
  fi
  echo "=== $(date '+%F %T %Z') chain_after：上一段完成 ${done_n} 条（门槛 ${MIN}），开 $NAME" >> "$B_LOG"
  exec stages/policy/day2_train_chain.sh "$@"
fi
{ echo "=== $(date '+%F %T %Z') chain_after：不开 ${NAME}。上一段训练日志最后一行：${last:-（空）}；要求 DAY2_TRAIN_DONE 且完成 ≥ ${MIN}"
  echo "=== $(date '+%F %T %Z') DAY2_TRAIN_FAILED"; } >> "$B_LOG"
exit 1
