#!/usr/bin/env bash
# 等一条训练链结束，再开下一条训练链（训练机上、仓库根执行，放 tmux）。计划 agent-doc/plan-t1-r3x-2026-10-08.md 第 7 节（ep5 seed 42 原样补跑）。
# 用法：stages/policy/train_after.sh <上一条训练链日志> <期望表> <冒烟队列> <正式队列> <冒烟显存上限 MiB> <日志名>
#   后五个参数原样交给 stages/policy/day2_train_chain.sh，环境变量（V2D_FC_SRC、V2D_PRUNE_SMOKE_STATE、V2D_MIN_FREE_MB 等）照样传下去。
# 0. runs/logs/<日志名>.log 已存在：拒绝（退出码 2，不写标记）。等这条日志的评测链靠它的最后一行开始，不能接着别的链的日志写
# 1. 一开始写长链标记 runs/.chain-active-<日志名>-wait（等待期间挡住同步），交给训练链之前删掉（训练链自己写 runs/.chain-active-<日志名>）
# 2. 每分钟（V2D_POLL_S）看上一条训练链日志的最后一行，变成 DAY2_TRAIN_DONE / DAY2_TRAIN_FAILED 才往下走
# 3. 是 DAY2_TRAIN_DONE：exec day2_train_chain.sh（后五个参数）
# 4. 是 DAY2_TRAIN_FAILED（上一条链本身出错：来历变了、GPU 上有别的进程、磁盘不够等，同样的原因多半也会挡住这一条）：不开，
#    在 runs/logs/<日志名>.log 写原因，最后一行 DAY2_TRAIN_FAILED（等这条日志的评测链照常开始，评已有 DONE 的 run），退出码 1
# 抄进本日志的上一条链的行把结束标记换成中文（同 export_chain.sh 的 demark：10/1 起链事故的教训）。
# 测试：stages/policy/test_train_after.py。
set -uo pipefail
cd "$(dirname "$0")/../.."
PREV=${1:?用法见文件头}; shift
[ $# -eq 5 ] || { echo "train_after: 训练链要 5 个参数，给了 $#" >&2; exit 2; }
NAME=$5
[[ $NAME =~ ^[A-Za-z0-9_-]+$ ]] || { echo "train_after: 日志名非法：$NAME" >&2; exit 2; }
mkdir -p runs/logs
LOG="runs/logs/$NAME.log"; MARK="runs/.chain-active-${NAME}-wait"
[ ! -e "$LOG" ] || { echo "train_after: $LOG 已存在，拒绝" >&2; exit 2; }
echo "${NAME}-wait waiting $(date '+%F %T %Z')" > "$MARK"; trap 'rm -f "$MARK"' EXIT
until grep -qE "DAY2_TRAIN_(DONE|FAILED)" < <(tail -1 "$PREV" 2>/dev/null); do sleep "${V2D_POLL_S:-60}"; done
last="$(tail -1 "$PREV")"
if grep -q "DAY2_TRAIN_DONE" <<< "$last"; then
  rm -f "$MARK"; trap - EXIT
  exec stages/policy/day2_train_chain.sh "$@"
fi
say() { echo "=== $(date '+%F %T %Z') $*" >> "$LOG"; }
say "上一条训练链 $PREV 没有正常结束，不开 ${NAME}。它的最后一行：$(sed -E 's/DAY2_TRAIN_(DONE|FAILED)/训练链结束/g' <<< "$last")"
say DAY2_TRAIN_FAILED
exit 1
