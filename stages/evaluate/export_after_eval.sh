#!/usr/bin/env bash
# 评测链成功结束才开导出链（训练机上、仓库根执行，放 tmux）。计划 agent-doc/plan-t1-r3x-2026-10-08.md 第 3 节（Fable 5.1 审查 M2）。
# export_chain.sh 自己的 V2D_WAIT_FOR_EVAL_LOG 见到 DAY2_EVAL_DONE 或 DAY2_EVAL_FAILED 都会开始。评测链失败（例如开头见到 GPU 上有别的进程）时
# 照样导出，导出的轨迹没有 4096 世界评测，记 no_eval、进不了候选，而“每个 checkpoint 只导出一次”又不许重导。这里先拦一道：
# 1. 一开始写长链标记 runs/.chain-active-<导出名>-gate（等待期间挡住同步），退出时删
# 2. 每分钟（V2D_POLL_S）看评测日志的最后一行，变成 DAY2_EVAL_DONE / DAY2_EVAL_FAILED 才往下走
# 3. 最后一行是 DAY2_EVAL_DONE（成功 N、失败 0）且 N ≥ 1：在本进程里跑 export_chain.sh <清单> <导出名>（环境变量照样传下去），退出码照传
# 4. 否则不导出：在 runs/logs/<导出名>_gate.log 写原因，最后一行 EXPORT_GATE_STOPPED，退出码 1。导出链的日志与结果表都不建，
#    补评之后可以用同一个名字手动起导出链
# 门日志里不抄评测链结束标记的原文（同 export_chain.sh 的 demark：10/1 起链事故的教训）。
# 用法：stages/evaluate/export_after_eval.sh <评测日志> <导出清单> <导出名>
# 测试：stages/evaluate/test_export_after_eval.py。
set -uo pipefail
cd "$(dirname "$0")/../.."
EVAL_LOG=${1:?用法见文件头}; LIST=${2:?用法见文件头}; NAME=${3:?用法见文件头}
[[ $NAME =~ ^[A-Za-z0-9_-]+$ ]] || { echo "export_after_eval: 名字非法：$NAME" >&2; exit 2; }
mkdir -p runs/logs
GLOG="runs/logs/${NAME}_gate.log"; MARK="runs/.chain-active-${NAME}-gate"
echo "${NAME}-gate waiting $(date '+%F %T %Z')" > "$MARK"; trap 'rm -f "$MARK"' EXIT
say() { echo "=== $(date '+%F %T %Z') $*" >> "$GLOG"; }
say "等 $EVAL_LOG 的最后一行变成评测链的结束行"
until grep -qE "DAY2_EVAL_(DONE|FAILED)" < <(tail -1 "$EVAL_LOG" 2>/dev/null); do sleep "${V2D_POLL_S:-60}"; done
last="$(tail -1 "$EVAL_LOG")"
n_ok="$(sed -n 's/.*DAY2_EVAL_DONE（成功 \([0-9][0-9]*\)、失败 0）.*/\1/p' <<< "$last")"
if [ -n "$n_ok" ] && [ "$n_ok" -ge 1 ]; then
  say "评测链成功 ${n_ok} 条、失败 0，开导出链 $NAME"
  stages/evaluate/export_chain.sh "$LIST" "$NAME"; rc=$?
  say "导出链结束，退出码 $rc"
  exit "$rc"
fi
say "评测链没有以“成功 N、失败 0”结束，不导出。评测日志最后一行：$(sed -E 's/DAY2_EVAL_(DONE|FAILED)/评测链结束/' <<< "$last")"
say EXPORT_GATE_STOPPED
exit 1
