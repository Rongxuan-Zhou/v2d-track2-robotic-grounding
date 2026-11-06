#!/usr/bin/env bash
# 起 MP-SR 为 0 的序列诊断链（训练机上、仓库根执行）。计划 agent-doc/plan-diag-mpsr0-2026-10-02.md 第 5 节第 4 步。
# 起之前确认：诊断日志、长链标记、tmux 会话 v2d-diag2 都不存在；要等的本机导出链日志 runs/logs/export_<机器>.log 存在；运行表在。
# 起的链：等导出链日志的最后一行变成结束行，再按 stages/evaluate/diag2/<机器>.tsv 诊断，日志 runs/logs/diag2_<机器>.log。最后一行 ARMED。
# 用法：stages/evaluate/arm_diag2.sh mf4090|ml40
set -euo pipefail
cd "$(dirname "$0")/../.."
M=${1:?用法：arm_diag2.sh mf4090|ml40}
case "$M" in mf4090|ml40) ;; *) echo "机器只能是 mf4090 或 ml40" >&2; exit 2 ;; esac
NAME="diag2_$M"; TABLE="stages/evaluate/diag2/$M.tsv"; WAIT="runs/logs/export_$M.log"
for f in "runs/logs/$NAME.log" "runs/.chain-active-$NAME"; do [ ! -e "$f" ] || { echo "已存在 ${f}，停" >&2; exit 1; }; done
if tmux has-session -t v2d-diag2 2>/dev/null; then echo "tmux 会话 v2d-diag2 已存在，停" >&2; exit 1; fi
[ -f "$WAIT" ] || { echo "要等的 $WAIT 不存在，停" >&2; exit 1; }
[ -r "$TABLE" ] || { echo "读不了运行表 ${TABLE}，停" >&2; exit 1; }
# 以下 tmux 起来之后的几行只供人看：以会话与长链标记在不在为准打印 ARMED
set +e
tmux new-session -d -s v2d-diag2 -c "$PWD" "env -u V2D_FC_SRC V2D_WAIT_FOR_LOG=$WAIT stages/evaluate/diag2_chain.sh $TABLE $NAME"
sleep 3
tmux ls | grep -E '^v2d-diag2:'
echo "diag2：$(tail -1 "runs/logs/$NAME.log" 2>/dev/null)"
if tmux has-session -t v2d-diag2 2>/dev/null && [ -e "runs/.chain-active-$NAME" ]; then echo ARMED; else echo "没起来：会话或长链标记不在" >&2; exit 1; fi
