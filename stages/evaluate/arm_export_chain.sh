#!/usr/bin/env bash
# 起提交导出链的等待中的 tmux（训练机上、仓库根执行）。计划 agent-doc/plan-submission-exports-2026-10-02.md 第 4 节。
# 起之前确认：导出日志、结果表、长链标记、tmux 会话 v2d-export 都不存在；要等的方盒评测日志存在；8c778f9 副本在；
# 预检（V2D_EXPORT_CHAIN_DRYRUN=1，经 scripts/dev/record 落盘到 results/submission/exports/arm_dryrun_<机器>.txt）是 EXPORT_DRYRUN_OK。
# 起的链：等 runs/logs/box_eval_<机器>.log 的最后一行变成评测链结束行，看 runs/logs/compat_<机器>.gate（不是“完成 1”只记警告），
# 然后按 stages/evaluate/exports/<机器>.tsv 导出，日志 runs/logs/export_<机器>.log，结果表 runs/exports/export_<机器>.tsv。最后一行 ARMED。
# 用法：stages/evaluate/arm_export_chain.sh mf4090|ml40
set -euo pipefail
cd "$(dirname "$0")/../.."
M=${1:?用法：arm_export_chain.sh mf4090|ml40}
case "$M" in
  mf4090) SRC="$HOME/v2d/fc_src/fc_8c778f9" ;;
  ml40)   SRC=/lab_data/rxzhou/v2d/fc_src/fc_8c778f9 ;;
  *) echo "机器只能是 mf4090 或 ml40" >&2; exit 2 ;;
esac
NAME="export_$M"; LIST="stages/evaluate/exports/$M.tsv"; WAIT="runs/logs/box_eval_$M.log"; GATE="runs/logs/compat_$M.gate"
for f in "runs/logs/$NAME.log" "runs/exports/$NAME.tsv" "runs/.chain-active-$NAME"; do
  [ ! -e "$f" ] || { echo "已存在 ${f}，停" >&2; exit 1; }
done
if tmux has-session -t v2d-export 2>/dev/null; then echo "tmux 会话 v2d-export 已存在，停" >&2; exit 1; fi
[ -f "$WAIT" ] || { echo "要等的 $WAIT 不存在，停" >&2; exit 1; }
[ -f "$SRC/SOURCE_V2D.txt" ] || { echo "$SRC 不是 8c778f9 副本，停" >&2; exit 1; }
ENVS=(V2D_FC_SRC="$SRC" V2D_WAIT_FOR_EVAL_LOG="$WAIT" V2D_COMPAT_GATE="$GATE")
scripts/dev/record "submission/exports/arm_dryrun_$M" env V2D_EXPORT_CHAIN_DRYRUN=1 "${ENVS[@]}" stages/evaluate/export_chain.sh "$LIST" "$NAME" \
  || { echo "预检不过，见 results/submission/exports/arm_dryrun_${M}.txt.failed，停" >&2; exit 1; }
tail -3 "results/submission/exports/arm_dryrun_$M.txt"
# 以下 tmux 起来之后的几行只供人看：起来了就以会话与标记为准，不让显示出错把已经在跑的链记成失败
set +e
tmux new-session -d -s v2d-export -c "$PWD" "env ${ENVS[*]} stages/evaluate/export_chain.sh $LIST $NAME"
sleep 3
tmux ls | grep -E '^v2d-export:'
echo "export：$(tail -1 "runs/logs/$NAME.log" 2>/dev/null)"
if tmux has-session -t v2d-export 2>/dev/null && [ -e "runs/.chain-active-$NAME" ]; then echo ARMED; else echo "没起来：会话或长链标记不在" >&2; exit 1; fi
