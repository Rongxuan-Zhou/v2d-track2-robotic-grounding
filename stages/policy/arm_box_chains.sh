#!/usr/bin/env bash
# 起方盒重训的三个等待中的 tmux（训练机上、仓库根执行）。计划 agent-doc/plan-box6-and-eval-runtime-2026-10-01.md 第 4.4 节。
# 起之前确认：compat、方盒训练与评测的日志、门槛文件、长链标记、同名 tmux 会话都不存在；要等的上一段评测日志存在。
# 按 compat → box（chain_after.sh，门槛 1）→ box-eval 的顺序起，起完打印三个会话与各自日志的最后一行。最后一行 ARMED。
# 用法：stages/policy/arm_box_chains.sh mf4090|ml40
set -euo pipefail
cd "$(dirname "$0")/../.."
M=${1:?用法：arm_box_chains.sh mf4090|ml40}
case "$M" in
  mf4090) SRC="$HOME/v2d/fc_src"; WAIT=runs/logs/t2b_eval_mf4090.log; PEAK=24564; EXTRA="V2D_PRUNE_SMOKE_STATE=1 V2D_MIN_FREE_MB=1200" ;;
  ml40)   SRC=/lab_data/rxzhou/v2d/fc_src; WAIT=runs/logs/t2c_eval_ml40.log; PEAK=44000; EXTRA="" ;;
  *) echo "机器只能是 mf4090 或 ml40" >&2; exit 2 ;;
esac
for f in "runs/logs/compat_$M.log" "runs/logs/compat_$M.gate" "runs/logs/box_train_$M.log" "runs/logs/box_eval_$M.log" \
         "runs/.chain-active-compat_$M" "runs/.chain-active-box_train_$M" "runs/.chain-active-box_eval_$M"; do
  [ ! -e "$f" ] || { echo "已存在 ${f}，停" >&2; exit 1; }
done
for s in v2d-compat v2d-box v2d-box-eval; do
  if tmux has-session -t "$s" 2>/dev/null; then echo "tmux 会话 $s 已存在，停" >&2; exit 1; fi
done
[ -f "$WAIT" ] || { echo "要等的 $WAIT 不存在，停" >&2; exit 1; }
[ -f "$SRC/fc_8c778f9/SOURCE_V2D.txt" ] && [ -f "$SRC/v040_p185/SOURCE_V2D.txt" ] || { echo "$SRC 下缺副本，停" >&2; exit 1; }
tmux new-session -d -s v2d-compat -c "$PWD" \
  "V2D_FC_SRC=$SRC/fc_8c778f9 V2D_WAIT_FOR_EVAL_LOG=$WAIT stages/evaluate/compat_8c778f9.sh stages/evaluate/compat/$M.tsv compat_$M"
sleep 2
tmux new-session -d -s v2d-box -c "$PWD" \
  "V2D_ALLOW_FC_SRC=1 V2D_FC_SRC=$SRC/v040_p185 $EXTRA stages/policy/chain_after.sh runs/logs/compat_$M.gate runs/logs/compat_$M.log 1 stages/policy/queues/${M}_box.expect.tsv stages/policy/queues/${M}_box_smoke.txt stages/policy/queues/${M}_box.txt $PEAK box_train_$M"
sleep 2
tmux new-session -d -s v2d-box-eval -c "$PWD" \
  "V2D_FC_SRC=$SRC/fc_8c778f9 V2D_EVAL_STEPS=349999104 V2D_WAIT_FOR_LOG=runs/logs/box_train_$M.log stages/evaluate/day2_eval_chain.sh stages/policy/queues/${M}_box.expect.tsv - box_eval_$M"
sleep 3
tmux ls | grep -E '^v2d-(compat|box|box-eval):'
echo "compat：$(tail -1 "runs/logs/compat_$M.log")"
echo "box-eval：$(tail -1 "runs/logs/box_eval_$M.log")"
echo "chain_after 进程：$(pgrep -f "[c]hain_after.sh runs/logs/compat_$M.gate" | tr '\n' ' ')"
ls runs/.chain-active-*
echo ARMED
