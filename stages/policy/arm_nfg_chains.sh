#!/usr/bin/env bash
# 起 ep7 带非有限防护重训的两个 tmux（训练机上、仓库根执行）。计划 agent-doc/plan-ep7-eval-and-guards-2026-10-02.md 第 5 版 B.5。
# 起之前确认：两段日志、长链标记、同名 tmux 会话都不存在；副本 v040_p185_nfg 与 fc_8c778f9 都在，且 v040_p185_nfg 按 sha256 清单核对通过；
# 期望表里每条 run 与冒烟的 runs/cfgcheck/<run>.yaml 都在（fc_config_check.sh 先生成；链开头的正向核对缺它就停）。
# 训练链 day2_train_chain.sh 的环境带 V2D_ALLOW_FC_SRC=1 V2D_FC_SRC=<副本>/v040_p185_nfg（链自己读这两个变量，不写进队列行）；
# 评测链等训练日志的最后一行，再用 fc_8c778f9 副本只评最终 checkpoint（4096 世界）。起完打印两个会话与各自日志的最后一行，最后一行 ARMED。
# 用法：stages/policy/arm_nfg_chains.sh mf4090|ml40
set -euo pipefail
cd "$(dirname "$0")/../.."
M=${1:?用法：arm_nfg_chains.sh mf4090|ml40}
case "$M" in
  mf4090) SRC="$HOME/v2d/fc_src"; PEAK=24564; EXTRA="V2D_PRUNE_SMOKE_STATE=1 V2D_MIN_FREE_MB=1200" ;;
  ml40)   SRC=/lab_data/rxzhou/v2d/fc_src; PEAK=44000; EXTRA="" ;;
  *) echo "机器只能是 mf4090 或 ml40" >&2; exit 2 ;;
esac
Q=stages/policy/queues; EXP=$Q/${M}_nfg.expect.tsv
for f in "runs/logs/nfg_train_$M.log" "runs/logs/nfg_eval_$M.log" "runs/.chain-active-nfg_train_$M" "runs/.chain-active-nfg_eval_$M"; do
  [ ! -e "$f" ] || { echo "已存在 ${f}，停" >&2; exit 1; }
done
for s in v2d-nfg-train v2d-nfg-eval; do
  if tmux has-session -t "$s" 2>/dev/null; then echo "tmux 会话 $s 已存在，停" >&2; exit 1; fi
done
[ -f "$SRC/v040_p185_nfg/SOURCE_V2D.txt" ] && [ -f "$SRC/fc_8c778f9/SOURCE_V2D.txt" ] || { echo "$SRC 下缺副本，停" >&2; exit 1; }
python3 stages/policy/fc_src_check.py "$SRC/v040_p185_nfg" third_party/patches/flash_chord-p185-nfg.sha256 | tail -1 | grep -q '^FC_SRC_OK' \
  || { echo "$SRC/v040_p185_nfg 按清单核对不过，停" >&2; exit 1; }
while IFS=$'\t' read -r run smoke _; do
  for r in "$run" "$smoke"; do [ -f "runs/cfgcheck/$r.yaml" ] || { echo "缺 runs/cfgcheck/$r.yaml（先跑 fc_config_check.sh），停" >&2; exit 1; }; done
done < <(tail -n +2 "$EXP" | grep -v '^[[:space:]]*$')
tmux new-session -d -s v2d-nfg-train -c "$PWD" \
  "V2D_ALLOW_FC_SRC=1 V2D_FC_SRC=$SRC/v040_p185_nfg $EXTRA stages/policy/day2_train_chain.sh $EXP $Q/${M}_nfg_smoke.txt $Q/${M}_nfg.txt $PEAK nfg_train_$M"
sleep 2
tmux new-session -d -s v2d-nfg-eval -c "$PWD" \
  "V2D_FC_SRC=$SRC/fc_8c778f9 V2D_EVAL_STEPS=349999104 V2D_WAIT_FOR_LOG=runs/logs/nfg_train_$M.log stages/evaluate/day2_eval_chain.sh $EXP - nfg_eval_$M"
sleep 3
tmux ls | grep -E '^v2d-nfg-(train|eval):'
echo "train：$(tail -1 "runs/logs/nfg_train_$M.log")"
echo "eval：$(tail -1 "runs/logs/nfg_eval_$M.log")"
ls runs/.chain-active-*
echo ARMED
