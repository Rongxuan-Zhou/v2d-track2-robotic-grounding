#!/usr/bin/env bash
# 起 ep7 最后一份有限 checkpoint 的评测链（训练机上、仓库根执行）。计划 agent-doc/plan-ep7-eval-and-guards-2026-10-02.md 的 A 部分。
# 起之前确认：日志、长链标记、tmux 会话 v2d-ep7eval 都不存在；本机没有别的长链在跑；先跑一次试跑模式（V2D_EP7_EVAL_DRYRUN=1，
# 它核两份源码副本、policy 有限、没有旧的评测 JSON 与取证 npz），要打印 EP7_EVAL_DRYRUN_OK。
# 起的链：stages/evaluate/ep7_ckpt_eval.sh <run> <步数> 896 ep7_eval_<机器>，日志 runs/logs/ep7_eval_<机器>.log。最后一行 ARMED。
# 896 是 iron_round_table_02 的帧数（方盒期望表 stages/policy/queues/<机器>_box.expect.tsv 的 src_frames）。
# 用法：stages/evaluate/arm_ep7_eval.sh mf4090|ml40
set -euo pipefail
cd "$(dirname "$0")/../.."
M=${1:?用法：arm_ep7_eval.sh mf4090|ml40}
case "$M" in
  mf4090) SRC="$HOME/v2d/fc_src"; RUN=t1-ep7-w2048-cap4m-p185-mf4090; STEP=130004992 ;;
  ml40)   SRC=/lab_data/rxzhou/v2d/fc_src; RUN=t2-ep7-w2048-cap10m-p185-ml40; STEP=140005376 ;;
  *) echo "机器只能是 mf4090 或 ml40" >&2; exit 2 ;;
esac
NAME="ep7_eval_$M"; FRAMES=896
for f in "runs/logs/$NAME.log" "runs/.chain-active-$NAME"; do [ ! -e "$f" ] || { echo "已存在 ${f}，停" >&2; exit 1; }; done
if tmux has-session -t v2d-ep7eval 2>/dev/null; then echo "tmux 会话 v2d-ep7eval 已存在，停" >&2; exit 1; fi
for f in runs/.chain-active-*; do [ ! -e "$f" ] || { echo "还有长链在跑（${f}），停：先确认它与这次评测不抢 GPU" >&2; exit 1; }; done
ENVS="V2D_EVAL_SRC=$SRC/fc_8c778f9 V2D_ROLLOUT_SRC=$SRC/v040_p185"
env $ENVS V2D_EP7_EVAL_DRYRUN=1 stages/evaluate/ep7_ckpt_eval.sh "$RUN" "$STEP" "$FRAMES" "$NAME" | tee /dev/stderr | grep -q '^EP7_EVAL_DRYRUN_OK$' \
  || { echo "试跑模式没过，停" >&2; exit 1; }
# 以下 tmux 起来之后的几行只供人看：以会话与长链标记在不在为准打印 ARMED
set +e
tmux new-session -d -s v2d-ep7eval -c "$PWD" "env $ENVS stages/evaluate/ep7_ckpt_eval.sh $RUN $STEP $FRAMES $NAME"
sleep 3
tmux ls | grep -E '^v2d-ep7eval:'
echo "ep7eval：$(tail -1 "runs/logs/$NAME.log" 2>/dev/null)"
if tmux has-session -t v2d-ep7eval 2>/dev/null && [ -e "runs/.chain-active-$NAME" ]; then echo ARMED; else echo "没起来：会话或长链标记不在" >&2; exit 1; fi
