#!/usr/bin/env bash
# 正式训练加发散看守（训练机上、仓库根执行；训练链 day2_train_chain.sh 的正式阶段调用）。计划 agent-doc/plan-t2-batch-2026-10-01.md。
#   1. 后台起 ckpt_nonfinite.py --watch：按步数连续 2 份完整 policy 含非有限参数就 fc_stop（--parent 指向本脚本，本脚本不在了它也退出）；
#   2. 前台跑 fc_run.sh（参数与环境变量原样传过去）；
#   3. 收掉看守；fc_run 成功时等过 checkpoint 写入余量（ckpt_nonfinite 只认 10 秒前写完的文件），再扫一遍最新 policy。
# 退出码：fc_run.sh 的退出码，另有三种：
#   11 看守 2 秒内就退出了（没起来）：不开训练
#   9  本次看守止损：这次追加的看守日志里有 NAN_WATCH_STOPPED，或 fc_run 按 STOP 退出（3）且日志里已有看守调用 fc_stop 前打的那一行
#      （真的 fc_stop 要等 docker stop 最多 60 秒，看守可能还没打出 NAN_WATCH_STOPPED 就被收掉）；run 目录留着 state、没有 DONE，
#      以后重排要先移走或换名
#   10 fc_run 成功，但最新 policy 含非有限参数或不完整；在 run 目录写 NONFINITE，评测链见到就不评
# 看守日志 runs/logs/<run>.nanwatch.log（追加）。V2D_WATCH_INTERVAL（默认 120 秒）、V2D_WATCH_TIMEOUT（默认 43200 秒）、
# V2D_SETTLE_S（成功后等多久再扫，默认 12 秒）。
# 用法：stages/policy/fc_run_watched.sh <run> <parquet> <world_count> <total_steps> [覆盖...]
set -uo pipefail
cd "$(dirname "$0")/../.."
RUN=${1:?用法：fc_run_watched.sh <run> <parquet> <world_count> <total_steps> [覆盖...]}
[[ $RUN =~ ^[a-z0-9][a-z0-9_-]{2,59}$ ]] || { echo "fc_run_watched: run 名非法：$RUN" >&2; exit 2; }
WLOG="runs/logs/$RUN.nanwatch.log"; mkdir -p runs/logs; touch "$WLOG"
OFF="$(wc -l < "$WLOG")"
python3 stages/policy/ckpt_nonfinite.py --watch "$RUN" --interval "${V2D_WATCH_INTERVAL:-120}" --timeout "${V2D_WATCH_TIMEOUT:-43200}" \
  --stop-after 2 --parent "$$" --tag "src=$(cat .v2d-source-commit 2>/dev/null || echo unknown)" >> "$WLOG" 2>&1 &
WP=$!
sleep 2
kill -0 "$WP" 2>/dev/null || { echo "fc_run_watched: $RUN 的发散看守没起来（见 ${WLOG}），不开训练"; tail -3 "$WLOG"; exit 11; }
stop_watch() { kill "$WP" 2>/dev/null; wait "$WP" 2>/dev/null; }
trap stop_watch EXIT
trap 'exit 130' HUP INT TERM
bash stages/policy/fc_run.sh "$@" < /dev/null; rc=$?
stop_watch; trap - EXIT
NEW="$(tail -n +"$((OFF + 1))" "$WLOG")"   # 这次追加的看守日志（不经管道：pipefail 下 tail 收 SIGPIPE 会盖掉匹配）
if grep -q NAN_WATCH_STOPPED <<< "$NEW" || { [ "$rc" = 3 ] && grep -q "：fc_stop $RUN\$" <<< "$NEW"; }; then
  echo "fc_run_watched: $RUN 被发散看守止损（fc_run 退出码 ${rc}），见 $WLOG"; exit 9
fi
if [ "$rc" = 0 ]; then
  sleep "${V2D_SETTLE_S:-12}"
  if ! out="$(python3 stages/policy/ckpt_nonfinite.py --last --fail-on-nonfinite "$RUN")"; then
    echo "$out"; date '+%F %T %Z' > "runs/flash_chord/$RUN/NONFINITE"
    echo "fc_run_watched: $RUN 训完，但最新 policy 含非有限参数或不完整（已写 NONFINITE）"; exit 10
  fi
  echo "fc_run_watched: $RUN 最新 policy$(tail -1 <<< "$out")"
fi
exit "$rc"
