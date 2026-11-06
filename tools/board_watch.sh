#!/usr/bin/env bash
# 定时看 Kaggle 榜（Mac 上执行；凭据只用 kaggle CLI 自己读的 ~/.kaggle/access_token，不经命令行、不落日志）。
# 1. 12 个 Track 2 榜各取一份快照，记进 results/board/lb_<MMDD>_<HHMM>.txt（时间是 MDT）；
# 2. 与上一份快照逐榜比较、算总名次，记进 results/board/diff_<MMDD>_<HHMM>.txt（stages/evaluate/board_diff.py），并打印比较结果。
# 任何一个榜取不到（没有网、kaggle 出错、空表）就失败退出：半截快照不拿去比，“取不到”也不会被当成“没变”。
# 用法：bash tools/board_watch.sh [上一份快照]
#   不给就取 results/board/lb_*.txt 里最新的一份，没有就取 results/submission/leaderboard_1*.txt 里最新的一份。
set -uo pipefail
cd "$(dirname "$0")/.." || exit 2
K=${V2D_KAGGLE:-$HOME/.cache/uv/archive-v0/uvfKxZAYaAMiPZYi/bin/kaggle}
[ -x "$K" ] || K=$(command -v kaggle 2>/dev/null || true)
if [ -z "$K" ] || [ ! -x "$K" ]; then echo "board_watch: 找不到 kaggle CLI，用 V2D_KAGGLE 指定" >&2; exit 2; fi
prev=${1:-}
[ -n "$prev" ] || prev=$(ls results/board/lb_*.txt 2>/dev/null | sort | tail -n 1)
[ -n "$prev" ] || prev=$(ls results/submission/leaderboard_1*.txt 2>/dev/null | sort | tail -n 1)
if [ ! -f "$prev" ]; then echo "board_watch: 没有上一份快照" >&2; exit 2; fi
stamp=$(TZ=America/Denver date +%m%d_%H%M)
new="results/board/lb_${stamp}.txt"
if ! scripts/dev/record "board/lb_${stamp}" bash tools/board_snapshot.sh "$K" > /dev/null 2>&1; then
  echo "board_watch: 快照失败，见 ${new}.failed" >&2
  tail -n 5 "${new}.failed" >&2 2>/dev/null
  exit 1
fi
if ! scripts/dev/record "board/diff_${stamp}" python3 stages/evaluate/board_diff.py "$prev" "$new" > /dev/null 2>&1; then
  echo "board_watch: 比较失败，见 results/board/diff_${stamp}.txt.failed" >&2
  exit 1
fi
echo "快照 ${new}；比较 results/board/diff_${stamp}.txt"
sed -n '/^# -----/,$p' "results/board/diff_${stamp}.txt" | tail -n +2
