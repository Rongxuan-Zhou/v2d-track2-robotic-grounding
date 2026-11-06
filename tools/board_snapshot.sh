#!/usr/bin/env bash
# 12 个 Track 2 Kaggle 榜各打印一份（tools/board_watch.sh 调用，经 scripts/dev/record 落盘）。参数：kaggle CLI 的路径。
# 每个榜以“== <比赛名>”开头（stages/evaluate/track2_rank_sum.py 读的格式），不截行。凭据由 kaggle CLI 自己读 ~/.kaggle/access_token。
# 某个榜取不到（超时、退出码非 0）或一行队伍都没有，就在最后以退出码 1 结束，record 会把这份记成 .failed。
set -uo pipefail
K=${1:?kaggle CLI 的路径}
rc=0
for t in 1 2 3; do
  for m in auc sp-sr mp-sr mppe; do
    s="v2d-challenge-track2-tier${t}-${m}"
    echo "== ${s}"
    out=$(perl -e 'alarm 90; exec @ARGV' "$K" competitions leaderboard "$s" --show 2>&1); r=$?
    printf '%s\n' "$out"
    if [ "$r" -ne 0 ] || ! printf '%s\n' "$out" | grep -qE '^ *[0-9]+ +.+ +[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9:.]+ +-?[0-9.]+ *$'; then
      echo "board_snapshot: ${s} 取不到（退出码 ${r}）" >&2
      rc=1
    fi
  done
done
exit "$rc"
