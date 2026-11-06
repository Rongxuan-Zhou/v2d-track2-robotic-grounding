#!/usr/bin/env bash
# 逐集比两条 run 的队列行：去掉 V2D_WANDB_TAG 后按词比，打出只在一边出现的词（计划 agent-doc/plan-t1r3o-2026-10-31.md 第 2 版）。
# 同名 run 出现在几份队列里时打出每份文件；去掉标签后不止一种行就退出 1（不静默取第一行）。
# 用法：bash tools/queue_line_diff.sh <新 run> <对照 run>（在仓库根，在 stages/policy/queues/*.txt 里找）
set -u
hits() { grep -H -E "(^| )$1( |$)" stages/policy/queues/*.txt | grep -v ':#'; }
line() {
  local h; h=$(hits "$1")
  [ -n "$h" ] || { echo "找不到队列行：$1" >&2; return 1; }
  echo "$h" | cut -d: -f1 | sed "s#^#  $1 见 #" >&2
  local n; n=$(echo "$h" | cut -d: -f2- | sed -E 's/V2D_WANDB_TAG=[^ ]*//' | sort -u | wc -l)
  [ "$n" -eq 1 ] || { echo "$1 在队列里有 $n 种不同的行" >&2; return 1; }
  echo "$h" | head -1 | cut -d: -f2- | tr ' ' '\n' | grep -v -E '^V2D_WANDB_TAG=|^$'
}
a=$(line "$1") || exit 1; b=$(line "$2") || exit 1
echo "== $1 对 $2"
diff <(echo "$a" | sed "s#^$1\$#RUN#" | sort) <(echo "$b" | sed "s#^$2\$#RUN#" | sort) | grep -E '^[<>]' || echo "（相同）"
