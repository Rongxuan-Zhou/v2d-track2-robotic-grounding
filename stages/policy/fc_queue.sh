#!/usr/bin/env bash
# 逐行跑队列文件里的训练（训练机上，建议在 tmux 里）。
# 每行：[KEY=VAL ...] <run> <parquet> <world_count> <total_steps> [hydra 覆盖...]；# 之后是注释。
# 已完成（有 DONE）的跳过；某一行失败或被 STOP 就停下，不往后跑。中断后重跑同一条命令即从断点继续。
# 用法：stages/policy/fc_queue.sh stages/policy/queues/<机器>_<名字>.txt
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"; cd "$ROOT"
Q=${1:?用法：fc_queue.sh <队列文件>}; [ -f "$Q" ] || { echo "没有 $Q" >&2; exit 2; }
n=0
while IFS= read -r line || [ -n "$line" ]; do
  line="${line%%#*}"; [[ -z "${line//[[:space:]]/}" ]] && continue
  read -r -a tok <<< "$line"
  envs=(); i=0
  while (( i < ${#tok[@]} )) && [[ ${tok[$i]} =~ ^[A-Z_][A-Z0-9_]*= ]]; do envs+=("${tok[$i]}"); i=$((i + 1)); done
  args=("${tok[@]:$i}"); n=$((n + 1))
  echo "=== $(date '+%F %T %Z') 队列第 $n 项：${args[0]} ==="
  env "${envs[@]}" bash stages/policy/fc_run.sh "${args[@]}" < /dev/null
  rc=$?
  [ "$rc" = 0 ] || { echo "=== 队列在 ${args[0]} 停下（rc=$rc）；修好后重跑同一条命令即从这里继续 ==="; exit "$rc"; }
done < "$Q"
echo "=== $(date '+%F %T %Z') 队列 $Q 全部完成 ==="
