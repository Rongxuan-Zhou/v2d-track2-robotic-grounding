#!/usr/bin/env bash
# 有意停掉一个训练（不触发自动重试）：先放 STOP 标记，再停容器。
# 之后重跑 fc_run.sh / fc_queue.sh 即从最近的 checkpoint 继续。
# 用法：stages/policy/fc_stop.sh <run>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"; cd "$ROOT"
RUN=${1:?用法：fc_stop.sh <run>}; OUT="runs/flash_chord/$RUN"
[ -d "$OUT" ] || { echo "没有 $OUT" >&2; exit 2; }
touch "$OUT/STOP"
if docker stop -t 60 "fc-$RUN" >/dev/null 2>&1; then
  echo "已停止 fc-$RUN；fc_run 会记录 stopped 并退出（不重试）"
else
  echo "容器 fc-$RUN 不在运行；STOP 标记已放，下次 fc_run 结束时生效" >&2
fi
