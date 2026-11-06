#!/usr/bin/env bash
# Mac 上用：在训练机上跑 stages/policy/run_pace.py（脚本经 stdin 传过去，不需要同步仓库，批量训练期间也能用）。
# 主机按 V2D_ENV 选（同 scripts/dev/remote）。落盘用 scripts/dev/record：
#   V2D_ENV=ml40 scripts/dev/record batch1/pace_ml40 stages/policy/run_pace.sh runs/logs/batch1a_train_ml40.log t1-ep0-w2048-cap10m-ml40 …
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"
. "$ROOT/scripts/dev/_load-env" >/dev/null
[ $# -ge 2 ] || { echo "用法：run_pace.sh <训练链日志> <run 名>..." >&2; exit 2; }
for a in "$@"; do
  [[ $a =~ ^[A-Za-z0-9_./-]+$ && $a != *..* ]] || { echo "run_pace: 参数非法：$a" >&2; exit 2; }
done
PY="$ROOT/stages/policy/run_pace.py"
echo "# run_pace.py md5 $( (md5sum "$PY" 2>/dev/null || md5 -r "$PY") | cut -d' ' -f1)；主机 $V2D_HOST"
ssh -o ConnectTimeout=20 "$V2D_HOST" "cd '$V2D_REMOTE' && python3 - $*" < "$PY"
