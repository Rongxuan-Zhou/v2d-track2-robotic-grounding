#!/usr/bin/env bash
# 在一次性 FlashCHORD 容器里跑 wb_summary.py（容器里有 wandb，不占 GPU）。训练机上执行。
# W&B key 只从 600 权限的 .env 里取这一项（同 fc_run.sh），经环境变量名转发进容器，值不进 argv 与日志。
# 用法：stages/policy/wb_summary.sh <W&B run id>... [--project v2d-track2]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"; cd "$ROOT"
WANDB_API_KEY="$(sed -n 's/^WANDB_API_KEY=//p' .env 2>/dev/null | tail -1)"
[ -n "$WANDB_API_KEY" ] || { echo "wb_summary: .env 里没有 WANDB_API_KEY" >&2; exit 2; }
export WANDB_API_KEY
exec scripts/dev/in-flash-chord "python /repo/stages/policy/wb_summary.py $(printf '%q ' "$@")"
