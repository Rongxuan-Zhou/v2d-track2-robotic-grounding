#!/usr/bin/env bash
# Day 1 评测（在 ml40 上执行）。设置与判据事先登记在 agent-doc/experiments/2026-09-28-day1.md。
# 顺序：先用 ml40 冒烟 policy 在 256 个世界上试跑 fc_eval.sh（只验证包装能用，不进判据），
# 再按登记的设置（4096 世界、官方默认评测配置）评两条 run 的最终 policy。任何一步失败即停。
# mf4090 的 policy 经 Mac 中转拷来，来历见其目录下的 COPIED_FROM.json（sha256 三端一致）。
# 用法（仓库根）：tmux new-session -d -s v2d-eval stages/policy/queues/ml40_day1_eval.sh
# 日志：runs/logs/ml40_day1_eval.log；每次评测另有 runs/logs/<run>.eval_<步数>_w<世界数>.log
set -euo pipefail
cd "$(dirname "$0")/../../.."
mkdir -p runs/logs
exec >> runs/logs/ml40_day1_eval.log 2>&1
trap 'echo "=== $(date "+%F %T %Z") EVAL_CHAIN_FAILED (line $LINENO)"' ERR
echo "=== $(date '+%F %T %Z') start, source $(cat .v2d-source-commit 2>/dev/null || echo unknown)"
V2D_EVAL_WORLDS=256 stages/policy/fc_eval.sh smoke-t1-ep20-w2048-cap10m-ml40 4096000
stages/policy/fc_eval.sh t1-ep20-w2048-cap10m-ml40 349999104
stages/policy/fc_eval.sh t1-ep20-w2048-cap1m-mf4090 349999104
echo "=== $(date '+%F %T %Z') EVAL_CHAIN_DONE"
