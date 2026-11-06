#!/usr/bin/env bash
# Tier 1 逐条选：把这台机器上导出状态为 ok 的候选文件抄到一个暂存目录，供经 Mac 管道集中到 ml40（训练机上、仓库根执行，只读源文件）。
# 计划 agent-doc/plan-t1-per-episode-select-2026-10-08.md 第 2 版第 3 节。
# 用法：stages/evaluate/stage_t1_candidates.sh <暂存目录> <导出链结果表.tsv>…
# 每个候选一个子目录 <暂存目录>/<run>/：episode_XXXXXX.json、episode_XXXXXX.parquet（结果表里的路径），以及 4096 世界评测
# runs/flash_chord/<run>/eval_349999104_w4096.json（有就抄，只报“运气”列用）。结果表也抄到 <暂存目录>/../tables/。
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=${1:?暂存目录}; shift
[ $# -ge 1 ] || { echo "stage_t1_candidates: 要至少一张结果表" >&2; exit 2; }
mkdir -p "$OUT" "$OUT/../tables"
n=0
for t in "$@"; do
  cp "$t" "$OUT/../tables/"
  while IFS=$'\t' read -r tier ep run step status js pq jsha psha; do
    [ "$tier" = 1 ] && [ "$status" = ok ] || continue
    mkdir -p "$OUT/$run"
    cp "$js" "$pq" "$OUT/$run/"
    e="runs/flash_chord/$run/eval_${step}_w4096.json"
    [ -f "$e" ] && cp "$e" "$OUT/$run/"
    n=$((n + 1))
  done < "$t"
done
echo "stage_t1_candidates: 抄了 $n 个候选到 $OUT（$(ls "$OUT" | wc -l) 个 run 目录）"
