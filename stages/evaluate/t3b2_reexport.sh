#!/usr/bin/env bash
# t3b2：用现有 checkpoint 对新参考重新评测与导出（ml40 仓库根执行，放 setsid）。计划 agent-doc/plan-t3b2-2026-10-28.md 第 2 步第 6 条、第 3 步第 4 条。
# 每个 run：先评 4096 世界（只写指标），再导出 1 个世界（带轨迹）；都经 stages/evaluate/fc_export.sh（V2D_EXPORT_PARQUET 换参考，必须带后缀）。
# 用法：stages/evaluate/t3b2_reexport.sh <后缀> <新参考 parquet（容器内 /repo/... 路径）> <run>...   步数固定 349999104。
# 记录 results/ml40/t3b2/reexport_<run><后缀>_w{4096,1}.txt；日志 runs/logs/t3b2_reexport.log，最后一行 T3B2_REEXPORT_DONE <失败步数>。
set -uo pipefail
cd "$(dirname "$0")/../.."
SUF=${1:?后缀}; PQ=${2:?参考}; shift 2
[ $# -gt 0 ] || { echo "要至少一个 run" >&2; exit 2; }
LOG=runs/logs/t3b2_reexport.log
mkdir -p runs/logs
exec >> "$LOG" 2>&1
say() { echo "=== $(date '+%F %T %Z') $*"; }
export V2D_FC_SRC=/lab_data/rxzhou/v2d/fc_src/fc_8c778f9
[ -f "${PQ/#\/repo/.}" ] || { say "T3B2_REEXPORT_DONE 1（没有 ${PQ}）"; exit 1; }
say "开始，来历 $(cat .v2d-source-commit 2>/dev/null)；后缀 ${SUF}；参考 ${PQ}（sha256 $(sha256sum < "${PQ/#\/repo/.}" | cut -c1-16)）；run $*"
bad=0
for run in "$@"; do
  for w in 4096 1; do
    traj=0; [ "$w" = 1 ] && traj=1
    if V2D_EXPORT_PARQUET="$PQ" V2D_EXPORT_SUFFIX="$SUF" V2D_EXPORT_WORLDS=$w V2D_EXPORT_TRAJ=$traj \
         scripts/dev/record "ml40/t3b2/reexport_${run}${SUF}_w${w}" stages/evaluate/fc_export.sh "$run" 349999104 < /dev/null > /dev/null 2>&1; then
      say "${run} w${w} 过"
    else
      say "${run} w${w} 失败"; bad=$((bad + 1))
    fi
  done
done
say "T3B2_REEXPORT_DONE ${bad}"
