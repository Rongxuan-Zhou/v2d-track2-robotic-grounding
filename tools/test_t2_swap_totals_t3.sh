#!/usr/bin/env bash
# t2_swap_totals.py --tier 3 的测试（Mac 上、仓库根执行，只读记录；计划 agent-doc/plan-t3auc1-2026-11-03.md 第 2 节）：
#   1. Tier 3 恒等换集（ep12 换成它自己）两种 AUC 平均的变化都是 0，导出 SP 16、MP 12 与 t3m1029a 打包记录（plan-t3pack 执行记录）相同；
#   2. Tier 2 不给 --tier 时输出与 results/t2auc2/swap_ep26_vs_t2a1102.txt 逐字相同。最后一行 ALL_OK。
set -uo pipefail
o=$(python3 tools/t2_swap_totals.py stages/evaluate/exports/t3_select_t3m1029a.tsv runs/submission/t3m1029a/t3 --tier 3 ep12=t3-ep12-w2048-cap10m-r3a-gb5b2n12hdyn7-ml40 2>&1)
grep -q "导出 +0.00000，4096 世界 +0.00000" <<<"$o" && grep -q "SP 成功 16 → 16 / 30，MP 成功 12 → 12 / 30" <<<"$o" && echo T3_IDENTITY_OK || { echo "$o" | tail -4; echo FAIL; exit 1; }
python3 tools/t2_swap_totals.py stages/evaluate/exports/t2_select_t2a1102.tsv runs/submission/select_t2_t2a1102/t2 ep26=t2-ep26-w2048-cap10m-r3o-ml40 2>&1 \
  | diff <(grep -v '^#' results/t2auc2/swap_ep26_vs_t2a1102.txt) - >/dev/null && echo T2_SAME_OK || { echo FAIL; exit 1; }
echo ALL_OK
