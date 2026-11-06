#!/usr/bin/env bash
# 两档 × 30 条批量：转换 → 重定向 → 对齐检查。训练机上执行。
# 并发 3、每个容器 4 线程：22 核共用机器，给别人留一半。
# 用法：stages/retarget/batch.sh [档位...]   默认 1 2
set -uo pipefail
cd "$(dirname "$0")/../.."
TIERS=${*:-1 2}
L=runs/logs/batch; mkdir -p "$L"
for t in $TIERS; do for e in $(seq 0 29); do echo "$t $e"; done; done | xargs -P 3 -n 2 stages/retarget/one.sh
{ printf "tier\tep\tsequence_id\tframes\tfps\tcontact_L\tcontact_R\tik_med\tgap_mm\tsurface\tstatus\n"; cat "$L"/t*_ep*.tsv | sort -n -k1,1 -k2,2; } > "$L/summary.tsv"
echo "== 汇总 =="; column -t -s $'\t' "$L/summary.tsv" | cut -c1-160
echo; awk -F'\t' 'NR>1 {n[$1]++; if ($11=="OK") ok[$1]++} END {for (t in n) printf "Tier %s: %d/%d OK\n", t, ok[t], n[t]}' "$L/summary.tsv"
