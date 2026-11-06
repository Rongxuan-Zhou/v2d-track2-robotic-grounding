#!/usr/bin/env bash
# 一份选择清单（档位 episode run 步数 机器）里每集那条 run 的 4096 世界读数：ADD-AUC、MP-SR、SP-SR、MPPE（Mac 上、仓库根执行，只读记录）。
# 评测记录在 results/day2/、results/ml40/day2/、results/l40n/day2/ 下的 eval_<run>_s<步数>.txt，取其中的 summary_json 行。
# 按 ADD-AUC 从低到高排。找不到记录的行写 missing。用法：bash tools/select_4096_table.sh <清单.tsv>
set -uo pipefail
SEL=${1:?用法：select_4096_table.sh <清单.tsv>}
printf 'ep\trun\tadd_auc\tmaniptrans_sr\tspider_sr\tmppe_cm\n'
while IFS=$'\t' read -r tier ep run step machine; do
  f=""
  for d in results/day2 results/ml40/day2 results/l40n/day2; do
    [ -f "$d/eval_${run}_s${step}.txt" ] && { f="$d/eval_${run}_s${step}.txt"; break; }
  done
  if [ -z "$f" ]; then printf '%s\t%s\tmissing\n' "$ep" "$run"; continue; fi
  grep -m1 '^ *summary_json ' "$f" | sed 's/^ *summary_json //' | python3 -c '
import json, sys
j = json.load(sys.stdin)
print("\t".join([sys.argv[1], sys.argv[2]] + ["%.3f" % j[k] for k in ("add_auc", "maniptrans_sr", "spider_sr_uncentered", "mppe_cm")]))
' "$ep" "$run"
done < "$SEL" | sort -t$'\t' -k3,3g
