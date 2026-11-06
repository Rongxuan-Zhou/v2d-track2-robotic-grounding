#!/usr/bin/env bash
# 在训练机上读一条 Tier 3 run 的健康项与 4096 世界读数（只读）：冒烟与正式训练日志里 r3o 项那一行的个数、nanwatch 最后一行、
# 导出链的 EXPORT_CHAIN_DONE、4096 世界评测的 non_finite_worlds 与 SP-SR / MP-SR / ADD-AUC / MPPE。最后一行 T3_HEALTH_DONE。
# 用法（Mac 上）：ssh <机器> 'cd <仓库根> && bash -s <run 名> <段名>' < tools/t3_health_remote.sh
set -u
R=$1; SEG=$2
echo "== $R"
for f in runs/logs/smoke-$R.s1.log runs/logs/$R.s1.log; do echo "r3o 项 $(grep -c 'object_orientation term' "$f") 行：$f"; done
tail -n 1 "runs/logs/$R.nanwatch.log"
grep -h EXPORT_CHAIN_DONE runs/logs/cand_t3_${SEG}_*.log | tail -n 1
J=runs/flash_chord/$R/eval_349999104_w4096.json
python3 - "$J" <<'PY' 2>/dev/null || grep -o -E '"(non_finite_worlds|spider_sr_uncentered|maniptrans_sr|add_auc|mppe_cm)": *[^,]*' "$J"
import json, sys
d = json.load(open(sys.argv[1])); m = d["metrics"]
print(f"non_finite_worlds {len(d.get('non_finite_worlds') or [])}；SP-SR {m['spider_sr_uncentered']:.3f}，MP-SR {m['maniptrans_sr']:.3f}，ADD-AUC {m['add_auc']:.3f}，MPPE {m['mppe_cm']:.2f} cm；checkpoint {d['checkpoint_sha256'][:12]}")
PY
echo T3_HEALTH_DONE
