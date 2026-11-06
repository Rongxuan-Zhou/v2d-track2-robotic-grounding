#!/usr/bin/env bash
# 同一策略、同一评测器（V2D_FC_SRC 指的副本）对两份参考各评一次 4096 世界，比指标（训练机上、仓库根执行，占 GPU 约 2 × 5 分钟）。
# 用途：t1c（接触标记提前，计划 agent-doc/plan-t1-contact-anticipation-2026-10-16.md）只改了训练用的 hand_contact_active，按设计评测应不受影响；
# 拿一条现有策略对 tier1 与 tier1_c10 两份参考各评一次，在 k-fix 用过的容差内一致（MP-SR、SP-SR 差 ≤ 0.03，ADD-AUC ≤ 0.006，MPPE ≤ 0.6 cm）
# 就记“评测不受接触标记影响”，同时把新参考的加载完整跑通一遍（独立复核 MINOR 4）。两次都经 eval_playback_override.py（恒等回放、只换 parquet），
# 包装层的影响两边相同。fc_export.sh 的 V2D_EXPORT_PARQUET 只收 Tier 3 的带后缀目录，所以这里直接调包装脚本。
# 用法：V2D_FC_SRC=<fc_8c778f9 副本> stages/evaluate/eval_parquet_pair.sh <run> <步数> <参考 A（/repo/...）> <参考 B（/repo/...）> <输出目录（runs/ 下）>
# 最后一行 EVAL_PAIR 一致 / 不一致 / 失败。
set -uo pipefail
cd "$(dirname "$0")/../.."
RUN=${1:?}; STEP=${2:?}; PA=${3:?}; PB=${4:?}; OUT=${5:?}
[ -n "${V2D_FC_SRC:-}" ] || { echo "eval_parquet_pair: 要设 V2D_FC_SRC" >&2; exit 2; }
[[ $OUT =~ ^runs/[A-Za-z0-9_./-]+$ && $OUT != *..* ]] || { echo "eval_parquet_pair: 输出目录要在 runs/ 下：$OUT" >&2; exit 2; }
P="runs/flash_chord/$RUN/policy_$STEP.safetensors"
[ -f "$P" ] || { echo "eval_parquet_pair: 没有 $P" >&2; exit 2; }
for q in "$PA" "$PB"; do [ -f "${q#/repo/}" ] || { echo "eval_parquet_pair: 没有 ${q#/repo/}" >&2; exit 2; }; done
others="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')"
[ -z "$others" ] || { echo "eval_parquet_pair: GPU 上有别的进程（${others//$'\n'/,}），不评" >&2; echo "EVAL_PAIR 失败"; exit 6; }
mkdir -p "$OUT"
for tag in A B; do
  q=$PA; [ "$tag" = B ] && q=$PB
  J="$OUT/$tag.json"; rm -f "$J" "$J.override.json"
  echo "=== $(date '+%F %T %Z') $tag：$q"
  V2D_GPU=1 V2D_CONTAINER_NAME="fc-pair-$RUN-$tag" scripts/dev/in-flash-chord \
    "cd /workspace && python /repo/stages/evaluate/eval_playback_override.py $(printf '%q ' "/repo/$P" "/repo/$J" --playback same --worlds 4096 --parquet "$q" --extra evaluation.reset_mode=explicit --extra logging.mode=disabled)" \
    > "$OUT/$tag.log" 2>&1
  rc=$?
  tail -3 "$OUT/$tag.log"
  [ "$rc" = 0 ] && [ -s "$J" ] || { echo "$tag 评测失败（退出码 ${rc}），见 $OUT/$tag.log"; echo "EVAL_PAIR 失败"; exit 3; }
done
python3 - "$OUT/A.json" "$OUT/B.json" "$PA" "$PB" <<'PY'
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
for d, want in ((a, sys.argv[3]), (b, sys.argv[4])):
    if d.get("parquet") != want:
        print(f"评测 JSON 记的参考 {d.get('parquet')} 不是 {want}"); print("EVAL_PAIR 失败"); sys.exit(0)
ma, mb = a["metrics"], b["metrics"]
keys = (("maniptrans_sr", "MP-SR", 0.03), ("spider_sr_uncentered", "SP-SR", 0.03), ("add_auc", "ADD-AUC", 0.006), ("mppe_cm", "MPPE", 0.6))
ok = True
for k, name, tol in keys:
    d = mb[k] - ma[k]; ok &= abs(d) <= tol
    print(f"  {name}：A {ma[k]:.4f}，B {mb[k]:.4f}，B − A {d:+.4f}（容差 {tol}）")
print(f"EVAL_PAIR {'一致' if ok else '不一致'}")
PY
