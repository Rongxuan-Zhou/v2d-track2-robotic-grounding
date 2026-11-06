#!/usr/bin/env bash
# 跑分析脚本并把 stdout 连同溯源信息落盘到 results/<name>.txt。
#
# 存在的理由：此前的结论只出现在终端里，脚本虽在但没有输出记录，
# 写技术报告时无法复现任何一个报过的数。每个结果文件自带产生它的
# 提交号、时间和脚本指纹，便于事后对账。
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=results; mkdir -p $OUT
SHA=$(git rev-parse --short HEAD 2>/dev/null || echo nogit)
DIRTY=$(git diff --quiet 2>/dev/null && echo clean || echo dirty)

FAST="baseline_t2 decompose smooth_baseline static_window static_from_t2 static_signal velocity_probe denoise_v1 denoise_v2"
FK="fk_shape fk_probe hand_error foot_signal foot_slip contact_veto denoise_v3 denoise_v4 union_diag bone_flex_probe bone_flex_physics oracle_root grasp_rigidity support_bridge support_bridge_v2 overfit_audit residual_v51 increment_noise st_tune fusion_ceiling nopen_eval nopen_jitter cleanroom_params"

case "${1:-all}" in
  fast) LIST="$FAST" ;;
  fk)   LIST="$FK" ;;
  all)  LIST="$FAST $FK" ;;
  *)    LIST="$*" ;;          # 也可直接点名若干脚本
esac

FAILED=""
for n in $LIST; do
  f=stages/recovery/$n.py
  [ -f "$f" ] || { echo "跳过(不存在): $n"; continue; }
  echo "=== $n ==="
  t0=$(date +%s)
  {
    echo "# script   : $f"
    echo "# md5      : $(md5sum "$f" | cut -d' ' -f1)"
    echo "# commit   : $SHA ($DIRTY)"
    echo "# run_at   : $(date -Is)"
    echo "# ---------------------------------------------------------------"
  } > "$OUT/$n.txt"
  python3 "$f" > /tmp/_out_$n 2>&1; rc=$?
  grep -avE "Warning|warn|Module soma|^Warp |CUDA Toolkit|Devices:|x86_64|cuda:0|Kernel cache|cache/warp" \
    /tmp/_out_$n >> "$OUT/$n.txt"; rm -f /tmp/_out_$n
  dt=$(( $(date +%s) - t0 ))
  if [ $rc -ne 0 ] || grep -q "Traceback" "$OUT/$n.txt"; then
    FAILED="$FAILED $n"
    echo "    !! 失败 rc=$rc  -> $OUT/$n.txt"
    tail -3 "$OUT/$n.txt" | sed "s/^/       /"
  else
    echo "    -> $OUT/$n.txt  (${dt}s, $(wc -l < "$OUT/$n.txt") 行)"
  fi
done
if [ -n "$FAILED" ]; then echo "失败:$FAILED"; echo "ANALYSES_FAILED"; exit 1; fi
echo "ANALYSESDONE"
