#!/usr/bin/env bash
# Tier 3 初始化试点的核对（ml40 上、仓库根执行；只用 CPU；只读另一位用户的 tier3 目录与代码，结果写到本用户的 OUT）。
# 判读规则见 agent-doc/experiments/2026-10-16-t3-init-pilot-check.md：
#   A、C：stages/recovery/t3_init_compare.py（两版 CARI4D 原始输出的逐帧位姿比）→ $OUT/compare.jsonl
#   B：另一位用户的 t3_object_check.py（剪影对 SAM2 物体掩码）在 work_fp、work_init 两版上各跑一遍，步长 3 → $OUT/<版本>_<集>/
# 另一位用户的 tier3 根用环境变量 T3_SRC 给；OUT 默认 /lab_data/rxzhou/v2d/t3_init_check。最后一行 T3_INIT_CHECK_DONE 或 T3_INIT_CHECK_FAILED。
# 用法：T3_SRC=… stages/recovery/t3_init_check.sh ep13_snack_box ep28_squeegee ep04_baseball_bat ep03_contigo_coffee_mug
set -u
[ -n "${T3_SRC:-}" ] || { echo "t3_init_check: 要设 T3_SRC" >&2; exit 2; }
[ $# -ge 1 ] || { echo "t3_init_check: 要至少一集" >&2; exit 2; }
SRC=$T3_SRC; OUT=${OUT:-/lab_data/rxzhou/v2d/t3_init_check}; REPO=$(pwd)
mkdir -p "$OUT" || exit 1
D=(docker run --rm --cpus 4 --user "$(id -u):$(id -g)" --entrypoint "" --network none -e HOME=/tmp
   -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -w /workspace/v2d_cari4d/lib/cari4d)
echo "== A、C：两版位姿逐帧比"
"${D[@]}" -v "$SRC":/t:ro -v "$REPO/stages/recovery":/k:ro v2d_cari4d:latest python /k/t3_init_compare.py /t "$@" > "$OUT/compare.jsonl" 2> "$OUT/compare.log" \
  || { echo "比对失败（${OUT}/compare.log）"; tail -5 "$OUT/compare.log"; echo T3_INIT_CHECK_FAILED; exit 1; }
cat "$OUT/compare.jsonl"
echo "== B：剪影检查，两版各一遍（每集两版并行）"
fail=0
for name in "$@"; do
  seq=$(printf episode_%06d "$((10#${name:2:2}))")
  obj=${name:5}
  [ -f "$SRC/input/mesh/$obj/$obj.glb" ] || { echo "${name}：没有 input/mesh/${obj}/${obj}.glb"; fail=1; continue; }
  for w in work_fp work_init; do
    mkdir -p "$OUT/${w}_$name"
    "${D[@]}" -v "$SRC":/t:ro -v "$OUT/${w}_$name":/o v2d_cari4d:latest \
      python /t/code/stages/recovery/tier3/t3_object_check.py "/t/runs/$w/$name/cari4d/$seq" "/t/runs/$w/$name" "/t/input/mesh/$obj/$obj.glb" /o 3 \
      > "$OUT/${w}_$name.log" 2>&1 &
  done
  wait || true
  for w in work_fp work_init; do
    if ls "$OUT/${w}_$name"/*.json > /dev/null 2>&1; then echo "${w} ${name}：$(tail -c 600 "$OUT/${w}_$name.log" | tr '\n' ' ' | cut -c1-500)"
    else echo "${w} ${name}：没有输出（${OUT}/${w}_${name}.log）"; tail -3 "$OUT/${w}_$name.log"; fail=1; fi
  done
done
[ "$fail" = 0 ] && echo T3_INIT_CHECK_DONE || echo T3_INIT_CHECK_FAILED
