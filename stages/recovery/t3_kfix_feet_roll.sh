#!/usr/bin/env bash
# k-fix 判据 4b：人脚横滚检验（ml40 上、仓库根执行；只用 CPU；只读影子根、Tier 3 input 与另一位用户的代码）。
# 计划 agent-doc/plan-t3-kfix-2026-10-16.md 判据 4b；脚本 stages/recovery/t3_kfix_feet_roll.py（Mac 上先跑过 --selftest）。
# 另一位用户的两个路径用环境变量给（同 t3_kfix_feet.sh）：T3_SRC、T3_SOMA_ASSETS；OUT 默认 /lab_data/rxzhou/v2d/tier3_kfix_r2，MODE 默认 F。
# 输出 $OUT/$MODE/feet_roll.json、feet_roll.frames.npz（逐帧 h、横向与纵深坐标、集号）与 feet_roll.log；最后一行 FEET_ROLL_DONE <判读> 或 FEET_ROLL_FAILED。
set -u
for v in T3_SRC T3_SOMA_ASSETS; do [ -n "${!v:-}" ] || { echo "t3_kfix_feet_roll: 要设 ${v}" >&2; exit 2; }; done
SRC=$T3_SRC; CODE=$SRC/code; ASSETS=$T3_SOMA_ASSETS
OUT=${OUT:-/lab_data/rxzhou/v2d/tier3_kfix_r2}; MODE=${MODE:-F}; R=$OUT/$MODE
REPO=$(pwd)
[ -f "$R/runs/floor/floor_fit_kt.json" ] || { echo "没有 ${R}/runs/floor/floor_fit_kt.json"; echo FEET_ROLL_FAILED; exit 1; }
# 线程数钉到 4，与 --cpus 4 一致：不钉时 OpenBLAS 按 28 核开线程，配额下互相等，第一次跑自助抽样 18 分钟没算完（10/16 06:53 停掉重跑）
docker run --rm --cpus 4 --user "$(id -u):$(id -g)" --entrypoint "" --network none -e HOME=/tmp -w /workspace/v2d_cari4d/lib/cari4d \
  -e OMP_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 -e MKL_NUM_THREADS=4 \
  -v "$CODE":/c:ro -v "$REPO/stages/recovery":/k:ro -v "$R":/t -v "$SRC/input":/t/input:ro -v "$ASSETS":/a:ro v2d_cari4d:latest \
  python /k/t3_kfix_feet_roll.py /t /t/runs/floor/floor_fit_kt.json /a /t/feet_roll.json > "$R/feet_roll.log" 2>&1 \
  || { echo "人脚横滚检验失败（${R}/feet_roll.log）"; tail -5 "$R/feet_roll.log"; echo FEET_ROLL_FAILED; exit 1; }
grep -v -E "^(Warp|   |/opt|  self|  result|Module)" "$R/feet_roll.log" | cut -c1-300
