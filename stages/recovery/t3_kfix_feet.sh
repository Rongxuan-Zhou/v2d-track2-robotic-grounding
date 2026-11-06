#!/usr/bin/env bash
# k-fix 判据 4：人体支撑脚在世界步之后离发布地面的高度（ml40 上、仓库根执行；只用 CPU；只读另一位用户的代码、input 与交付链）。
# 计划 agent-doc/plan-t3-kfix-2026-10-16.md 判据 4：地面校正之后支撑脚高度中位的绝对值 ≤ 3 cm；交付链（她的 floor_fit_f830 世界步）作对照一并报。
# 用的是她的 stages/recovery/tier3/t3_feet_diag.py（只算做过世界步的集，逐帧取两只脚鞋底顶点里最低的、对平面的高度，再取中位），跑两遍：
#   1. k-fix：影子根 $OUT/<口径>（t3_kfix_run.sh 写的），地面 $OUT/<口径>/runs/floor/floor_fit_kt.json → $OUT/<口径>/feet_diag_kt.json；
#   2. 对照：她的 Tier 3 根（只读），地面 runs/floor/floor_fit_f830.json → $OUT/feet_diag_delivered.json（她 convert 目录里现有的世界步结果）。
# 她的三个路径用环境变量给（同 t3_kfix_run.sh）：T3_SRC、T3_SOMA_ASSETS；OUT 默认 /lab_data/rxzhou/v2d/tier3_kfix_r2，MODE 默认 F。
# 用法：T3_SRC=… T3_SOMA_ASSETS=… stages/recovery/t3_kfix_feet.sh   最后一行 KFIX_FEET_DONE 或 KFIX_FEET_FAILED
set -u
for v in T3_SRC T3_SOMA_ASSETS; do [ -n "${!v:-}" ] || { echo "t3_kfix_feet: 要设 $v" >&2; exit 2; }; done
SRC=$T3_SRC; CODE=$SRC/code; ASSETS=$T3_SOMA_ASSETS
OUT=${OUT:-/lab_data/rxzhou/v2d/tier3_kfix_r2}; MODE=${MODE:-F}; R=$OUT/$MODE
[ -f "$R/runs/floor/floor_fit_kt.json" ] || { echo "没有 $R/runs/floor/floor_fit_kt.json"; echo KFIX_FEET_FAILED; exit 1; }
D=(docker run --rm --cpus 4 --user "$(id -u):$(id -g)" --entrypoint "" --network none -e HOME=/tmp -w /workspace/v2d_cari4d/lib/cari4d)
"${D[@]}" -v "$CODE":/c:ro -v "$R":/t -v "$SRC/input":/t/input:ro -v "$ASSETS":/a:ro v2d_cari4d:latest \
  python /c/stages/recovery/tier3/t3_feet_diag.py /t /t/runs/floor/floor_fit_kt.json /a /t/feet_diag_kt.json > "$R/feet_diag_kt.log" 2>&1 \
  || { echo "k-fix 脚步诊断失败（$R/feet_diag_kt.log）"; echo KFIX_FEET_FAILED; exit 1; }
echo "== k-fix（口径 $MODE）："; grep '^{' "$R/feet_diag_kt.log" | cut -c1-300
"${D[@]}" -v "$CODE":/c:ro -v "$SRC":/t:ro -v "$ASSETS":/a:ro -v "$OUT":/o v2d_cari4d:latest \
  python /c/stages/recovery/tier3/t3_feet_diag.py /t /t/runs/floor/floor_fit_f830.json /a /o/feet_diag_delivered.json > "$OUT/feet_diag_delivered.log" 2>&1 \
  || { echo "对照脚步诊断失败（$OUT/feet_diag_delivered.log）"; echo KFIX_FEET_FAILED; exit 1; }
echo "== 对照（交付链现有的世界步）："; grep '^{' "$OUT/feet_diag_delivered.log" | cut -c1-300
echo KFIX_FEET_DONE
