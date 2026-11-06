#!/usr/bin/env bash
# Tier 3 k-fix 的 CPU 部分（ml40 上、仓库根执行；只用 CPU，不碰 GPU，不打包，不提交）。计划 agent-doc/plan-t3-kfix-2026-10-16.md 方案 1、2，判据 1、2。
# 另一位用户的 Tier 3 目录只读挂载（代码、convert、input、地面事件），一切输出写在我们自己的 OUT 下，每个深度口径一套“影子根目录”：
#   $OUT/<口径>/runs/convert/<集>/   t3_kfix.py 的输出（焦距步的文件名），再由她的世界步写 object_world.npz 等
#   $OUT/<口径>/runs/camera_groups/  只放 camera_groups.json 与 camera_height_main.json 的副本（第二、侧机位组没有事件，她的地面拟合按“不校正”处理）
#   $OUT/<口径>/runs/floor/floor_fit_kt.json   她的 t3_floor_fit.py 在新表达上重拟的地面校正
#   $OUT/<口径>/runs/tier3_kfix_<口径>/       她的装配写出的档目录（只有 BATCH 里的集）
# 步骤：1. 主机位 24 集按两种口径重新表达（物体与人体）；2. 每种口径重拟地面校正；3. BATCH 里的集跑世界步；4. 装配。
# 用法：T3_SRC=… T3_PY=… T3_SOMA_ASSETS=… setsid nohup stages/recovery/t3_kfix_run.sh > /dev/null 2>&1 < /dev/null &
#   （日志 $OUT/kfix_run.log，最后一行 KFIX_RUN_DONE 或 KFIX_RUN_FAILED）
#   KT=stages/recovery/kfix/kt_gb.json  BATCH="ep00_tall_bar_stool ep18_blue_trash_can …"  MODES="F Z"  OUT=/lab_data/rxzhou/v2d/tier3_kfix
set -u
REPO=$(pwd)
# 另一位用户的三个路径不写进仓库（记录与提交里不出现成员名字），启动时用环境变量给，缺一个就拒绝：
#   T3_SRC=她的 Tier 3 根目录（只读）  T3_PY=她的宿主 venv 的 python（只跑装配）  T3_SOMA_ASSETS=SOMA 资产目录
for v in T3_SRC T3_PY T3_SOMA_ASSETS; do [ -n "${!v:-}" ] || { echo "t3_kfix_run: 要设 $v" >&2; exit 2; }; done
SRC=$T3_SRC
CODE=$SRC/code                                            # 她的部署副本，DEPLOYED 在这里
PY=$T3_PY
ASSETS=$T3_SOMA_ASSETS
OUT=${OUT:-/lab_data/rxzhou/v2d/tier3_kfix}
KT=${KT:-stages/recovery/kfix/kt_gb.json}
MODES=${MODES:-F Z}
BATCH=${BATCH:-ep00_tall_bar_stool ep18_blue_trash_can ep20_rice_cooker ep22_step_ladder ep24_watering_can ep26_skateboard ep29_rolling_luggage}
SECOND_SIDE=" ep12_ ep14_ ep16_ ep21_ ep25_ ep28_ "
LOG=$OUT/kfix_run.log
mkdir -p "$OUT"
exec >> "$LOG" 2>&1
fail() { echo "=== $(date '+%F %T %Z') $*"; echo KFIX_RUN_FAILED; exit 1; }
echo "=== $(date '+%F %T %Z') 开始；仓库 $(git -C "$REPO" rev-parse --short HEAD)；她的部署 $(cat "$CODE/DEPLOYED")；KT $KT（sha256 $(sha256sum "$KT" | cut -c1-16)）"
[ -f "$KT" ] || fail "没有 $KT"
D=(docker run --rm --cpus 4 --user "$(id -u):$(id -g)" --entrypoint "" --network none -e HOME=/tmp -w /workspace/v2d_cari4d/lib/cari4d)
mains=()
for conv in "$SRC"/runs/convert/ep*; do
  name=$(basename "$conv")
  [[ "$SECOND_SIDE" == *" ${name:0:5} "* ]] && continue
  [ -f "$conv/soma_params.npz" ] && [ -f "$conv/object_cam.npz" ] || continue
  mains+=("$name")
done
echo "主机位 ${#mains[@]} 集：${mains[*]}"
[ "${#mains[@]}" = 24 ] || fail "主机位应是 24 集，找到 ${#mains[@]}"
for mode in $MODES; do
  R=$OUT/$mode
  [ -e "$R/runs/convert" ] && fail "$R/runs/convert 已存在（不覆盖）"
  mkdir -p "$R/runs/convert" "$R/runs/floor" "$R/runs/camera_groups"
  cp "$SRC/runs/camera_groups/camera_groups.json" "$SRC/runs/camera_groups/camera_height_main.json" "$R/runs/camera_groups/"
  for name in "${mains[@]}"; do   # 1. 重新表达
    "${D[@]}" -v "$CODE":/c:ro -v "$REPO/stages/recovery":/k:ro -v "$REPO/$(dirname "$KT")":/kt:ro -v "$SRC/runs/convert/$name":/conv/"$name":ro \
      -v "$ASSETS":/a:ro -v "$R/runs/convert":/o v2d_cari4d:latest \
      python /k/t3_kfix.py /conv/"$name" /o/"$name" /kt/"$(basename "$KT")" /a --mode "$mode" > "$R/express_$name.log" 2>&1 \
      && echo "$mode $name: $(tail -2 "$R/express_$name.log" | head -1 | cut -c1-200)" || fail "$mode $name：重新表达失败（$R/express_$name.log）"
  done
  # 2. 地面校正：她的 t3_floor_fit.py，影子根目录里 input 与 moge_ground 只读挂她的
  "${D[@]}" -v "$CODE":/c:ro -v "$R":/t -v "$SRC/input":/t/input:ro -v "$SRC/runs/moge_ground":/t/runs/moge_ground:ro v2d_cari4d:latest \
    python /c/stages/recovery/tier3/t3_floor_fit.py /t /t/runs/floor/floor_fit_kt.json > "$R/floor_fit.log" 2>&1 \
    || fail "$mode：地面拟合失败（$R/floor_fit.log）"
  echo "$mode 地面拟合：$(grep '^{' "$R/floor_fit.log" | head -1 | cut -c1-400)"
  for name in $BATCH; do   # 3. 世界步  4. 装配
    ep=$((10#${name:2:2}))
    "${D[@]}" -v "$CODE":/c:ro -v "$R/runs/floor":/f:ro -v "$ASSETS":/a:ro -v "$R/runs/convert/$name":/o/"$name" v2d_cari4d:latest \
      python /c/stages/recovery/tier3/t3_convert_ep.py world /o/"$name" /f/floor_fit_kt.json /a > "$R/world_$name.log" 2>&1 \
      || fail "$mode $name：世界步失败（$R/world_$name.log）"
    "$PY" "$CODE/stages/recovery/tier3/t3_convert_ep.py" assemble "$R/runs/convert/$name" "$SRC/input" "$ep" "$R/runs/tier3_kfix_$mode" \
      > "$R/assemble_$name.log" 2>&1 || fail "$mode $name：装配失败（$R/assemble_$name.log）"
    echo "$mode $name：世界步与装配完成 $(grep -o '"tilt_deg": [0-9.]*\|"scale": [0-9.]*' "$R/world_$name.log" | head -2 | tr '\n' ' ')"
  done
done
echo "=== $(date '+%F %T %Z') 完成"
echo KFIX_RUN_DONE
