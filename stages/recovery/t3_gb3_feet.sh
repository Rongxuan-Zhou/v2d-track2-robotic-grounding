#!/usr/bin/env bash
# gb3 参考的人体支撑脚离发布地面的高度（ml40 上、仓库根执行；只用 CPU；只读另一位用户的代码、input、convert 与地面拟合）。
# 计划 agent-doc/plan-t3-gb3-switch-2026-10-17.md 判据 G0（第 2 版，复核 MAJOR 2：ep22 换了重建版本 convert_v2，k-fix 的人脚检验没有覆盖它）。
# 用她的 stages/recovery/tier3/t3_feet_diag.py（同 stages/recovery/t3_kfix_feet.sh；逐帧取两只脚鞋底顶点里最低的、对平面的高度，再取中位）。
# 那个脚本只读 <根>/runs/convert，这里给每种 convert 建一个影子根：容器里 /t/runs/convert 只读挂她的 runs/<convert>，/t/input 只读挂她的 input，
# 地面拟合用她的 runs/floor/floor_fit_gb3.json（gb3 的世界步，交付说明：地面校正 1.222°/0.984）。gb3 的人体就是所选 convert 的世界步结果
# （dr2s、dyn7 只改物体位姿一列，交付说明自检第 11 项），所以这里量的就是 gb3 参考里的人体。
# 只量这一轮训练的 8 集所在的两种 convert：convert_gb3（ep0、8、20、21、29）与 convert_v2_gb3（ep6、7、22）；脚本对每种 convert 下有世界步结果的集都会算。
# 目录自己找（同 tier3_pack_remote.sh）：/lab_data/*/v2d/tier3 与 /lab_data/*/v2d/soma_assets 里不属于本用户的，各恰好一个；输出只有容器内路径。
# 输出：每集一行 JSON（她的脚本打印的，feet_world_m 是世界步之后的支撑脚高度中位，米）；结果文件写到 /lab_data/<本用户>/v2d/tier3_gb3feet/。
# 最后一行 GB3_FEET_DONE 或 GB3_FEET_FAILED。
# 用法（ml40，仓库根）：setsid nohup scripts/dev/record t3gb3/feet_gb3_ml40 bash stages/recovery/t3_gb3_feet.sh > runs/logs/t3gb3_feet_ml40.log 2>&1 < /dev/null &
set -u
me="$(id -un)"
mapfile -t roots < <(ls -d /lab_data/*/v2d/tier3 2>/dev/null | grep -v "^/lab_data/$me/")
mapfile -t assets < <(ls -d /lab_data/*/v2d/soma_assets 2>/dev/null | grep -v "^/lab_data/$me/")
[ ${#roots[@]} = 1 ] && [ ${#assets[@]} = 1 ] || { echo "别的用户的 tier3 或 soma_assets 目录不是恰好一个"; echo GB3_FEET_FAILED; exit 1; }
SRC=${roots[0]}; ASSETS=${assets[0]}; OUT=/lab_data/$me/v2d/tier3_gb3feet
[ -r "$SRC/runs/floor/floor_fit_gb3.json" ] || { echo "读不了 runs/floor/floor_fit_gb3.json"; echo GB3_FEET_FAILED; exit 1; }
mkdir -p "$OUT"
D=(docker run --rm --cpus 4 --user "$(id -u):$(id -g)" --entrypoint "" --network none -e HOME=/tmp -w /workspace/v2d_cari4d/lib/cari4d)
fail=0
for conv in convert_gb3 convert_v2_gb3; do
  [ -d "$SRC/runs/$conv" ] || { echo "没有 runs/$conv"; fail=1; continue; }
  echo "== ${conv}（地面拟合 runs/floor/floor_fit_gb3.json）"
  "${D[@]}" -v "$SRC/code":/c:ro -v "$SRC/runs/$conv":/t/runs/convert:ro -v "$SRC/input":/t/input:ro \
    -v "$SRC/runs/floor/floor_fit_gb3.json":/f/fit.json:ro -v "$ASSETS":/a:ro -v "$OUT":/o v2d_cari4d:latest \
    python /c/stages/recovery/tier3/t3_feet_diag.py /t /f/fit.json /a "/o/feet_${conv}.json" > "$OUT/feet_${conv}.log" 2>&1 \
    || { echo "${conv} 的脚步诊断失败（退出码 $?，日志 tier3_gb3feet/feet_${conv}.log 的最后几行如下）"; tail -5 "$OUT/feet_${conv}.log" | sed "s#$SRC#<她的 tier3>#g"; fail=1; continue; }
  grep '^{' "$OUT/feet_${conv}.log" | cut -c1-300
done
[ "$fail" = 0 ] && echo GB3_FEET_DONE || echo GB3_FEET_FAILED
