#!/usr/bin/env bash
# 方盒三条（ep1、ep7、ep28）某一档的参考，在 #185 几何（Cube 按 size × scale 建）下与支撑面、地面的穿透（训练机宿主机上、仓库根执行，
# 只读：numpy、pyarrow）。计划 agent-doc/plan-box6-and-eval-runtime-2026-10-01.md 第 7 节第 3 步；Tier 1 的结果 9/30 已有
# （results/diverge/support_penetration.txt）。按 stages/retarget/support_penetration.py 要的目录结构，在 runs/diag/box_refs_t<档位>/<序列>/
# 下用符号链接放三个文件：这一档的 motion parquet、这一档的物体网格、这一档的支撑面拷贝。只读这一档自己的文件。
# 用法：stages/retarget/box_ref_check.sh <档位 1|2>
#   落盘：scripts/dev/record box/support_penetration_t2_<机器> stages/retarget/box_ref_check.sh 2
set -euo pipefail
cd "$(dirname "$0")/../.."
T=${1:?用法：box_ref_check.sh <档位 1|2>}
[[ $T =~ ^[12]$ ]] || { echo "档位只能是 1 或 2" >&2; exit 2; }
D="runs/diag/box_refs_t$T"; mkdir -p "$D"
dirs=()
for seq in 2026-05-18_17-06-21_vacuum_left_lean_02 2026-05-29_15-20-43_iron_round_table_02 2026-05-19_11-27-07_squeegee_sweep_away_from_you_03_; do
  mkdir -p "$D/$seq"
  for pair in "runs/motion/tier$T/whole_body/soma/sequence_id=$seq/robot_name=g1/data.parquet:data.parquet" \
              "runs/motion_src/tier$T/$seq/object/textured_mesh.obj:textured_mesh.obj" \
              "runs/motion/tier$T/whole_body/reconstructed_stage/${seq}_support.usda:${seq}_support.usda"; do
    src="${pair%%:*}"; dst="${pair##*:}"
    [ -f "$src" ] || { echo "缺 $src" >&2; exit 1; }
    ln -sfn "$PWD/$src" "$D/$seq/$dst"
  done
  echo "md5 $(md5sum "runs/motion/tier$T/whole_body/reconstructed_stage/${seq}_support.usda" | cut -c1-12)… ${seq}_support.usda（Tier ${T}）"
  dirs+=("$D/$seq")
done
python3 stages/retarget/support_penetration.py "${dirs[@]}"
