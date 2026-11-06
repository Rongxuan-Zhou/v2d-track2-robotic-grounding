#!/usr/bin/env bash
# Tier 3 跟丢集的手-物诊断（训练机上执行，经 scripts/dev/in-flash-chord 在容器里跑，只用 CPU，只读参考 parquet；只报不判）。
# 计划 agent-doc/plan-t3hand-2026-10-27.md 第 0 节：对 5 集跟丢的“重建”类（ep13、15、16、19、27）与 3 集对照（ep3、12、23），
# 在 Tier 3 gb5 的重定向参考上逐帧量机器人手到物体的距离（ref_clearance，整集、每 3 帧）、接触帧里手插进物体多深（ref_penetration）、
# 接触标记与物体位移（ref_contact_window --summary）；同一集的 Tier 2 参考只作诊断的尺子（不是 Tier 3 方法的输入），同样量一遍。
# 用法：bash stages/retarget/t3hand_diag.sh   结果 results/ml40/t3hand/<工具>_<档>_ep<N>.txt，最后打 T3HAND_DIAG_DONE
set -uo pipefail
cd "$(dirname "$0")/../.."
declare -A SEQ=(
  [13]=2026-06-09_10-19-09_snack_box_desk_through_06
  [15]=2026-05-06_11-06-15_wet_floor_sign_pick_place_ground_08
  [16]=2026-03-12_15-06-36_spray_bottle_left_desk_ground_02
  [19]=2026-05-19_10-19-48_cane_swing_05
  [27]=2026-05-01_11-15-05_woven_basket_ground_desk_10
  [3]=2026-04-27_11-50-10_contigo_coffee_mug_handle_table_05
  [12]=2026-04-25_17-20-04_black_platform_turn_05
  [23]=2026-06-03_15-08-03_toy_stroller_back_forth_ground_04
)
fail=0
for ep in 16 19 13 15 27 3 12 23; do
  for tier in tier3_gb5 tier2; do
    pq="runs/motion/$tier/whole_body/soma/sequence_id=${SEQ[$ep]}/robot_name=g1/data.parquet"
    [ -f "$pq" ] || { echo "缺 $pq"; fail=$((fail + 1)); continue; }
    scripts/dev/record "ml40/t3hand/clearance_${tier}_ep${ep}" scripts/dev/in-flash-chord "python /repo/stages/retarget/ref_clearance.py /repo/$pq --from 0 --after 3000 --every 3" >/dev/null 2>&1 || fail=$((fail + 1))
    scripts/dev/record "ml40/t3hand/penetration_${tier}_ep${ep}" scripts/dev/in-flash-chord "python /repo/stages/retarget/ref_penetration.py --every 3 /repo/$pq" >/dev/null 2>&1 || fail=$((fail + 1))
    scripts/dev/record "ml40/t3hand/contact_${tier}_ep${ep}" scripts/dev/in-flash-chord "python /repo/stages/evaluate/ref_contact_window.py --summary /repo/$pq" >/dev/null 2>&1 || fail=$((fail + 1))
  done
done
echo "T3HAND_DIAG_DONE（失败 ${fail}）"
