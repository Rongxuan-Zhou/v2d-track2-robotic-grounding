#!/usr/bin/env bash
# 单条序列：转换 → 重定向（v0.4.0 + SOMAWithFlex，CPU 一次性容器）→ 对齐检查。训练机上执行。
# 结果写 runs/logs/batch/t<N>_ep<EE>.tsv 一行：tier ep seq 帧数 fps 左手接触 右手接触 IK中位 间隙mm 支撑面 状态
# Tier 3 设了 V2D_T3_SET=<名> 时，目录名用 tier3_<名>、日志名用 t3_<名>_ep<EE>（libs/v2d_infra/paths.py 的 tier_dirname），不覆盖旧版。
# Tier 2 的物体位姿来自 recovery 落盘产物（只经文件交接），目录由 T2_POSES 指定，默认 tier2_v54（净室参数；v5.3 的参数来自 Tier 1，不再作默认）。
set -uo pipefail
TIER=$1; EP=$2
cd "$(dirname "$0")/../.."
DN=$(python3 -c "import sys; sys.path.insert(0, 'libs/v2d_infra'); from paths import tier_dirname; print(tier_dirname($TIER))") || exit 2
L=runs/logs/batch; mkdir -p "$L"
TAG=t${TIER}${DN#tier$TIER}_ep$(printf %02d "$EP"); LOG=$L/$TAG.log; RES=$L/$TAG.tsv; : > "$LOG"
SEQ=$(python3 -c "
import json,sys; sys.path.insert(0,'libs/v2d_infra'); from paths import tier
print({json.loads(l)['episode_index']:json.loads(l) for l in open(tier($TIER)/'meta'/'episodes_metadata.jsonl')}[$EP]['sequence_id'])")
fail() { printf "%s\t%s\t%s\t\t\t\t\t\t\t\tFAIL:%s\n" "$TIER" "$EP" "$SEQ" "$1" > "$RES"; exit 1; }

OP=(); [ "$TIER" = 2 ] && OP=(--object-poses "runs/recovery/${T2_POSES:-tier2_v54}/episode_$(printf %06d "$EP").npz")
python3 stages/retarget/challenge_to_soma.py --tier "$TIER" --episode "$EP" "${OP[@]}" >> "$LOG" 2>&1 || fail convert

MR=runs/motion/$DN
rm -f "$MR/whole_body/reconstructed_stage/${SEQ}_g1_support.usda"   # 旧版重定向时代重建的，会被优先加载且已不对齐
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 scripts/dev/incontainer stages/retarget/run_soma_to_g1.py \
  "/repo/runs/motion_src/$DN/$SEQ" --save --motion-root "/repo/$MR" >> "$LOG" 2>&1 || fail retarget

PART=$MR/whole_body/soma/sequence_id=$SEQ/robot_name=g1
CHK=$(OMP_NUM_THREADS=2 scripts/dev/incontainer stages/retarget/check_support_alignment.py "/repo/$PART" 2>&1 | tee -a "$LOG" | grep -a "^RESULT")
GAP=$(sed -n 's/.*gap_mm=\([^ ]*\).*/\1/p' <<<"$CHK"); SURF=$(sed -n 's/.*surface=\([^ ]*\).*/\1/p' <<<"$CHK")
OK=$(sed -n 's/.*ok=\([01]\).*/\1/p' <<<"$CHK")
CL=$(grep -ao "left_hand_contact_active: [0-9]*" "$LOG" | tail -1 | grep -o "[0-9]*$")
CR=$(grep -ao "right_hand_contact_active: [0-9]*" "$LOG" | tail -1 | grep -o "[0-9]*$")
read -r NF FPS IK < <(python3 -c "
import pyarrow.parquet as pq, numpy as np
r=pq.read_table('$PART/data.parquet').to_pylist()[0]
print(len(r['robot_root_position']), r['fps'], round(float(np.median(r['ik_error_per_frame'])),3))")
ST=OK; [ "$OK" = 1 ] || ST=ALIGN; { [ "${CL:-0}" = 0 ] && [ "${CR:-0}" = 0 ]; } && ST="$ST+NOCONTACT"
printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$TIER" "$EP" "$SEQ" "$NF" "$FPS" "${CL:-?}" "${CR:-?}" "$IK" "${GAP:-?}" "${SURF:-?}" "$ST" > "$RES"
[ "$ST" = OK ]
