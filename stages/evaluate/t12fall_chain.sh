#!/usr/bin/env bash
# t12fall 诊断链（ml40 上、仓库根执行，放 tmux；计划 agent-doc/plan-t12fall-2026-10-20.md 第 2 版第 1 节）。
# 1. 等 V2D_WAIT_FOR_LOG（默认 runs/logs/cand_t1_t1ho_ml40.log，t12lift 的 t1ho 导出链）最后一行是 EXPORT_CHAIN_DONE/FAILED（最多 8 小时）；
# 2. 每次用 GPU 前等卡上没有进程（每分钟看一次，最多 2 小时；不停任何进程）；
# 3. 试跑：ep4 的 16 个世界（--palms --bodies），diag2_sync.py 看手掌与物体的同步闸，fall_mechanism.py 看 G0；任一不过就停；
# 4. 五条 run 各 1024 个世界（--palms --bodies），各跑一次 fall_mechanism.py。诊断挂 fc_8c778f9 副本（同那几份评测）。
#    rollout 给 V2D_GPU=1（in-flash-chord 只在这时挂 GPU，同 diag2_chain.sh 的 fc），同步与分析给 V2D_GPU=0。
# 日志 runs/logs/t12fall_ml40.log，最后一行 T12FALL_DONE（成功 a、失败 b）或 T12FALL_FAILED <原因>。记录在 results/t12fall/。
set -uo pipefail
cd "$(dirname "$0")/../.."
LOG=runs/logs/t12fall_ml40.log
exec >> "$LOG" 2>&1
say() { echo "=== $(date '+%F %T %Z') $*"; }
fail() { say "T12FALL_FAILED $*"; exit 1; }
WAIT=${V2D_WAIT_FOR_LOG:-runs/logs/cand_t1_t1ho_ml40.log}
SRC=/lab_data/rxzhou/v2d/fc_src/fc_8c778f9
M=/repo/runs/motion/tier2/whole_body/soma
declare -A SEQ=([0]=2026-06-04_14-16-23_tall_bar_stool_tip_up_04 [1]=2026-05-18_17-06-21_vacuum_left_lean_02 [4]=2026-04-24_11-40-52_baseball_bat_rotate_05
                [16]=2026-03-12_15-06-36_spray_bottle_left_desk_ground_02 [25]=2026-03-20_11-59-57_cyan_water_bottle_rotate_01)
touch runs/.chain-active-t12fall_ml40; trap 'rm -f runs/.chain-active-t12fall_ml40' EXIT
say "开始，来历 $(cat .v2d-source-commit 2>/dev/null || echo unknown)；等 ${WAIT}"
for _ in $(seq 1 480); do grep -qE "EXPORT_CHAIN_(DONE|FAILED)" < <(tail -n 1 "$WAIT" 2>/dev/null) && break; sleep 60; done
grep -qE "EXPORT_CHAIN_(DONE|FAILED)" < <(tail -n 1 "$WAIT" 2>/dev/null) || fail "等的导出链 8 小时内没结束"
say "上一段结束：$(tail -n 1 "$WAIT" | sed -E 's/EXPORT_CHAIN_(DONE|FAILED)/导出链结束/')"
wait_gpu() { local t0=$SECONDS; while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')" ]; do
  [ $((SECONDS - t0)) -lt 7200 ] || return 1; sleep 60; done; }
roll() {  # roll <ep> <世界数> <前缀名>
  local run="t2-ep$1-w2048-cap10m-r3a-ml40" out="runs/t12fall/$3"
  mkdir -p runs/t12fall
  wait_gpu || return 1
  V2D_GPU=1 V2D_FC_SRC=$SRC scripts/dev/record "t12fall/rollout_$3" scripts/dev/in-flash-chord \
    "cd /workspace && python /repo/stages/evaluate/diag_rollout.py /repo/runs/flash_chord/$run/policy_349999104.safetensors /repo/$out --worlds $2 --palms --bodies" > /dev/null 2>&1 || return 1
  rm -rf "${out}_hydra"
  V2D_GPU=0 scripts/dev/record "t12fall/mech_$3" scripts/dev/in-flash-chord \
    "python /repo/stages/evaluate/fall_mechanism.py --npz /repo/$out.npz --motion $M/sequence_id=${SEQ[$1]}/robot_name=g1/data.parquet --label ep$1" > /dev/null 2>&1
}
roll 4 16 smoke_ep4 || fail "试跑 ep4 没成（results/t12fall/rollout_smoke_ep4.txt.failed 或 mech_smoke_ep4.txt.failed）"
V2D_GPU=0 V2D_FC_SRC=$SRC scripts/dev/record t12fall/sync_smoke_ep4 scripts/dev/in-flash-chord "python /repo/stages/evaluate/diag2_sync.py /repo/runs/t12fall/smoke_ep4.npz" > /dev/null 2>&1 \
  || fail "试跑的同步闸不过（results/t12fall/sync_smoke_ep4.txt.failed）"
grep -q "G0 第 0 步.*PASS" results/t12fall/mech_smoke_ep4.txt || fail "试跑的 G0 不过"
say "试跑通过：$(grep -h 'G0 第 0 步' results/t12fall/mech_smoke_ep4.txt | cut -c1-160)"
ok=0; bad=0
for ep in 0 16 4 1 25; do
  if roll "$ep" 1024 "ep${ep}_w1024"; then ok=$((ok + 1)); say "ep${ep}：$(grep -h '摔倒（' results/t12fall/mech_ep${ep}_w1024.txt | cut -c1-150)"
  else bad=$((bad + 1)); say "ep${ep}：没成"; fi
done
say "T12FALL_DONE（成功 ${ok}、失败 ${bad}）"
