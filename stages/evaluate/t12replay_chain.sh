#!/usr/bin/env bash
# t12replay 诊断链（ml40 上、仓库根执行，放 tmux；计划 agent-doc/plan-t12replay-2026-10-20.md 第 2 版第 1 节）。
# 用法：stages/evaluate/t12replay_chain.sh <run>:<步数>:<序列名> ...
# 每条：用 GPU 前等卡上没有进程（每分钟看一次，最多 2 小时；不停任何进程）→ diag_rollout.py --worlds 1024 --palms --bodies（V2D_GPU=1，挂 fc_8c778f9 副本）
#        → diag2_sync.py 看手掌与物体的同步闸（V2D_GPU=0）→ fall_mechanism.py（V2D_GPU=0）。记录在 results/t12replay/。
# 日志 runs/logs/t12replay_ml40.log，最后一行 T12REPLAY_DONE（成功 a、失败 b）。
set -uo pipefail
cd "$(dirname "$0")/../.."
LOG=runs/logs/t12replay_ml40.log
exec >> "$LOG" 2>&1
say() { echo "=== $(date '+%F %T %Z') $*"; }
SRC=/lab_data/rxzhou/v2d/fc_src/fc_8c778f9
M=/repo/runs/motion/tier2/whole_body/soma
touch runs/.chain-active-t12replay_ml40; trap 'rm -f runs/.chain-active-t12replay_ml40' EXIT
wait_gpu() { local t0=$SECONDS; while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')" ]; do
  [ $((SECONDS - t0)) -lt 7200 ] || return 1; sleep 60; done; }
say "开始，来历 $(cat .v2d-source-commit 2>/dev/null || echo unknown)；$# 条"
ok=0; bad=0
for spec in "$@"; do
  IFS=: read -r run step seq <<< "$spec"
  name="${run}_s${step}"; out="runs/t12replay/$name"
  mkdir -p runs/t12replay
  P="runs/flash_chord/$run/policy_${step}.safetensors"
  if [ ! -f "$P" ]; then say "${run}：没有 $P"; bad=$((bad + 1)); continue; fi
  if ! wait_gpu; then say "${run}：GPU 等了 2 小时还被占"; bad=$((bad + 1)); continue; fi
  if ! V2D_GPU=1 V2D_FC_SRC=$SRC scripts/dev/record "t12replay/rollout_$name" scripts/dev/in-flash-chord \
      "cd /workspace && python /repo/stages/evaluate/diag_rollout.py /repo/$P /repo/$out --worlds 1024 --palms --bodies" > /dev/null 2>&1; then
    say "${run}：rollout 没成"; bad=$((bad + 1)); continue
  fi
  rm -rf "${out}_hydra"
  V2D_GPU=0 V2D_FC_SRC=$SRC scripts/dev/record "t12replay/sync_$name" scripts/dev/in-flash-chord "python /repo/stages/evaluate/diag2_sync.py /repo/$out.npz" > /dev/null 2>&1 \
    || say "${run}：同步闸不过（只报）"
  V2D_GPU=0 scripts/dev/record "t12replay/mech_$name" scripts/dev/in-flash-chord \
    "python /repo/stages/evaluate/fall_mechanism.py --npz /repo/$out.npz --motion $M/sequence_id=$seq/robot_name=g1/data.parquet --label $run" > /dev/null 2>&1
  ok=$((ok + 1)); say "${run}：$(grep -h '摔倒（' "results/t12replay/mech_$name.txt"* 2>/dev/null | head -1 | cut -c1-150)"
done
say "T12REPLAY_DONE（成功 ${ok}、失败 ${bad}）"
