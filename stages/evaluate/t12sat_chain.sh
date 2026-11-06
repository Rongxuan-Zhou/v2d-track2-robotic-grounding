#!/usr/bin/env bash
# t12sat 链（ml40 上、仓库根执行，放 tmux；计划 agent-doc/plan-t12sat-2026-10-20.md 第 1 节）。
# 用法：stages/evaluate/t12sat_chain.sh <tag>:<run>:<步数>:<序列名>:<接触帧>:<窗口,窗口,...> ...   （窗口写 a-b，参考帧）
# 每条：用 GPU 前等卡上没有进程（每分钟看一次，最多 2 小时；不停任何进程）→ diag_rollout.py --worlds 1024 --palms --bodies --actions 300（V2D_GPU=1，挂 fc_8c778f9；不换参考）
#        → diag2_sync.py → fall_mechanism.py --before <接触帧> → residual_sat.py。记录在 results/t12sat/，产物在 runs/t12sat/<run>_s<步数>。
# 日志 runs/logs/t12sat_ml40.log，最后一行 T12SAT_DONE（成功 a、失败 b）。
set -uo pipefail
cd "$(dirname "$0")/../.."
LOG=runs/logs/t12sat_ml40.log
exec >> "$LOG" 2>&1
say() { echo "=== $(date '+%F %T %Z') $*"; }
SRC=/lab_data/rxzhou/v2d/fc_src/fc_8c778f9
M=runs/motion/tier2/whole_body/soma
SRCC="$(cat .v2d-source-commit 2>/dev/null || echo unknown)"
case "$SRCC" in *-syncing) say "来历标记是 ${SRCC}，同步还没完"; say "T12SAT_DONE（成功 0、失败 $#）"; exit 1 ;; esac
touch runs/.chain-active-t12sat_ml40; trap 'rm -f runs/.chain-active-t12sat_ml40' EXIT
wait_gpu() { local t0=$SECONDS; while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')" ]; do
  [ $((SECONDS - t0)) -lt 7200 ] || return 1; sleep 60; done; }
say "开始，来历 ${SRCC}；$# 条"
mkdir -p runs/t12sat
ok=0; bad=0
for spec in "$@"; do
  IFS=: read -r tag run step seq tc wins <<< "$spec"
  P="runs/flash_chord/$run/policy_${step}.safetensors"
  Q="$M/sequence_id=$seq/robot_name=g1/data.parquet"
  out="runs/t12sat/${run}_s${step}"
  [ -f "$P" ] && [ -f "$Q" ] || { say "${tag}：没有 $P 或 $Q"; bad=$((bad + 1)); continue; }
  if ! wait_gpu; then say "${tag}：GPU 等了 2 小时还被占"; bad=$((bad + 1)); continue; fi
  if ! V2D_GPU=1 V2D_FC_SRC=$SRC scripts/dev/record "t12sat/rollout_$tag" scripts/dev/in-flash-chord \
      "cd /workspace && python /repo/stages/evaluate/diag_rollout.py /repo/$P /repo/$out --worlds 1024 --palms --bodies --actions 300" > /dev/null 2>&1; then
    say "${tag}：rollout 没成"; bad=$((bad + 1)); continue
  fi
  rm -rf "${out}_hydra"
  V2D_GPU=0 V2D_FC_SRC=$SRC scripts/dev/record "t12sat/sync_$tag" scripts/dev/in-flash-chord "python /repo/stages/evaluate/diag2_sync.py /repo/$out.npz" > /dev/null 2>&1 || say "${tag}：同步闸不过"
  V2D_GPU=0 scripts/dev/record "t12sat/mech_$tag" scripts/dev/in-flash-chord \
    "python /repo/stages/evaluate/fall_mechanism.py --npz /repo/$out.npz --motion /repo/$Q --label $tag --before $tc" > /dev/null 2>&1 || say "${tag}：fall_mechanism 没成"
  wargs=""; for w in ${wins//,/ }; do wargs="$wargs --window ${w/-/:}"; done
  if ! V2D_GPU=0 scripts/dev/record "t12sat/sat_$tag" scripts/dev/in-flash-chord \
      "python /repo/stages/evaluate/residual_sat.py --npz /repo/$out.npz --policy /repo/$P --label $tag$wargs" > /dev/null 2>&1; then
    say "${tag}：residual_sat 没成"; bad=$((bad + 1)); continue
  fi
  ok=$((ok + 1))
  say "${tag}：$(grep -h '^FALL_BEFORE' "results/t12sat/mech_$tag.txt" 2>/dev/null | tr '\n' ' ')"
done
say "T12SAT_DONE（成功 ${ok}、失败 ${bad}）"
