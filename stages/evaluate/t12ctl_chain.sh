#!/usr/bin/env bash
# t12ctl 链（ml40 上、仓库根执行，放 tmux；计划 agent-doc/plan-t12ctl-2026-10-20.md 第 1 版第 1 节）。
# 用法：stages/evaluate/t12ctl_chain.sh <run>:<步数> <tag>:<序列名>:<第一次接触帧> ...
# V2D_CTL_MOTION=<参考根>（默认 runs/motion/tier2）换参考根。
# 每条：用 GPU 前等卡上没有进程（每分钟看一次，最多 2 小时；不停任何进程）→ 用这个零残差 checkpoint 换上 runs/motion/tier2 里这一集的原参考
# diag_rollout.py --worlds 1024 --palms --bodies --parquet（V2D_GPU=1，挂 fc_8c778f9）→ diag2_sync.py → fall_mechanism.py --before <接触帧>。
# 记录在 results/t12ctl/，产物在 runs/t12ctl/<run>_s<步数>_<tag>。日志 runs/logs/t12ctl_ml40.log，最后一行 T12CTL_DONE（成功 a、失败 b）。
set -uo pipefail
cd "$(dirname "$0")/../.."
LOG=runs/logs/t12ctl_ml40.log
exec >> "$LOG" 2>&1
say() { echo "=== $(date '+%F %T %Z') $*"; }
SRC=/lab_data/rxzhou/v2d/fc_src/fc_8c778f9
M=${V2D_CTL_MOTION:-runs/motion/tier2}/whole_body/soma   # 参考根，默认原 Tier 2（计划 agent-doc/plan-t12arm-2026-10-20.md 用改过的根）
touch runs/.chain-active-t12ctl_ml40; trap 'rm -f runs/.chain-active-t12ctl_ml40' EXIT
wait_gpu() { local t0=$SECONDS; while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')" ]; do
  [ $((SECONDS - t0)) -lt 7200 ] || return 1; sleep 60; done; }
IFS=: read -r run step <<< "${1:?用法：t12ctl_chain.sh <run>:<步数> <tag>:<序列名>:<接触帧> ...}"; shift
P="runs/flash_chord/$run/policy_${step}.safetensors"
say "开始，来历 $(cat .v2d-source-commit 2>/dev/null || echo unknown)；${run} 第 ${step} 步；参考根 ${M%/whole_body/soma}；$# 集"
[ -f "$P" ] || { say "没有 $P"; say "T12CTL_DONE（成功 0、失败 $#）"; exit 1; }
mkdir -p runs/t12ctl
ok=0; bad=0
for spec in "$@"; do
  IFS=: read -r tag seq tc <<< "$spec"
  Q="$M/sequence_id=$seq/robot_name=g1/data.parquet"
  name="${run}_s${step}_${tag}"; out="runs/t12ctl/$name"
  [ -f "$Q" ] || { say "${tag}：没有 $Q"; bad=$((bad + 1)); continue; }
  if ! wait_gpu; then say "${tag}：GPU 等了 2 小时还被占"; bad=$((bad + 1)); continue; fi
  if ! V2D_GPU=1 V2D_FC_SRC=$SRC scripts/dev/record "t12ctl/rollout_$name" scripts/dev/in-flash-chord \
      "cd /workspace && python /repo/stages/evaluate/diag_rollout.py /repo/$P /repo/$out --worlds 1024 --palms --bodies --parquet /repo/$Q" > /dev/null 2>&1; then
    say "${tag}：rollout 没成"; bad=$((bad + 1)); continue
  fi
  rm -rf "${out}_hydra"
  V2D_GPU=0 V2D_FC_SRC=$SRC scripts/dev/record "t12ctl/sync_$name" scripts/dev/in-flash-chord "python /repo/stages/evaluate/diag2_sync.py /repo/$out.npz" > /dev/null 2>&1 \
    || say "${tag}：同步闸不过"
  if ! V2D_GPU=0 scripts/dev/record "t12ctl/mech_$name" scripts/dev/in-flash-chord \
      "python /repo/stages/evaluate/fall_mechanism.py --npz /repo/$out.npz --motion /repo/$Q --label $tag --before $tc" > /dev/null 2>&1; then
    say "${tag}：fall_mechanism 没成"; bad=$((bad + 1)); continue
  fi
  ok=$((ok + 1))
  say "${tag}：$(grep -h '^FALL_BEFORE' "results/t12ctl/mech_$name.txt" | tr '\n' ' ')；$(grep -h '摔倒帧分位' "results/t12ctl/mech_$name.txt" | head -1 | cut -c1-90)"
done
say "T12CTL_DONE（成功 ${ok}、失败 ${bad}）"
