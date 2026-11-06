#!/usr/bin/env bash
# trajopt C 段链（mf4090 上、仓库根执行，放 tmux；计划 agent-doc/plan-trajopt-2026-10-26.md 第 5 节第 3 版）。照 stages/evaluate/t12sat_chain.sh 的做法，换成 mf4090。
# 用法：stages/evaluate/trajopt_c_chain.sh <世界数> <tag>:<run>:<步数>:<接触帧>:<训练回合长度>:<窗口,窗口,...> ...   （窗口写 a-b，参考帧；世界数 16 = 冒烟）
# 每条：GPU 上有别的进程就等（每分钟看一次，最多 2 小时；不停任何进程）→ diag_rollout.py --worlds N --bodies --actions <参考帧数> --action-worlds N（V2D_GPU=1，挂 fc_8c778f9；不换参考）
#        → fall_mechanism.py --before <接触帧> --save → residual_sat.py（--window 与 --fall-npz、--horizon）。记录在 results/trajopt/，产物在 runs/trajopt/c_<run>_w<N>。
# 日志 runs/logs/trajopt_c_mf4090.log，最后一行 TRAJOPT_C_DONE（成功 a、失败 b）。
set -uo pipefail
cd "$(dirname "$0")/../.."
LOG=runs/logs/trajopt_c_mf4090.log; mkdir -p runs/logs runs/trajopt results/trajopt
exec >> "$LOG" 2>&1
say() { echo "=== $(date '+%F %T %Z') $*"; }
SRC="$HOME/v2d/fc_src/fc_8c778f9"
WN=${1:?世界数}; shift
SRCC="$(cat .v2d-source-commit 2>/dev/null || echo unknown)"
case "$SRCC" in *-syncing|*-dirty|unknown) say "来历标记是 ${SRCC}"; say "TRAJOPT_C_DONE（成功 0、失败 $#）"; exit 1 ;; esac
touch runs/.chain-active-trajopt_c_mf4090; trap 'rm -f runs/.chain-active-trajopt_c_mf4090' EXIT
wait_gpu() { local t0=$SECONDS; while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')" ]; do
  [ $((SECONDS - t0)) -lt 7200 ] || return 1; sleep 60; done; }
say "开始，来历 ${SRCC}；$# 条，${WN} 个世界"
ok=0; bad=0
for spec in "$@"; do
  IFS=: read -r tag run step tc hz wins <<< "$spec"
  P="runs/flash_chord/$run/policy_${step}.safetensors"
  Q="$(grep -o "task.parquet=[^ ]*" runs/flash_chord/$run/hydra/s1/.hydra/overrides.yaml | head -1 | cut -d= -f2- | sed "s/['\"]//g; s#^/repo/##")"
  out="runs/trajopt/c_${run}_w${WN}"
  [ -f "$P" ] && [ -f "$Q" ] || { say "${tag}：没有 $P 或 $Q"; bad=$((bad + 1)); continue; }
  nf="$(PYTHONDONTWRITEBYTECODE=1 scripts/dev/in-flash-chord "python -c \"import pyarrow.parquet as pq; print(len(pq.read_table('/repo/$Q', columns=['robot_joint_positions']).to_pylist()[0]['robot_joint_positions']))\"" 2>/dev/null | tail -n 1)"
  [[ $nf =~ ^[0-9]+$ ]] || { say "${tag}：读不出参考帧数"; bad=$((bad + 1)); continue; }
  if ! wait_gpu; then say "${tag}：GPU 等了 2 小时还被占"; bad=$((bad + 1)); continue; fi
  if ! V2D_GPU=1 V2D_FC_SRC=$SRC scripts/dev/record "trajopt/c_rollout_${tag}_w${WN}" scripts/dev/in-flash-chord \
      "cd /workspace && python /repo/stages/evaluate/diag_rollout.py /repo/$P /repo/$out --worlds $WN --bodies --actions $nf --action-worlds $WN" > /dev/null 2>&1; then
    say "${tag}：rollout 失败"; bad=$((bad + 1)); continue; fi
  if ! V2D_GPU=0 scripts/dev/record "trajopt/c_fall_${tag}_w${WN}" scripts/dev/in-flash-chord \
      "python /repo/stages/evaluate/fall_mechanism.py --npz /repo/$out.npz --motion /repo/$Q --label $tag --before $tc --save /repo/${out}_fall.npz" > /dev/null 2>&1; then
    say "${tag}：fall_mechanism 失败"; bad=$((bad + 1)); continue; fi
  wargs=""; for w in ${wins//,/ }; do wargs="$wargs --window ${w/-/:}"; done
  if ! V2D_GPU=0 scripts/dev/record "trajopt/c_sat_${tag}_w${WN}" scripts/dev/in-flash-chord \
      "python /repo/stages/evaluate/residual_sat.py --npz /repo/$out.npz --policy /repo/$P $wargs --fall-npz /repo/${out}_fall.npz --horizon $hz --label $tag" > /dev/null 2>&1; then
    say "${tag}：residual_sat 失败"; bad=$((bad + 1)); continue; fi
  say "${tag}：完成（参考 ${nf} 帧）"; ok=$((ok + 1))
done
say "TRAJOPT_C_DONE（成功 ${ok}、失败 ${bad}）"
