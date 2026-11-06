#!/usr/bin/env bash
# t12ref 链（ml40 上、仓库根执行，放 tmux；计划 agent-doc/plan-t12ref-2026-10-20.md 第 2 版第 1 节）。
# 用法：stages/evaluate/t12ref_chain.sh <run>:<步数>:<序列名> <tag>:<上限 rad>:<过渡帧数> ...
# 1. CPU：每个 tag 用 waist_cap.py 写 runs/motion/tier2_<tag>（元数据 runs/motion_src/tier2_<tag>；已在就不重写）→ check_inputs → 支撑面与 tier2 逐字节比。
# 2. GPU（用前等卡上没有进程，每分钟看一次，最多 2 小时；不停任何进程；V2D_GPU=1，挂 fc_8c778f9）：零残差回放换成这份参考
#    diag_rollout.py --worlds 1024 --palms --bodies --parquet → diag2_sync.py（同步闸）→ fall_mechanism.py --before 133（--motion 给同一份参考）
#    → replay_diff.py（与原参考的回放 runs/t12replay/<run>_s<步数>.npz 比，Q0b：改动有没有进到仿真）。
# 记录在 results/t12ref/，产物在 runs/t12ref/<run>_s<步数>_<tag>。日志 runs/logs/t12ref_ml40.log，最后一行 T12REF_DONE（成功 a、失败 b）。
set -uo pipefail
cd "$(dirname "$0")/../.."
LOG=runs/logs/t12ref_ml40.log
exec >> "$LOG" 2>&1
say() { echo "=== $(date '+%F %T %Z') $*"; }
SRC=/lab_data/rxzhou/v2d/fc_src/fc_8c778f9
touch runs/.chain-active-t12ref_ml40; trap 'rm -f runs/.chain-active-t12ref_ml40' EXIT
wait_gpu() { local t0=$SECONDS; while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')" ]; do
  [ $((SECONDS - t0)) -lt 7200 ] || return 1; sleep 60; done; }
IFS=: read -r run step seq <<< "${1:?用法：t12ref_chain.sh <run>:<步数>:<序列名> <tag>:<上限>:<过渡> ...}"; shift
say "开始，来历 $(cat .v2d-source-commit 2>/dev/null || echo unknown)；${run} 第 ${step} 步，${seq}；$# 份参考"
P="runs/flash_chord/$run/policy_${step}.safetensors"
A="runs/t12replay/${run}_s${step}.npz"
ORIG="runs/motion/tier2/whole_body/soma/sequence_id=$seq/robot_name=g1/data.parquet"
for f in "$P" "$A" "$ORIG"; do [ -f "$f" ] || { say "没有 $f"; say "T12REF_DONE（成功 0、失败 $#）"; exit 1; }; done
mkdir -p runs/t12ref
ok=0; bad=0; pqs=""
for spec in "$@"; do   # 1. 写参考
  IFS=: read -r tag cap blend <<< "$spec"
  D="runs/motion/tier2_$tag"; Q="$D/whole_body/soma/sequence_id=$seq/robot_name=g1/data.parquet"
  if [ ! -f "$Q" ]; then
    scripts/dev/record "t12ref/write_$tag" scripts/dev/in-flash-chord \
      "python /repo/stages/retarget/waist_cap.py /repo/$ORIG /repo/$D --meta-src /repo/runs/motion_src/tier2 --meta-dst /repo/runs/motion_src/tier2_$tag --cap $cap --blend $blend" \
      > /dev/null 2>&1 || { say "${tag}：写参考没成"; continue; }
  fi
  same=1; for f in "$D"/whole_body/reconstructed_stage/*; do cmp -s "$f" "runs/motion/tier2/whole_body/reconstructed_stage/$(basename "$f")" || same=0; done
  [ "$same" = 1 ] || { say "${tag}：支撑面与 tier2 不同"; continue; }
  pqs="$pqs /repo/$Q"
  say "${tag}：$(grep -h '^参考' "results/t12ref/write_$tag.txt" 2>/dev/null | sed 's/.*；waist_pitch/waist_pitch/' | cut -c1-150)"
done
if ! scripts/dev/record t12ref/check_inputs_ml40 scripts/dev/in-flash-chord "python /repo/stages/policy/check_inputs.py $pqs" > /dev/null 2>&1; then
  say "check_inputs 没过"; say "T12REF_DONE（成功 0、失败 $#）"; exit 1
fi
say "check_inputs 过"
for spec in "$@"; do   # 2. 回放
  IFS=: read -r tag cap blend <<< "$spec"
  Q="runs/motion/tier2_$tag/whole_body/soma/sequence_id=$seq/robot_name=g1/data.parquet"
  name="${run}_s${step}_${tag}"; out="runs/t12ref/$name"
  [ -f "$Q" ] || { bad=$((bad + 1)); continue; }
  if ! wait_gpu; then say "${tag}：GPU 等了 2 小时还被占"; bad=$((bad + 1)); continue; fi
  if ! V2D_GPU=1 V2D_FC_SRC=$SRC scripts/dev/record "t12ref/rollout_$name" scripts/dev/in-flash-chord \
      "cd /workspace && python /repo/stages/evaluate/diag_rollout.py /repo/$P /repo/$out --worlds 1024 --palms --bodies --parquet /repo/$Q" > /dev/null 2>&1; then
    say "${tag}：rollout 没成"; bad=$((bad + 1)); continue
  fi
  rm -rf "${out}_hydra"
  V2D_GPU=0 V2D_FC_SRC=$SRC scripts/dev/record "t12ref/sync_$name" scripts/dev/in-flash-chord "python /repo/stages/evaluate/diag2_sync.py /repo/$out.npz" > /dev/null 2>&1 \
    || say "${tag}：同步闸不过"
  V2D_GPU=0 scripts/dev/record "t12ref/mech_$name" scripts/dev/in-flash-chord \
    "python /repo/stages/evaluate/fall_mechanism.py --npz /repo/$out.npz --motion /repo/$Q --label $tag --before 133" > /dev/null 2>&1
  V2D_GPU=0 scripts/dev/record "t12ref/diff_$name" scripts/dev/in-flash-chord \
    "python /repo/stages/evaluate/replay_diff.py --npz-a /repo/$A --motion-a /repo/$ORIG --npz-b /repo/$out.npz --motion-b /repo/$Q" > /dev/null 2>&1
  ok=$((ok + 1))
  say "${tag}：$(grep -h '^FALL_BEFORE' "results/t12ref/mech_$name.txt"* 2>/dev/null | head -1)；$(grep -h '^REPLAY_DIFF' "results/t12ref/diff_$name.txt"* 2>/dev/null | head -1)"
done
say "T12REF_DONE（成功 ${ok}、失败 ${bad}）"
