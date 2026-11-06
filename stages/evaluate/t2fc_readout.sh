#!/usr/bin/env bash
# t2fc 的读数（ml40、仓库根执行；计划 agent-doc/plan-t2fc-2026-10-20.md 第 1 节第 4、6 步）：给一组 run 补 4096 世界评测 dump（fc_eval_dump.sh，8c778f9 副本），
# 再用 ep_drop_facts.py 报每份 dump 的“物体位置误差第一次 > 0.1 m 的帧”分布。用 GPU 前等卡上没有进程（每分钟看一次，最多 2 小时）；不停任何进程。
# 用法：stages/evaluate/t2fc_readout.sh <run>:<motion parquet（仓库内相对路径）>...   每条的记录在 results/t2fc/；最后一行 T2FC_READOUT_DONE（成功 a、失败 b）
set -uo pipefail
cd "$(dirname "$0")/../.."
SRC=/lab_data/rxzhou/v2d/fc_src/fc_8c778f9; STEP=349999104
ok=0; bad=0
wait_gpu() {
  local t0=$SECONDS
  while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')" ]; do
    [ $((SECONDS - t0)) -lt 7200 ] || return 1
    sleep 60
  done
}
for pair in "$@"; do
  run=${pair%%:*}; mp=${pair#*:}
  [[ $run =~ ^t2-ep[0-9]+-[a-z0-9-]+$ && -f $mp ]] || { echo "=== ${run}：run 名或参考 ${mp} 不对，跳过"; bad=$((bad + 1)); continue; }
  if ! wait_gpu; then echo "=== ${run}：GPU 等了 2 小时还被占，跳过"; bad=$((bad + 1)); continue; fi
  echo "=== $(date '+%F %T %Z') ${run}：评测 dump"
  if ! V2D_FC_SRC=$SRC scripts/dev/record "t2fc/eval_dump_$run" stages/evaluate/fc_eval_dump.sh "$run" "$STEP" > /dev/null 2>&1; then
    echo "=== ${run}：评测 dump 失败，见 results/t2fc/eval_dump_${run}.txt.failed"; bad=$((bad + 1)); continue
  fi
  D="runs/flash_chord/$run/eval_dump/eval_${STEP}_w4096.npz"
  if scripts/dev/record "t2fc/drop_facts_$run" scripts/dev/in-flash-chord "python /repo/stages/evaluate/ep_drop_facts.py --motion /repo/$mp --dump /repo/$D --every 20" > /dev/null 2>&1; then
    ok=$((ok + 1)); echo "=== ${run}：$(grep -h '第一次 > 0.1 m' "results/t2fc/drop_facts_$run.txt" | sed 's/^ *//' | head -1)"
  else
    bad=$((bad + 1)); echo "=== ${run}：ep_drop_facts 失败"
  fi
done
echo "T2FC_READOUT_DONE（成功 ${ok}、失败 ${bad}）"
