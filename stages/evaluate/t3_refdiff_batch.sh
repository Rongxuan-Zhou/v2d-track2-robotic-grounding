#!/usr/bin/env bash
# 按清单逐行比两份重定向参考（训练机上、仓库根执行；每行在一次性 FlashCHORD 容器里跑 stages/evaluate/ref_pair_diff.py，不挂 GPU，只读两份参考）。
# 计划 agent-doc/plan-t3-gb3-switch-2026-10-17.md 做法 2.6（只报，不判）。
# 清单每行 tab 分隔：集、标签、旧参考 parquet、新参考 parquet（容器内路径 /repo/runs/motion/...；# 开头为注释）。
# 每行先照抄 ref_pair_diff.py 的输出，再打一行汇总：REFDIFF ep<集> <标签> 物体相对机器人根 <中位 m> 机器人根 <中位 m>。
# 某一行失败（退出码非 0 或没有 REF_PAIR_DIFF_DONE）记 REFDIFF_FAIL 并接着做下一行。最后一行 T3_REFDIFF_DONE <成功> <失败>。
# 用法：scripts/dev/record t3gb3/refdiff_mf4090 stages/evaluate/t3_refdiff_batch.sh stages/evaluate/exports/t3_gb3_refdiff_pairs.tsv
set -uo pipefail
cd "$(dirname "$0")/../.."
L=${1:?用法：t3_refdiff_batch.sh <清单>}
[ -f "$L" ] || { echo "t3_refdiff_batch: 没有清单 $L" >&2; exit 2; }
ok=0; bad=0
while IFS=$'\t' read -r ep label a b; do
  [[ -z $ep || $ep == \#* ]] && continue
  [[ $ep =~ ^[0-9]+$ && $a == /repo/runs/motion/* && $b == /repo/runs/motion/* && $a == *.parquet && $b == *.parquet ]] \
    || { echo "REFDIFF_FAIL ep${ep} ${label}：清单行不合要求"; bad=$((bad + 1)); continue; }
  echo "== ep${ep} ${label}：${a#/repo/} → ${b#/repo/}"
  out="$(V2D_GPU=0 scripts/dev/in-flash-chord "python /repo/stages/evaluate/ref_pair_diff.py $a $b" 2>&1 < /dev/null)"; rc=$?   # 容器不吃清单的 stdin
  printf '%s\n' "$out" | grep -v "接触帧（各手）"
  if [ "$rc" != 0 ] || ! grep -q '^REF_PAIR_DIFF_DONE$' <<< "$out"; then
    echo "REFDIFF_FAIL ep${ep} ${label}：退出码 ${rc}"; bad=$((bad + 1)); continue
  fi
  rel=$(grep -oE '物体相对机器人根的位置差：中位 [0-9.]+' <<< "$out" | grep -oE '[0-9.]+$')
  root=$(grep -oE '机器人根位置差：中位 [0-9.]+' <<< "$out" | grep -oE '[0-9.]+$')
  echo "REFDIFF ep${ep} ${label} 物体相对机器人根 ${rel} 机器人根 ${root}"
  ok=$((ok + 1))
done < "$L"
echo "T3_REFDIFF_DONE ${ok} ${bad}"
