#!/usr/bin/env bash
# t3g5 一段的读数（训练机上、仓库根执行；由 arm_t3g5.sh 经 run_after_log.sh 在评测链结束后调用）。计划 agent-doc/plan-t3g5-2026-10-21.md 第 2 版第 1 节第 3–4 步、判据 R1/R2。
# 1. eval_table.py：这一段各 run 第 349999104 步的 4096 世界评测（对自己参考），记 results/t3g5/eval_<段>_<机器>.txt；
# 2. 判据 R1 逐集列出（SP-SR ≥ 0.5 且 MPPE ≤ 10 cm 记“跟上”，其余记“仍跟丢”），记在同一份记录的末尾；
# 3. 开局诊断（10/15 的工具，只报）：ref_early_motion.py、ref_start_height.py（容器、CPU），记 results/t3g5/early_<段>_<机器>.txt、start_<段>_<机器>.txt；
# 4. 训练回合怎么结束：train_term_table.py（宿主机 python3），记 results/t3g5/term_<段>_<机器>.txt。
# 用法：stages/evaluate/t3g5_readout.sh <段名> <机器>   最后一行 T3G5_READOUT_DONE（各步的退出码）
set -uo pipefail
cd "$(dirname "$0")/../.."
SEG=${1:?段名}; MACH=${2:?机器}
STEP=349999104
mapfile -t RUNS < <(grep -v '^#' "stages/evaluate/exports/cand_t3_${SEG}_${MACH}.tsv" | cut -f3)
[ "${#RUNS[@]}" -ge 1 ] || { echo "t3g5_readout: 清单里没有 run"; exit 2; }
files=()
for r in "${RUNS[@]}"; do files+=("results/day2/eval_${r}_s${STEP}.txt"); done
scripts/dev/record "t3g5/eval_${SEG}_${MACH}" bash -c 'python3 stages/evaluate/eval_table.py "$@" && python3 stages/evaluate/t3g5_r1.py "$@"' _ "${files[@]}" > /dev/null 2>&1; e1=$?
V2D_GPU=0 scripts/dev/record "t3g5/early_${SEG}_${MACH}" scripts/dev/in-flash-chord "python /repo/stages/evaluate/ref_early_motion.py ${RUNS[*]} --root /repo" > /dev/null 2>&1; e2=$?
V2D_GPU=0 scripts/dev/record "t3g5/start_${SEG}_${MACH}" scripts/dev/in-flash-chord "python /repo/stages/evaluate/ref_start_height.py ${RUNS[*]} --root /repo" > /dev/null 2>&1; e3=$?
specs=(); for r in "${RUNS[@]}"; do specs+=("$(sed -E 's/^t3-ep([0-9]+)-.*/\1/' <<< "$r"):$r"); done
scripts/dev/record "t3g5/term_${SEG}_${MACH}" python3 stages/evaluate/train_term_table.py "${specs[@]}" > /dev/null 2>&1; e4=$?
echo "T3G5_READOUT_DONE（eval ${e1}、early ${e2}、start ${e3}、term ${e4}）"
