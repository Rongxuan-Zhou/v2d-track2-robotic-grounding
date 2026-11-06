#!/usr/bin/env bash
# 按队列行单独跑一条 run，拆法与 day2_train_chain.sh 的 run_q 相同：行首的 NAME=值 作为环境变量，余下的原样交给 fc_run.sh。训练机上、仓库根执行，放 tmux。
# 用途：链开跑之前先单独跑它的冒烟（ep7 防护重训的 T5，计划 agent-doc/plan-ep7-eval-and-guards-2026-10-02.md 第 5 版 B.4），链见 DONE 就跳过。
# 环境里的 V2D_FC_SRC 等照样传下去（T5 要带 V2D_FC_SRC=<v040_p185_nfg 副本>，与链的启动命令相同）。
# 用法：stages/policy/run_queue_line.sh <队列文件> <run>；退出码就是 fc_run.sh 的退出码。
set -uo pipefail
cd "$(dirname "$0")/../.."
Q=${1:?用法：run_queue_line.sh <队列文件> <run>}; R=${2:?run}
line="$(grep -v '^[[:space:]]*#' "$Q" | awk -v r="$R" '{for (i = 1; i <= NF; i++) if ($i == r) {print; exit}}')"
[ -n "$line" ] || { echo "队列 $Q 里没有 $R" >&2; exit 2; }
read -r -a tok <<< "$line"; envs=(); i=0
while (( i < ${#tok[@]} )) && [[ ${tok[$i]} =~ ^[A-Z_][A-Z0-9_]*= ]]; do envs+=("${tok[$i]}"); i=$((i + 1)); done
echo "run_queue_line: $R ← ${Q}（行首环境 ${envs[*]}；V2D_FC_SRC=${V2D_FC_SRC:-未设}）"
env "${envs[@]}" bash stages/policy/fc_run.sh "${tok[@]:$i}" < /dev/null
