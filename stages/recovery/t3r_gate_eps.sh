#!/usr/bin/env bash
# 在只读命名空间里跑我们 gate 仓库的 eps.sh（只经 stages/recovery/t3r_run.sh 调用）。计划 agent-doc/plan-t3takeover-v3-2026-10-21.md 复核第 1 条。
# 她的 run_parallel.sh 23 行给 eps.sh 的环境是 OMP/MKL 3 线程、PATH 先放 gate venv（one.sh 里的 python3 要用它）、V2D_T3_SET=<集>。
# PATH 只在这里（命名空间里、能力已丢）改：gate venv 的 bin 目录经环境变量 T3R_GATE_VENV_BIN 传进来（第 1 级是她的，1b 是 t3recon/env/gate/bin），
# 不出现在 t3r_run.sh 的参数里，也不进外层的 PATH。
# 用法：T3R_GATE_VENV_BIN=<她的 gate venv/bin> stages/recovery/t3r_run.sh /bin/bash <本脚本> <gate 仓库> <集> <ep>...
set -uo pipefail
GR=${1:?gate 仓库}; SET=${2:?集}; shift 2
[ -n "${T3R_GATE_VENV_BIN:-}" ] || { echo "T3R_GATE_EPS_FAILED 没给 T3R_GATE_VENV_BIN"; exit 2; }
[ ! -w "$T3R_HER" ] || { echo "T3R_GATE_EPS_FAILED 她的目录可写（不在只读命名空间里？）"; exit 2; }   # 1b 复核改动第 3 条
vb="$(realpath -e "$T3R_GATE_VENV_BIN")" || { echo "T3R_GATE_EPS_FAILED T3R_GATE_VENV_BIN 不存在"; exit 2; }
case "$vb/" in "$T3R_HER"/?*|/lab_data/rxzhou/v2d/t3recon/env/?*) ;; *) echo "T3R_GATE_EPS_FAILED T3R_GATE_VENV_BIN 不在她的目录（第 1 级）或 t3recon/env（1b）下"; exit 2 ;; esac
cd "$GR" || exit 2
export OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 PATH="$T3R_GATE_VENV_BIN:$PATH" V2D_T3_SET="$SET"
exec nice -n 10 bash stages/retarget/eps.sh 3 "$@"
