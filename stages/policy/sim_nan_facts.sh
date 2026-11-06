#!/usr/bin/env bash
# 仿真器对非有限状态的行为：从训练镜像（flash_chord:latest）里摘 Newton、mujoco_warp、Warp 的源码原文，每行带文件与行号，供 jev 核对。
# Mac 上执行，只读；在训练机上起一个不带 GPU 的一次性容器（docker run --rm，不加 --gpus），不碰在跑的链。主机按 V2D_ENV 选（同 scripts/dev/remote）。
# 计划 agent-doc/plan-ep7-eval-and-guards-2026-10-02.md 的 B 部分。查：
#   1. 三个包的版本
#   2. mujoco_warp 的约束求解：起点直接取上一步的 qacc（warmstart），每步结束把 qacc 存回 qacc_warmstart，有没有有限性检查
#   3. Newton 的 SolverMuJoCo.step：每步把 Newton 状态拷进 mjw_data、跑 mujoco_warp、再拷回；有没有清 warmstart
#   4. Warp 对 NaN 的行为：浮点 min / max / clamp 用 fmin / fmax（遇 NaN 返回另一个参数），四元数 normalize 长度不大于 kEps 时返回单位四元数
# 用法（仓库根）：V2D_ENV=ml40 scripts/dev/record ep7/sim_nan_facts stages/policy/sim_nan_facts.sh
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"
. "$ROOT/scripts/dev/_load-env" >/dev/null
ssh -o ConnectTimeout=20 "$V2D_HOST" 'docker run --rm -i --entrypoint bash flash_chord:latest -s' <<'IN_CONTAINER'
set -uo pipefail
S=/opt/venv/lib/python3.11/site-packages
g() {  # g <文件> <正则>：带行号列出匹配行
  echo "== $1"
  grep -nE "$2" "$S/$1" | sed "s|^|$1:|"
}
count() { echo "$1：匹配 /$2/ 的行数 $(grep -ciE "$2" "$S/$1")"; }
ctx() {  # ctx <文件> <sed 行号范围>
  echo "== $1（原文段落 $2）"
  awk -v f="$1" '{printf "%s:%d:%s\n", f, NR, $0}' "$S/$1" | sed -n "$2"
}
echo "## 1. 版本"
for p in newton mujoco_warp warp_lang; do
  d=$(ls -d "$S"/${p}-*.dist-info 2>/dev/null | head -1)
  echo "$p：$(basename "${d:-未找到}")"
done
echo "## 2. mujoco_warp：约束求解的起点与 warmstart"
ctx mujoco_warp/_src/solver.py '3349,3355p'
g mujoco_warp/_src/forward.py 'wp.copy\(d.qacc_warmstart, d.qacc\)'
count mujoco_warp/_src/solver.py 'isnan|isfinite'
count mujoco_warp/_src/forward.py 'isnan|isfinite'
echo "## 3. Newton 的 SolverMuJoCo.step：状态同步，warmstart 没有被清"
ctx newton/_src/solvers/mujoco/solver_mujoco.py '3266,3289p'
count newton/_src/solvers/mujoco/solver_mujoco.py 'warmstart'
count newton/_src/solvers/mujoco/solver_mujoco.py 'isnan|isfinite'
echo "## 4. Warp 对 NaN 的行为"
ctx warp/native/builtin.h '763,768p'
ctx warp/native/quat.h '213,223p'
g warp/native/builtin.h 'define _wp_native_f(min|max)|_wp_native_fmin\(|_wp_native_fmax\(' | head -12
IN_CONTAINER
