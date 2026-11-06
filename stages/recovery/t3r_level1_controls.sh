#!/usr/bin/env bash
# t3takeover 第 1 级 A 部分的两项补跑（训练机 ml40 上执行，放 tmux；只用 CPU）。计划 agent-doc/plan-t3takeover-v3-2026-10-21.md 判据 K1（正对照）与 K2 登记第 3 条。
#   1. t3_refw：种根（输入 + 参照的 decode、focal、world 产物），从 assemble 往后跑，与参照比——ep27 的下游段不被 world 的不同遮住；
#   2. t3_rep：种根（只有输入），从 decode 起整条重跑，与 t3 互比——正对照（同步骤跑两遍）。
# 每步失败只记下、不拦后面的步（两项互不依赖）。给了等待标记就先等 runs/logs/t3rw_ml40.log 里出现它（world 重复跑的链跑完再开始，免得互相抢 CPU）。
# 用法：stages/recovery/t3r_level1_controls.sh [<等待标记>]   日志 runs/logs/t3r_level1_controls.log，最后一行 T3R_CONTROLS_DONE。
set -uo pipefail
cd "$(dirname "$0")/../.."
exec >> runs/logs/t3r_level1_controls.log 2>&1
say() { echo "=== $(date '+%F %T %Z') $*"; }
if [ -n "${1:-}" ]; then
  say "等 runs/logs/t3rw_ml40.log 里出现 $1"
  until grep -q "$1" runs/logs/t3rw_ml40.log 2>/dev/null; do sleep 30; done
fi
step() { local n=$1; shift; say "开始 $n"; "$@"; say "$n 退出码 $?"; }
step seed_t3_refw nice -n 10 scripts/dev/record t3recon/seed_t3_refw stages/recovery/t3r_seed_root.sh t3_refw world-from-ref
step level1_t3_refw env T3R_FROM=assemble nice -n 10 stages/recovery/t3r_level1.sh t3_refw
step compare_t3_refw nice -n 10 stages/recovery/t3r_level1_compare.sh t3_refw
step seed_t3_rep nice -n 10 scripts/dev/record t3recon/seed_t3_rep stages/recovery/t3r_seed_root.sh t3_rep inputs
step level1_t3_rep nice -n 10 stages/recovery/t3r_level1.sh t3_rep
step compare_t3_rep nice -n 10 stages/recovery/t3r_level1_compare.sh t3_rep t3
say T3R_CONTROLS_DONE
