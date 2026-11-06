#!/usr/bin/env bash
# t3b2 第 1 步：第 0 帧朝向表的对照与正式运行（ml40 仓库根执行，放 setsid；只用 CPU，容器里跑）。计划 agent-doc/plan-t3b2-2026-10-28.md 判据 T0 与 B2 规则。
# 前提：runs/t3b2/ 下已有 frefs_{hm,gb5,hr2,hr}（frefs_from_motion_t3.py）、cmp_t3m1022a（已交 t3m1022a 的 hm 那 18 集帧参考与清单）、sym_t3fx.json（t3_sym_dump.py）。
#   A  对照：在已交的帧参考上出表，对齐前的转角差、倾斜、两个偏航与自动方案要与 results/t3align/align_t3m1022a_auto.txt 相同；
#   B  在新出的 hm 帧参考上出表（已交的多一个 4.652° 的世界偏航，倾斜与人体偏航应不变、物体偏航差 4.652°）；
#   C  负对照：gb5 按 obj 方案、--swap-quat 对齐，出表时照常读 Tier 1，对齐输出的第 0 帧差应 > 30°；
#   D  阳性对照：gb5 按 obj 方案正常对齐，对齐输出的第 0 帧差应 ≤ 1°；
#   E–G 正式：gb5、gb5hr2dyn7（ep27）、gb5hrdyn7（ep13）按自动规则对齐、出表、写 B2 选择。
# 日志 runs/logs/t3b2_frame0.log，最后一行 T3B2_FRAME0_DONE 或 T3B2_FRAME0_FAILED <步>。
set -uo pipefail
cd "$(dirname "$0")/../.."
LOG=runs/logs/t3b2_frame0.log
mkdir -p runs/logs
exec >> "$LOG" 2>&1
say() { echo "=== $(date '+%F %T %Z') $*"; }
fail() { say "T3B2_FRAME0_FAILED $*"; exit 1; }
T1=/repo/data/challenge/track_2/tier_1_multiview_caption
B=/repo/runs/t3b2
rec() { local n=$1; shift; scripts/dev/record "ml40/t3b2/$n" scripts/dev/in-flash-chord "cd /repo/stages/evaluate && $*" > /dev/null 2>&1 || fail "$n"; say "$n 过"; }
table() { echo "python t3_frame0_table.py $1 --tier1 $T1 --t3 /repo/runs/tier3 --sym $B/sym_t3fx.json ${2:-}"; }
align() { echo "python align_frefs_t3.py $1 $2 --tier1 $T1 --t3 /repo/runs/tier3 $3"; }
say "开始，来历 $(cat .v2d-source-commit 2>/dev/null || echo unknown)"
for d in frefs_hm frefs_gb5 frefs_hr2 frefs_hr cmp_t3m1022a; do [ -d "runs/t3b2/$d" ] || fail "没有 runs/t3b2/$d"; done
for d in al_gb5_obj_swap al_gb5_obj al_gb5_auto al_hr2_auto al_hr_auto; do [ -e "runs/t3b2/$d" ] && fail "runs/t3b2/$d 已存在"; done
rec table_cmp_ml40 "$(table $B/cmp_t3m1022a)"
rec table_hm_ml40 "$(table $B/frefs_hm)"
rec align_gb5_obj_swap_ml40 "$(align $B/frefs_gb5 $B/al_gb5_obj_swap '--variant obj --swap-quat')"
rec table_gb5_negctl_ml40 "$(table $B/frefs_gb5 "--aligned $B/al_gb5_obj_swap")"
rec align_gb5_obj_ml40 "$(align $B/frefs_gb5 $B/al_gb5_obj '--variant obj')"
rec table_gb5_posctl_ml40 "$(table $B/frefs_gb5 "--aligned $B/al_gb5_obj")"
rec align_gb5_auto_ml40 "$(align $B/frefs_gb5 $B/al_gb5_auto '--variant auto')"
rec table_gb5_ml40 "$(table $B/frefs_gb5 "--aligned $B/al_gb5_auto --choice $B/b2_choice_gb5.json")"
rec align_hr2_auto_ml40 "$(align $B/frefs_hr2 $B/al_hr2_auto '--variant auto')"
rec table_hr2_ml40 "$(table $B/frefs_hr2 "--aligned $B/al_hr2_auto --choice $B/b2_choice_hr2.json")"
rec align_hr_auto_ml40 "$(align $B/frefs_hr $B/al_hr_auto '--variant auto')"
rec table_hr_ml40 "$(table $B/frefs_hr "--aligned $B/al_hr_auto --choice $B/b2_choice_hr.json")"
say "T3B2_FRAME0_DONE"
