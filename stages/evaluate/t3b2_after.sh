#!/usr/bin/env bash
# t3b2 第 2、3 步的后续串联（ml40 仓库根执行，放 setsid；不停任何进程）。计划 agent-doc/plan-t3b2-2026-10-28.md 第 2 步第 6、7 条、第 3 步。
#   1. 等 ep27 的 B2 下游链（t3fx_build.sh 27，集名 gb5hr2b2dyn7）写出 T3FX_BUILD_DONE；把 gate_e 里的 motion 与 motion_src 复制进本仓（已存在就停）、
#      parquet_repath、check_inputs；
#   2. 等 ep15 的重新评测导出（t3b2_reexport.sh _hr3s）写出 T3B2_REEXPORT_DONE；对 ep27 的新参考重新评测导出 t3-ep27-w2048-cap10m-r3a-gb5hr2dyn7-ml40（后缀 _b2）；
#   3. arm_seg.sh t3qhr3s ml40（ep15 两个种子，配置核对、预检、预演、起链）。
# 日志 runs/logs/t3b2_after.log，最后一行 T3B2_AFTER_DONE 或 T3B2_AFTER_FAILED <步>。
set -uo pipefail
cd "$(dirname "$0")/../.."
LOG=runs/logs/t3b2_after.log
mkdir -p runs/logs
exec >> "$LOG" 2>&1
say() { echo "=== $(date '+%F %T %Z') $*"; }
fail() { say "T3B2_AFTER_FAILED $*"; exit 1; }
G=/lab_data/rxzhou/v2d/t3recon/gate_e/repo
say "开始，来历 $(cat .v2d-source-commit 2>/dev/null)"
for i in $(seq 1 240); do
  last=$(grep -E "开始|T3FX_BUILD_(DONE|FAILED)" runs/logs/t3fx_build.log | tail -1)
  case "$last" in *T3FX_BUILD_DONE*) break ;; *T3FX_BUILD_FAILED*) fail "ep27 下游链失败：$last" ;; esac
  sleep 30
done
case "$last" in *T3FX_BUILD_DONE*) ;; *) fail "ep27 下游链 2 小时没完" ;; esac
grep -q "ep27 重定向结果行" <(grep -A0 "ep27 重定向结果行" runs/logs/t3fx_build.log | tail -1) || fail "没有 ep27 的重定向结果行"
say "ep27 下游链完：$(grep "ep27 重定向结果行" runs/logs/t3fx_build.log | tail -1)"
for d in motion motion_src; do [ -e "runs/$d/tier3_gb5hr2b2dyn7" ] && fail "runs/$d/tier3_gb5hr2b2dyn7 已存在"; cp -a "$G/runs/$d/tier3_gb5hr2b2dyn7" "runs/$d/" || fail "复制 $d"; done
P=$(ls runs/motion/tier3_gb5hr2b2dyn7/whole_body/soma/*/robot_name=g1/data.parquet) || fail "没有 ep27 B2 的 parquet"
scripts/dev/record ml40/t3b2/repath_hr2b2_ml40 scripts/dev/in-flash-chord "python /repo/stages/retarget/parquet_repath.py /repo/$P $G /repo" > /dev/null 2>&1 || fail repath
scripts/dev/record ml40/t3b2/check_inputs_hr2b2_ml40 scripts/dev/in-flash-chord "python /repo/stages/policy/check_inputs.py /repo/$P" > /dev/null 2>&1 || fail check_inputs
say "ep27 B2 参考就位：$P"
for i in $(seq 1 240); do grep -q "T3B2_REEXPORT_DONE" runs/logs/t3b2_reexport.log && break; sleep 30; done
grep -q "T3B2_REEXPORT_DONE" runs/logs/t3b2_reexport.log || fail "ep15 重新导出 2 小时没完"
say "ep15 重新导出：$(grep T3B2_REEXPORT_DONE runs/logs/t3b2_reexport.log | tail -1)"
bash stages/evaluate/t3b2_reexport.sh _b2 "/repo/$P" t3-ep27-w2048-cap10m-r3a-gb5hr2dyn7-ml40
say "ep27 重新导出：$(grep T3B2_REEXPORT_DONE runs/logs/t3b2_reexport.log | tail -1)"
bash stages/policy/arm_seg.sh t3qhr3s ml40
say "起链：$(tail -1 runs/logs/arm_t3qhr3s_ml40.log)"
grep -q ARM_SEG_DONE runs/logs/arm_t3qhr3s_ml40.log || fail arm
say "T3B2_AFTER_DONE"
