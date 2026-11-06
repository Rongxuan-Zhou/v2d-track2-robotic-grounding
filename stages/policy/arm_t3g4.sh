#!/usr/bin/env bash
# t3g4 一条龙（mf4090 上、仓库根执行，放 tmux；Mac 只负责启动）。计划 agent-doc/plan-t3g4-2026-10-19.md 第 1 节第 2–4 步，判据 G0。
# 1. 等重定向记录 results/t3gb4/retarget_mf4090.txt 写完（最后一行 EPS_DONE、没有 FAIL）；
# 2. check_inputs.py（容器）核 11 份 motion parquet；3. 支撑面：support_kinds.py 核档目录与训练场景两份，support_import_check.py（p185 副本）核 ep6、7、20；
# 4. 参考差（t3_refdiff_batch.sh，只报）；5. 配置核对、预检、起链预演、起链（arm_var_chain.sh t3pg4）；
# 6. 挂读数等待：run_after_log.sh 等评测链结束行，再跑 gb3_switch.py 的 gb4 规格（results/t3gb4/switch_t3pg4_mf4090.txt）。
# 日志 runs/logs/arm_t3g4_mf4090.log，最后一行 ARM_T3G4_DONE，或 ARM_T3G4_FAILED <原因>。任何一步不过就停，不起链。
set -uo pipefail
cd "$(dirname "$0")/../.."
LOG=runs/logs/arm_t3g4_mf4090.log
mkdir -p runs/logs
exec >> "$LOG" 2>&1
say() { echo "=== $(date '+%F %T %Z') $*"; }
fail() { say "ARM_T3G4_FAILED $*"; exit 1; }
M=runs/motion/tier3_gb4/whole_body
Q=stages/policy/queues/mf4090_t3pg4
REC=results/t3gb4/retarget_mf4090.txt
say "开始，来历 $(cat .v2d-source-commit 2>/dev/null || echo unknown)"
for _ in $(seq 1 240); do [ -f "$REC" ] && break; [ -f "$REC.failed" ] && fail "重定向记录是 .failed"; sleep 15; done
[ -f "$REC" ] || fail "60 分钟内重定向没写完"
[ "$(tail -n 1 "$REC")" = EPS_DONE ] || fail "重定向最后一行不是 EPS_DONE：$(tail -n 1 "$REC")"
grep -q 'FAIL' "$REC" && fail "重定向里有 FAIL：$(grep 'FAIL' "$REC" | head -n 3 | tr '\n' ' ')"
say "重定向 EPS_DONE；ALIGN/NOCONTACT：$(grep -E 'ALIGN|NOCONTACT' "$REC" | tr '\n' ' ' | cut -c1-300)"
n="$(ls $M/soma/sequence_id=*/robot_name=g1/data.parquet 2>/dev/null | wc -l)"
[ "$n" = 11 ] || fail "motion parquet 有 ${n} 份，不是 11"
pq="$(ls $M/soma/sequence_id=*/robot_name=g1/data.parquet | sed 's,^,/repo/,' | tr '\n' ' ')"
scripts/dev/record t3gb4/check_inputs_mf4090 scripts/dev/in-flash-chord "python /repo/stages/policy/check_inputs.py $pq" > /dev/null 2>&1 || fail "check_inputs 没过（results/t3gb4/check_inputs_mf4090.txt.failed）"
scripts/dev/record t3gb4/support_kinds_root_mf4090 python3 stages/policy/support_kinds.py runs/tier3/root_gb4/support_surface > /dev/null 2>&1 || fail "support_kinds（档目录）没过"
scripts/dev/record t3gb4/support_kinds_stage_mf4090 python3 stages/policy/support_kinds.py $M/reconstructed_stage > /dev/null 2>&1 || fail "support_kinds（训练场景）没过"
if ! diff <(cut -f4 results/t3gb4/support_kinds_root_mf4090.txt | grep -E '^[0-9a-f]{32}$' | sort) <(cut -f4 results/t3gb4/support_kinds_stage_mf4090.txt | grep -E '^[0-9a-f]{32}$' | sort) > /dev/null; then
  fail "档目录与训练场景的支撑面 md5 不一致"
fi
us=""
for s in 2026-06-01_11-26-44_monitor_rotate_05 2026-05-29_15-20-43_iron_round_table_02 2026-06-08_16-57-34_rice_cooker_clockwise_04; do
  f="$M/reconstructed_stage/${s}_support.usda"; [ -f "$f" ] || fail "没有 $f"; us="$us /repo/$f"
done
scripts/dev/record t3gb4/support_import_p185_mf4090 env V2D_ALLOW_FC_SRC=1 V2D_FC_SRC="$HOME/v2d/fc_src/v040_p185_nfg" scripts/dev/in-flash-chord "python /repo/stages/policy/support_import_check.py$us" > /dev/null 2>&1 \
  || fail "ep6、7、20 的桌面在训练场景里建不出来（results/t3gb4/support_import_p185_mf4090.txt.failed）"
scripts/dev/record t3gb4/refdiff_mf4090 stages/evaluate/t3_refdiff_batch.sh stages/evaluate/exports/t3_gb4_refdiff_pairs.tsv > /dev/null 2>&1 || say "参考差没跑完（只报，不挡）"
scripts/dev/record t3gb4/cfgcheck_t3pg4_mf4090 stages/policy/fc_config_check.sh "${Q}_smoke.txt" "${Q}.txt" > /dev/null 2>&1 || fail "配置核对没过"
scripts/dev/record t3gb4/precheck_t3pg4_mf4090 stages/policy/precheck_expect.sh "${Q}.expect.tsv" "${Q}_smoke.txt" "${Q}.txt" > /dev/null 2>&1 || fail "预检没过"
dry="$(V2D_ARM_DRYRUN=1 bash stages/policy/arm_var_chain.sh t3pg4 2>&1)"; tail -n 3 <<< "$dry"
[ "$(tail -n 1 <<< "$dry")" = DRYRUN_OK ] || fail "起链预演没过"
arm="$(bash stages/policy/arm_var_chain.sh t3pg4 2>&1)"; tail -n 3 <<< "$arm"
[ "$(tail -n 1 <<< "$arm")" = ARMED ] || fail "起链没成"
V2D_POLL_S=60 V2D_MAX_POLLS=2400 setsid nohup stages/policy/run_after_log.sh runs/logs/t3pg4_eval_mf4090.log "DAY2_EVAL_(DONE|FAILED)" \
  t3gb4/switch_t3pg4_mf4090 python3 stages/evaluate/gb3_switch.py stages/evaluate/gb4_switch_t3pg4.json > runs/logs/after_switch_t3pg4_mf4090.log 2>&1 < /dev/null &
sleep 3
say "读数等待已挂：$(head -n 1 runs/logs/after_switch_t3pg4_mf4090.log | cut -c1-200)"
say ARM_T3G4_DONE
