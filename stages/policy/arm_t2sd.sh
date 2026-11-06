#!/usr/bin/env bash
# t2sd 一条龙（mf4090 上、仓库根执行，放 tmux；Mac 只负责启动）。计划 agent-doc/plan-t2sd-2026-10-19.md 第 1 节（复核第 2 条：停链标记由训练机自己建）。
# 1. 配置核对、预检（经 scripts/dev/record 落 results/t2sd/）；2. 起链预演，DRYRUN_OK 才起；3. 起链（arm_var_chain.sh t2msd）；
# 4. 训练链日志出现冒烟开始那一行（“冒烟 smoke-<run>” 结尾）之后建 runs/.chain-stop-t2msd_train_mf4090，正式训练不开；
# 5. 等冒烟 run 有 DONE、训练链写出结束行；6. 起评测链（附加表 stages/evaluate/t2sd_mf4090.tsv）与导出门（清单 cand_t2_t2sd_mf4090.tsv）。
# 日志 runs/logs/arm_t2sd_mf4090.log，最后一行 ARM_T2SD_DONE，或 ARM_T2SD_FAILED <原因>。
set -uo pipefail
cd "$(dirname "$0")/../.."
LOG=runs/logs/arm_t2sd_mf4090.log
mkdir -p runs/logs
exec >> "$LOG" 2>&1
say() { echo "=== $(date '+%F %T %Z') $*"; }
fail() { say "ARM_T2SD_FAILED $*"; exit 1; }
Q=stages/policy/queues/mf4090_t2msd
RUN=t2-ep25-w2048-cap4m-r3a-sd-mf4090
SM="smoke-${RUN}"
T=runs/logs/t2msd_train_mf4090.log
SRC="$HOME/v2d/fc_src/fc_8c778f9"
say "开始，来历 $(cat .v2d-source-commit 2>/dev/null || echo unknown)"
scripts/dev/record t2sd/cfgcheck_mf4090 stages/policy/fc_config_check.sh "${Q}_smoke.txt" "${Q}.txt" > /dev/null 2>&1 || fail "配置核对没过（results/t2sd/cfgcheck_mf4090.txt.failed）"
scripts/dev/record t2sd/precheck_mf4090 stages/policy/precheck_expect.sh "${Q}.expect.tsv" "${Q}_smoke.txt" "${Q}.txt" > /dev/null 2>&1 || fail "预检没过（results/t2sd/precheck_mf4090.txt.failed）"
dry="$(V2D_ARM_DRYRUN=1 bash stages/policy/arm_var_chain.sh t2msd 2>&1)"; tail -n 3 <<< "$dry"
[ "$(tail -n 1 <<< "$dry")" = DRYRUN_OK ] || fail "起链预演没过"
arm="$(bash stages/policy/arm_var_chain.sh t2msd 2>&1)"; tail -n 3 <<< "$arm"
[ "$(tail -n 1 <<< "$arm")" = ARMED ] || fail "起链没成"
for _ in $(seq 1 120); do grep -qE "冒烟 ${SM}\$" "$T" 2>/dev/null && break; sleep 5; done
grep -qE "冒烟 ${SM}\$" "$T" || fail "10 分钟内没等到冒烟开始"
touch runs/.chain-stop-t2msd_train_mf4090
say "冒烟已开始，建了停链标记 runs/.chain-stop-t2msd_train_mf4090"
for _ in $(seq 1 480); do [ -f "runs/flash_chord/${SM}/DONE" ] && break; sleep 5; done
[ -f "runs/flash_chord/${SM}/DONE" ] || fail "40 分钟内冒烟没有 DONE"
say "冒烟 DONE，policy 份数 $(ls runs/flash_chord/${SM}/policy_*.safetensors 2>/dev/null | wc -l)"
for _ in $(seq 1 360); do tail -n 1 "$T" | grep -qE "DAY2_TRAIN_(DONE|FAILED)" && break; sleep 10; done
say "训练链最后一行：$(tail -n 1 "$T" | sed 's/DAY2_TRAIN_/训练链结束：/')"
TMUX= tmux new-session -d -s v2d-t2sd-eval -c "$PWD" "V2D_FC_SRC=$SRC V2D_GPU_WAIT_S=7200 stages/evaluate/day2_eval_chain.sh - stages/evaluate/t2sd_mf4090.tsv t2sd_eval_mf4090"
sleep 2
TMUX= tmux new-session -d -s v2d-t2sd-export -c "$PWD" "V2D_FC_SRC=$SRC V2D_FALLBACK_MAX=0 stages/evaluate/export_after_eval.sh runs/logs/t2sd_eval_mf4090.log stages/evaluate/exports/cand_t2_t2sd_mf4090.tsv cand_t2_t2sd_mf4090"
sleep 5
tmux ls 2>/dev/null | grep t2sd
say ARM_T2SD_DONE
