#!/usr/bin/env bash
# t3g5 起链（训练机上、仓库根执行，放 tmux；Mac 只负责启动）。计划 agent-doc/plan-t3g5-2026-10-21.md 第 2 版第 1 节第 2–3 步、判据 R0。
# mf4090：A 段 t3pg5a（4 集）、B 段 t3pg5b（5 集）；ml40：A 段 t3qg5a（3 集）、B 段 t3qg5b（6 集）。
# 1. 这台机器两段队列里的 motion parquet 都在（runs/motion/tier3_gb5；mf4090 由重定向写、ml40 由 tier3_relay.sh 转来）；
# 2. check_inputs.py（容器）核这些 parquet；3. 开支撑面的集：训练场景里的 usda 在、support_import_check.py（p185 副本）建得出来；
#    mf4090 另核档目录与训练场景两份支撑面的 md5（support_kinds.py）；
# 4. A、B 两段的配置核对与预检；5. 起 A 段（预演 DRYRUN_OK 再 ARMED）；
# 6. B 段：arm_after.sh（复制到 runs/tools/、记 md5，V2D_AFTER_INDEPENDENT=1，setsid）等 A 段训、评、导出都结束、卡空再起；
# 7. 两段各挂读数等待：run_after_log.sh 等评测链结束行，跑 stages/evaluate/t3g5_readout.sh <段> <机器>（记 results/t3g5/）。
# 不停任何进程；任何一步不过就停，不起链。日志 runs/logs/arm_t3g5_<机器>.log，最后一行 ARM_T3G5_DONE，或 ARM_T3G5_FAILED <原因>。
# 用法：stages/policy/arm_t3g5.sh mf4090|ml40
set -uo pipefail
cd "$(dirname "$0")/../.."
MACH=${1:?用法：arm_t3g5.sh mf4090|ml40}
case "$MACH" in
  mf4090) A=t3pg5a; B=t3pg5b; FC="$HOME/v2d/fc_src" ;;
  ml40) A=t3qg5a; B=t3qg5b; FC=/lab_data/rxzhou/v2d/fc_src ;;
  *) echo "arm_t3g5: 机器要是 mf4090 或 ml40" >&2; exit 2 ;;
esac
LOG="runs/logs/arm_t3g5_${MACH}.log"
mkdir -p runs/logs runs/tools results/t3g5
exec >> "$LOG" 2>&1
say() { echo "=== $(date '+%F %T %Z') $*"; }
fail() { say "ARM_T3G5_FAILED $*"; exit 1; }
SRCC="$(cat .v2d-source-commit 2>/dev/null || echo unknown)"
case "$SRCC" in *-syncing|*-dirty|unknown) fail "来历标记是 ${SRCC}" ;; esac
M=runs/motion/tier3_gb5/whole_body
QA="stages/policy/queues/${MACH}_${A}"; QB="stages/policy/queues/${MACH}_${B}"
say "开始，来历 ${SRCC}；A 段 ${A}，B 段 ${B}"
for f in "$QA.txt" "$QA.expect.tsv" "${QA}_smoke.txt" "$QB.txt" "$QB.expect.tsv" "${QB}_smoke.txt"; do [ -s "$f" ] || fail "没有 $f"; done
# 1. parquet 都在
mapfile -t PQ < <(grep -hv '^#' "$QA.txt" "$QB.txt" | tr ' ' '\n' | grep '^/repo/runs/motion/tier3_gb5/' | sed 's,^/repo/,,')
[ "${#PQ[@]}" -ge 1 ] || fail "队列里找不到 tier3_gb5 的 parquet"
for p in "${PQ[@]}"; do [ -f "$p" ] || fail "没有 $p"; done
say "${#PQ[@]} 份 motion parquet 都在"
# 2. check_inputs
scripts/dev/record "t3g5/check_inputs_${MACH}" scripts/dev/in-flash-chord "python /repo/stages/policy/check_inputs.py $(printf '/repo/%s ' "${PQ[@]}")" > /dev/null 2>&1 \
  || fail "check_inputs 没过（results/t3g5/check_inputs_${MACH}.txt.failed）"
# 3. 支撑面：开支撑面的行（没有 scene.support=false）
us=""; n=0
while read -r line; do
  p="$(tr ' ' '\n' <<< "$line" | grep '^/repo/runs/motion/tier3_gb5/' | sed 's,^/repo/,,')"
  seq="$(sed -E 's,.*/sequence_id=([^/]+)/robot_name=g1/data.parquet,\1,' <<< "$p")"
  f="$M/reconstructed_stage/${seq}_support.usda"; [ -f "$f" ] || fail "没有 $f"
  us="$us /repo/$f"; n=$((n + 1))
done < <(grep -hv '^#' "$QA.txt" "$QB.txt" | grep -v 'scene.support=false')
if [ "$n" -gt 0 ]; then
  scripts/dev/record "t3g5/support_import_p185_${MACH}" env V2D_ALLOW_FC_SRC=1 V2D_FC_SRC="$FC/v040_p185_nfg" scripts/dev/in-flash-chord \
    "python /repo/stages/policy/support_import_check.py$us" > /dev/null 2>&1 || fail "开支撑面的 ${n} 集在训练场景里建不出来（results/t3g5/support_import_p185_${MACH}.txt.failed）"
fi
say "开支撑面的 ${n} 集在训练场景里建得出来"
if [ -d runs/tier3/root_gb5/support_surface ]; then
  scripts/dev/record "t3g5/support_kinds_root_${MACH}" python3 stages/policy/support_kinds.py runs/tier3/root_gb5/support_surface > /dev/null 2>&1 || fail "support_kinds（档目录）没过"
  scripts/dev/record "t3g5/support_kinds_stage_${MACH}" python3 stages/policy/support_kinds.py "$M/reconstructed_stage" > /dev/null 2>&1 || fail "support_kinds（训练场景）没过"
  if ! diff <(cut -f4 "results/t3g5/support_kinds_root_${MACH}.txt" | grep -E '^[0-9a-f]{32}$' | sort) \
            <(cut -f4 "results/t3g5/support_kinds_stage_${MACH}.txt" | grep -E '^[0-9a-f]{32}$' | sort) > /dev/null; then
    fail "档目录与训练场景的支撑面 md5 不一致"
  fi
  say "档目录与训练场景的支撑面 md5 一致"
fi
# 4. 两段的配置核对与预检
for S in "$A" "$B"; do
  Q="stages/policy/queues/${MACH}_${S}"
  scripts/dev/record "t3g5/cfgcheck_${S}_${MACH}" stages/policy/fc_config_check.sh "${Q}_smoke.txt" "${Q}.txt" > /dev/null 2>&1 || fail "${S} 配置核对没过"
  scripts/dev/record "t3g5/precheck_${S}_${MACH}" stages/policy/precheck_expect.sh "${Q}.expect.tsv" "${Q}_smoke.txt" "${Q}.txt" > /dev/null 2>&1 || fail "${S} 预检没过"
done
say "A、B 两段配置核对与预检都过"
# 5. 起 A 段
dry="$(V2D_ARM_DRYRUN=1 bash stages/policy/arm_var_chain.sh "$A" 2>&1)"; tail -n 3 <<< "$dry"
[ "$(tail -n 1 <<< "$dry")" = DRYRUN_OK ] || fail "${A} 起链预演没过"
arm="$(bash stages/policy/arm_var_chain.sh "$A" 2>&1)"; tail -n 3 <<< "$arm"
[ "$(tail -n 1 <<< "$arm")" = ARMED ] || fail "${A} 起链没成"
# 6. B 段接在 A 段后面
NA="$(grep -hv '^#' "$QA.txt" | grep -c .)"
cp stages/policy/arm_after.sh runs/tools/arm_after.sh && say "arm_after.sh md5 $(md5sum runs/tools/arm_after.sh | cut -c1-32)"
V2D_AFTER_INDEPENDENT=1 setsid nohup bash runs/tools/arm_after.sh "$B" "runs/logs/${A}_train_${MACH}.log" "runs/logs/${A}_eval_${MACH}.log" \
  "runs/logs/cand_t3_${A}_${MACH}.log" "$NA" > "runs/logs/after_${B}_${MACH}.log" 2>&1 < /dev/null &
sleep 3
say "B 段的等待已挂：$(head -n 2 "runs/logs/after_${B}_${MACH}.log" | tr '\n' ' ' | cut -c1-200)"
# 7. 两段的读数等待
for S in "$A" "$B"; do
  V2D_POLL_S=60 V2D_MAX_POLLS=3000 setsid nohup stages/policy/run_after_log.sh "runs/logs/${S}_eval_${MACH}.log" "DAY2_EVAL_(DONE|FAILED)" \
    "t3g5/readout_${S}_${MACH}" bash stages/evaluate/t3g5_readout.sh "$S" "$MACH" > "runs/logs/after_readout_${S}_${MACH}.log" 2>&1 < /dev/null &
done
sleep 3
say "读数等待已挂：$(head -n 1 "runs/logs/after_readout_${A}_${MACH}.log" | cut -c1-160)"
say ARM_T3G5_DONE
