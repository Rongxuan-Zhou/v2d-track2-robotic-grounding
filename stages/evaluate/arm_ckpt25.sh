#!/usr/bin/env bash
# 挂 10/18 白天的两个 arm_after（训练机上、仓库根执行）。计划 agent-doc/plan-ep25-sweep-seeds-2026-10-18.md 第 4 节，命令与计划里的原样相同：
#   第一个：等 gb3 那一段（mf4090 t3pg、ml40 t3qg）的链结束、卡空 → 前置命令 ep25 扫描（ckpt_sweep.sh）→ 起 Tier 1 补 seed 的 s44 段；
#   第二个：等 s44 段训练、评测、导出全部做完 → 前置命令 seed_gate.py → 起 s45 段。
# 起之前核（不过就退出码 1，什么都不起）：runs/tools/arm_after.sh 的 md5 是 07173c9b1013c1dd46752b861fb5b3e1；两份 arm_after 日志都不存在；
# 没有在跑的 arm_after 进程；gb3 那一段的训练、评测日志在；同一串扫描前置命令加 V2D_SWEEP_DRYRUN=1 预演最后一行是 CKPT_SWEEP_DRYRUN_OK。
# 起的时候用 env -u 去掉 V2D_SWEEP_DRYRUN、V2D_ARM_DRYRUN（带着它们，扫描或起链会变成什么都不做）。
# 用法：stages/evaluate/arm_ckpt25.sh mf4090|ml40       V2D_ARM25_DRYRUN=1：核完只打印两条命令，不起。最后一行 ARMED_2 或 ARM25_DRYRUN_OK。
set -uo pipefail
cd "$(dirname "$0")/../.."
M=${1:?用法：arm_ckpt25.sh mf4090|ml40}
case "$M" in
  mf4090) SW='V2D_FC_SRC=$HOME/v2d/fc_src/fc_8c778f9 stages/evaluate/ckpt_sweep.sh ckpt25_mf4090 stages/evaluate/ckpt25_coarse_mf4090.tsv 100'
          TO=28800; P=t3pg; N1=t1xb_s44; N2=t1xb_s45
          G='python3 stages/evaluate/seed_gate.py t1-ep14-w2048-cap4m-r3a-s44-mf4090 14 maniptrans_sr 0.3' ;;
  ml40)   SW='V2D_FC_SRC=/lab_data/rxzhou/v2d/fc_src/fc_8c778f9 stages/evaluate/ckpt_sweep.sh ckpt25_ml40 stages/evaluate/ckpt25_coarse_ml40.tsv 100'
          TO=36000; P=t3qg; N1=t1c_s44; N2=t1c_s45
          G='python3 stages/evaluate/seed_gate.py t1-ep11-w2048-cap10m-r3a-c10-s44-ml40 11 spider_sr_uncentered 0.4' ;;
  *) echo "机器只能是 mf4090 或 ml40" >&2; exit 2 ;;
esac
L1="runs/logs/arm_after_${N1}_$M.log"; L2="runs/logs/arm_after_${N2}_$M.log"
stop() { echo "arm_ckpt25: $*，不起" >&2; exit 1; }
[ "$(md5sum runs/tools/arm_after.sh 2>/dev/null | cut -d' ' -f1)" = 07173c9b1013c1dd46752b861fb5b3e1 ] || stop "runs/tools/arm_after.sh 的 md5 不对"
for f in "$L1" "$L2"; do [ ! -e "$f" ] || stop "$f 已存在"; done
[ -z "$(pgrep -f '[a]rm_after\.sh' || true)" ] || stop "已有在跑的 arm_after：$(pgrep -af '[a]rm_after\.sh' | tr '\n' ' ')"
for f in "runs/logs/${P}_train_$M.log" "runs/logs/${P}_eval_$M.log"; do [ -f "$f" ] || stop "没有 $f"; done
dry="$(env V2D_SWEEP_DRYRUN=1 bash -c "$SW" 2>&1)"; [ "$(tail -1 <<< "$dry")" = CKPT_SWEEP_DRYRUN_OK ] || stop "扫描预演没过：$(tail -2 <<< "$dry" | tr '\n' ' ')"
C1=(env -u V2D_SWEEP_DRYRUN -u V2D_ARM_DRYRUN V2D_AFTER_INDEPENDENT=1 V2D_POLL_S=60 V2D_AFTER_MAX_POLLS=720 V2D_PRE_TIMEOUT_S="$TO" V2D_PRE_CMD="$SW"
    setsid nohup bash runs/tools/arm_after.sh "$N1" "runs/logs/${P}_train_$M.log" "runs/logs/${P}_eval_$M.log" "runs/logs/cand_t3_${P}_$M.log" 4)
C2=(env -u V2D_SWEEP_DRYRUN -u V2D_ARM_DRYRUN V2D_POLL_S=60 V2D_AFTER_MAX_POLLS=1440 V2D_PRE_CMD="$G"
    setsid nohup bash runs/tools/arm_after.sh "$N2" "runs/logs/${N1}_train_$M.log" "runs/logs/${N1}_eval_$M.log" "runs/logs/cand_t1_${N1}_$M.log" 1)
echo "第一个：$(printf '%q ' "${C1[@]}")> $L1"
echo "第二个：$(printf '%q ' "${C2[@]}")> $L2"
if [ "${V2D_ARM25_DRYRUN:-0}" = 1 ]; then echo ARM25_DRYRUN_OK; exit 0; fi
"${C1[@]}" > "$L1" 2>&1 < /dev/null &
"${C2[@]}" > "$L2" 2>&1 < /dev/null &
sleep 5
head -2 "$L1" | cut -c1-240; head -2 "$L2" | cut -c1-240
n="$(pgrep -f '[a]rm_after\.sh' | wc -l)"
[ "$n" = 2 ] && grep -q '^=== .* 等 runs/.chain-active-' "$L1" && grep -q '^=== .* 等 runs/.chain-active-' "$L2" \
  || { echo "arm_ckpt25: 起完核不过（arm_after 进程 ${n} 个），去看两份日志" >&2; exit 1; }
echo ARMED_2
