#!/usr/bin/env bash
# 回放速度对照（Day 1b 诊断触发条件 T2，与 Tier 2 无关）ml40 训练链：ep20、满配（1000 万 buffer）、按真实时长回放（921 帧）。
# 计划与事先登记：agent-doc/plan-playback-2026-09-28.md（第 2 版）。在 ml40 上执行，放 tmux，不依赖 Mac 在线；任何一步不符即停。
#   0. 守卫：snack 对照链已结束；GPU 上没有别的进程；parquet 与 Day 1 是同一份；正式 run 不处于“有 checkpoint、没有 DONE”；展开配置正向核对
#   1. 冒烟（queues/ml40_playback_smoke.txt）与冒烟判据（显存峰值、镜像指纹）
#   2. 正式训练（queues/ml40_playback.txt）
#   3. 最终 policy 的 4096 世界评测（步数应为 921）与输入清单（经 scripts/dev/record）
# 真实时长参考的核对与诊断链试跑已在 mf4090 上做过：同一份 parquet、同样的回放设置，参考相同，这里不重复。
# 用法（仓库根）：tmux new-session -d -s v2d-playback stages/policy/queues/ml40_playback.sh
# 日志 runs/logs/ml40_playback.log，最后一行 PLAYBACK_TRAIN_DONE 或 PLAYBACK_FAILED。
set -Eeuo pipefail   # -E：函数里的失败也触发 ERR trap
cd "$(dirname "$0")/../../.."
mkdir -p runs/logs results/playback
exec >> runs/logs/ml40_playback.log 2>&1
say() { echo "=== $(date '+%F %T %Z') $*"; }
fail() { say "$*"; say PLAYBACK_FAILED; exit 1; }
trap 'say "PLAYBACK_FAILED（第 $LINENO 行命令失败）"' ERR
SMOKE=smoke-t1-ep20-w2048-cap10m-rt-ml40; RUN=t1-ep20-w2048-cap10m-rt-ml40; STEPS=349999104
PARQ=runs/motion/tier1/whole_body/soma/sequence_id=2026-06-08_16-57-34_rice_cooker_clockwise_04/robot_name=g1/data.parquet
PARQ_SHA16=9e3503b5396b42a4                                   # Day 1 两条 ep20 用的同一份
DAY1_BASE11=e2934912858820b8; DAY1_FREEZE=df0cf35711a68fb3    # Day 1 run_record.jsonl 里的镜像指纹（ml40 重建的镜像与之相同）
PEAK_MAX_MIB=44000                                            # 沿用 Day 1 / Day 1b 在 ml40 上的满配冒烟判据
say "start, source $(cat .v2d-source-commit 2>/dev/null || echo unknown)"

# snack 对照链必须已经结束（它与本链共用这张卡）
tmux has-session -t v2d-day1b 2>/dev/null && fail "snack 对照链（tmux v2d-day1b）还在，不跑"
tail -1 runs/logs/ml40_day1b.log | grep -qE "DAY1B_(DONE|FAILED)" || fail "snack 对照链的日志最后一行不是 DAY1B_DONE / DAY1B_FAILED"

# 0. 守卫
others="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')"
[ -z "$others" ] || fail "GPU 上有别的进程（$others），不跑"
[ "$(sha256sum "$PARQ" | cut -c1-16)" = "$PARQ_SHA16" ] || fail "$PARQ 与 Day 1 用的不是同一份"
R=runs/flash_chord/$RUN
if [ ! -f "$R/DONE" ] && compgen -G "$R/state_*.safetensors" > /dev/null; then
  fail "$R 有 checkpoint 却没有 DONE：续训不恢复 replay，要从 0 重来就先移走该目录"
fi
# 正向核对：配置核对展开的两份配置里确实是真实时长回放、放宽后的校验容差（核对只拦意外的差异，漏写覆盖拦不住）
for r in "$SMOKE" "$RUN"; do
  y=runs/cfgcheck/$r.yaml
  [ -f "$y" ] || fail "缺 $y：先在本机跑 fc_config_check.sh"
  grep -q "source_frame_playback: false" "$y" && grep -q "frame_position_tolerance: 0.002" "$y" \
    && grep -q "frame_orientation_tolerance: 0.005" "$y" || fail "$y 里不是真实时长回放或容差不对"
done

# 1. 冒烟
S=runs/flash_chord/$SMOKE
[ -f "$S/DONE" ] || stages/policy/fc_queue.sh stages/policy/queues/ml40_playback_smoke.txt
stages/policy/smoke_check.sh "$SMOKE" 4096000 "$PEAK_MAX_MIB" "$DAY1_BASE11" "$DAY1_FREEZE" || fail "冒烟判据没过（原因见上一行）"

# 2. 正式训练
others="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')"
[ -z "$others" ] || fail "GPU 上有别的进程（$others），不开正式训练"
stages/policy/fc_queue.sh stages/policy/queues/ml40_playback.txt
[ -f "$R/DONE" ] || fail "正式训练没有 DONE"
segs="$(grep -c '"event": "start"' "$R/run_record.jsonl")"
if [ "$segs" = 1 ]; then say "正式训练一段跑完"; else say "注意：正式训练有 $segs 段，按登记只列数字、不判定"; fi

# 3. 最终 policy 的评测与输入清单（步数应为 921）
scripts/dev/record "playback/eval_$RUN" stages/policy/fc_eval.sh "$RUN" "$STEPS"
n="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["step_count"])' "$R/eval_${STEPS}_w4096.json")"
[ "$n" = 921 ] || fail "正式评测步数 $n，不是 921"
scripts/dev/record "playback/policy_inputs_$RUN" python3 stages/policy/policy_inputs.py "$R/policy_$STEPS.safetensors"
say PLAYBACK_TRAIN_DONE
