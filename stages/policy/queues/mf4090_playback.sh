#!/usr/bin/env bash
# 回放速度对照（Day 1b 诊断触发条件 T2，与 Tier 2 无关）mf4090 训练链：ep20、100 万 buffer、按真实时长回放（921 帧）。
# 计划与事先登记：agent-doc/plan-playback-2026-09-28.md。在 mf4090 上执行，放 tmux，不依赖 Mac 在线；任何一步不符即停。
#   0. 守卫：GPU 上没有别的进程；parquet 与 Day 1 是同一份；正式 run 不处于“有 checkpoint、没有 DONE”
#   1. 冒烟（queues/mf4090_playback_smoke.txt）与冒烟判据
#   2. 参考核对：冒烟 policy 的 static_baseline → playback_refcheck.py（同时给出诊断运行表的一行）→ 冒烟 256 世界评测（步数应为 921）
#      → 在冒烟 policy 上把逐帧诊断整条链试跑一遍（256 世界）：921 帧的 rollout 在正式训练前先过一遍所有闸
#   3. 正式训练（queues/mf4090_playback.txt）
#   4. 最终 policy 的 4096 世界评测与输入清单（经 scripts/dev/record）。逐帧诊断、各 checkpoint 的评测与判定由分析脚本另做
# 用法（仓库根）：tmux new-session -d -s v2d-playback stages/policy/queues/mf4090_playback.sh
# 日志 runs/logs/mf4090_playback.log，最后一行 PLAYBACK_TRAIN_DONE 或 PLAYBACK_FAILED。
set -Eeuo pipefail   # -E：函数里的失败也触发 ERR trap
cd "$(dirname "$0")/../../.."
mkdir -p runs/logs results/playback
exec >> runs/logs/mf4090_playback.log 2>&1
say() { echo "=== $(date '+%F %T %Z') $*"; }
fail() { say "$*"; say PLAYBACK_FAILED; exit 1; }
trap 'say "PLAYBACK_FAILED（第 $LINENO 行命令失败）"' ERR
SMOKE=smoke-t1-ep20-w2048-cap1m-rt-mf4090; RUN=t1-ep20-w2048-cap1m-rt-mf4090; STEPS=349999104
PARQ=runs/motion/tier1/whole_body/soma/sequence_id=2026-06-08_16-57-34_rice_cooker_clockwise_04/robot_name=g1/data.parquet
PARQ_SHA16=9e3503b5396b42a4                                   # Day 1 两条 ep20 用的同一份
DAY1_BASE11=e2934912858820b8; DAY1_FREEZE=df0cf35711a68fb3    # Day 1 mf4090 run_record.jsonl 里的镜像指纹
PEAK_MAX_MIB=14000                                            # Day 1 同配置峰值 11,555 MiB（results/playback/day1_mf4090_peak.txt）
say "start, source $(cat .v2d-source-commit 2>/dev/null || echo unknown)"

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
[ -f "$S/DONE" ] || stages/policy/fc_queue.sh stages/policy/queues/mf4090_playback_smoke.txt
stages/policy/smoke_check.sh "$SMOKE" 4096000 "$PEAK_MAX_MIB" "$DAY1_BASE11" "$DAY1_FREEZE" || fail "冒烟判据没过（原因见上一行）"

# 2. 参考核对与诊断链试跑
scripts/dev/record playback/static_baseline_ep20_rt scripts/dev/in-flash-chord \
  "python /repo/stages/evaluate/static_baseline.py /repo/$S/policy_4096000.safetensors"
scripts/dev/record playback/refcheck_ep20_rt python3 stages/evaluate/playback_refcheck.py \
  results/day1/static_baseline_ep20.txt results/playback/static_baseline_ep20_rt.txt --row "$SMOKE" 4096000 - \
  || fail "参考核对没过（见 results/playback/refcheck_ep20_rt.txt.failed）"
scripts/dev/record "playback/eval_${SMOKE}_w256" env V2D_EVAL_WORLDS=256 stages/policy/fc_eval.sh "$SMOKE" 4096000
n="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["step_count"])' "$S/eval_4096000_w256.json")"
[ "$n" = 921 ] || fail "冒烟评测步数 $n，不是 921"
T=results/playback/diag_smoke_rt.runs
sed -n 's/^diag_row //p' results/playback/refcheck_ep20_rt.txt > "$T"
[ "$(wc -l < "$T")" = 1 ] || fail "参考核对没有给出诊断运行表的一行"
V2D_EVAL_WORLDS=256 V2D_DIAG_RECORD_DIR=playback stages/evaluate/diag_day1b.sh "$T" || fail "冒烟 policy 上的诊断链试跑没过"

# 3. 正式训练
others="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')"
[ -z "$others" ] || fail "GPU 上有别的进程（$others），不开正式训练"
stages/policy/fc_queue.sh stages/policy/queues/mf4090_playback.txt
[ -f "$R/DONE" ] || fail "正式训练没有 DONE"
segs="$(grep -c '"event": "start"' "$R/run_record.jsonl")"
if [ "$segs" = 1 ]; then say "正式训练一段跑完"; else say "注意：正式训练有 $segs 段，按登记只列数字、不判定"; fi

# 4. 最终 policy 的评测与输入清单
scripts/dev/record "playback/eval_$RUN" stages/policy/fc_eval.sh "$RUN" "$STEPS"
n="$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["step_count"])' "$R/eval_${STEPS}_w4096.json")"
[ "$n" = 921 ] || fail "正式评测步数 $n，不是 921"
scripts/dev/record "playback/policy_inputs_$RUN" python3 stages/policy/policy_inputs.py "$R/policy_$STEPS.safetensors"
say PLAYBACK_TRAIN_DONE
