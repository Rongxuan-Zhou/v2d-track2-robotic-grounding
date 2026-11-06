#!/usr/bin/env bash
# 冒烟判据（训练机上、仓库根执行）：一条冒烟 run 是否达标。沿用 Day 1 的冒烟判据，另核对镜像指纹与发散。
# 判据：有 DONE；最新一段日志有 “compiled FlashSAC JAX executables”；无 RESOURCE_EXHAUSTED / exceeds the base limit；
#       最新完整 checkpoint 等于预期步数（ckpt_latest.py）；W&B 同步成功；整卡显存峰值 ≤ 上限（<run>.gpu）；
#       最后一次 start 记录的 image_base11_sha / deps_freeze_sha 等于给定值；
#       发散（Tier 2 批量计划 agent-doc/plan-t2-batch-2026-10-01.md 起）：最新一份完整 policy 没有非有限参数
#       （ckpt_nonfinite.py --last --fail-on-nonfinite），最新一段日志的 W&B 摘要里动作均值、熵、回报没有 nan 且 loss_scale 不为 0
#       （train_nan_scan.py --log）。摘要里的回合长度与回报照打、不作判据。
#       W&B 关闭的 run（最新一段 hydra 配置 logging.mode 为 disabled，即 V2D_WANDB=0；10/30 起 W&B 返回 403）不判 W&B 同步与摘要两条，
#       发散只靠 checkpoint 非有限检查（计划 agent-doc/plan-t1palm-2026-10-30.md 第 3 版执行记录）。
# 用法：stages/policy/smoke_check.sh <run> <步数> <显存上限 MiB> <base11> <freeze>
#   例（在 Day 1 已知通过的冒烟上核对这段检查本身）：
#   stages/policy/smoke_check.sh smoke-t1-ep20-w2048-cap10m-ml40 4096000 44000 e2934912858820b8 df0cf35711a68fb3
# 全过退出码 0，最后一行 SMOKE_OK；任何一条不过退出码 1，最后一行 SMOKE_FAIL 与原因。
set -uo pipefail
cd "$(dirname "$0")/../.."
RUN=${1:?用法：smoke_check.sh <run> <步数> <显存上限> <base11> <freeze>}; STEPS=${2:?}; PEAK_MAX=${3:?}; B11=${4:?}; FRZ=${5:?}
S=runs/flash_chord/$RUN
bad() { echo "SMOKE_FAIL $RUN：$*"; exit 1; }
[ -f "$S/DONE" ] || bad "没有 DONE"
L="$(ls runs/logs/"$RUN".s*.log 2>/dev/null | sort -V | tail -1)"
[ -n "$L" ] || bad "没有分段日志"
grep -q "compiled FlashSAC JAX executables" "$L" || bad "$L 没有 compiled FlashSAC JAX executables"
! grep -qE "RESOURCE_EXHAUSTED|exceeds the base limit" "$L" || bad "$L 有显存错误"
got="$(python3 stages/policy/ckpt_latest.py "$S")"
# 训练链开了 V2D_PRUNE_SMOKE_STATE=1 时，冒烟判过之后会删掉它的 state、把步数记进 STATE_PRUNED；重跑判据时按记下的步数认
[ -z "$got" ] && [ -f "$S/STATE_PRUNED" ] && got="$(cat "$S/STATE_PRUNED")"
[ "$got" = "$STEPS" ] || bad "最新完整 checkpoint 是 ${got:-无}，应为 $STEPS"
seg="${L##*.s}"; seg="${seg%.log}"
wbmode="$(awk '/^logging:/ {f = 1; next} f && /^  mode:/ {print $2; exit} f && /^[^ ]/ {exit}' "$S/hydra/s$seg/.hydra/config.yaml" 2>/dev/null)"
[ "$wbmode" = disabled ] || grep -qE "wandb: Synced .*artifact" "$L" || bad "$L 里没有 W&B 同步成功的行"
[ -s "runs/logs/$RUN.gpu" ] || bad "没有显存采样 runs/logs/$RUN.gpu"
peak="$(awk -F', ' '{gsub(/ MiB/, "", $2); if ($2 + 0 > m) m = $2 + 0} END {print m + 0}' "runs/logs/$RUN.gpu")"
[ "$peak" -le "$PEAK_MAX" ] || bad "整卡显存峰值 $peak MiB 超过 $PEAK_MAX"
rec="$(grep '"event": "start"' "$S/run_record.jsonl" | tail -1)"
b11="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["image_base11_sha"])' "$rec")"
frz="$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["deps_freeze_sha"])' "$rec")"
[ "$b11" = "$B11" ] && [ "$frz" = "$FRZ" ] || bad "镜像指纹 $b11 / $frz，应为 $B11 / $FRZ"
# ckpt_nonfinite 只认 10 秒前写完的文件；冒烟刚训完就判，先等最新 policy 写完满 11 秒（最多等 11 秒）
python3 -c 'import os, sys, time; time.sleep(max(0.0, min(11.0, 11.0 - (time.time() - os.path.getmtime(sys.argv[1])))))' "$S/policy_$STEPS.safetensors" 2>/dev/null
nf="$(python3 stages/policy/ckpt_nonfinite.py --last --fail-on-nonfinite "$RUN")" || bad "最新 policy 有非有限参数或不完整：$(tail -1 <<< "$nf")"
if [ "$wbmode" = disabled ]; then ws="W&B 关闭（s$seg 的 logging.mode=disabled）：不判 W&B 同步与摘要，发散只看 checkpoint"
else ws="$(python3 stages/policy/train_nan_scan.py --log "$L")" || bad "W&B 摘要显示发散：$ws"; fi
echo "日志 $L；checkpoint $got；显存峰值 $peak MiB；镜像指纹 $b11 / $frz"
echo "policy：$(tail -1 <<< "$nf" | sed 's/^ *//')；摘要：$ws"
echo "SMOKE_OK $RUN"
