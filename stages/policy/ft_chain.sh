#!/usr/bin/env bash
# 续训微调的短链（mf4090 上、仓库根、tmux 里跑）。计划 agent-doc/plan-t2ft-2026-10-26.md（t2ft）、agent-doc/plan-t2ft2-2026-10-26.md（t2ft2）。
# 每段的配置在 stages/policy/queues/mf4090_<段>.ft.sh：队列、步数、起点 sha256、每条 run 的起点与课程第 8 段、终止阈值、先评的中间 checkpoint。
# 现有训练链要求正式步数 349999104、冒烟从 0 步起训，续训两样都不满足，所以单写这一条：
#   1. 起点核对：两份起点 state 的 sha256 与计划登记的相同（续训期间不动起点目录，结束时再核一次）；
#   2. 配置核对：fc_config_check.sh（冒烟与正式）＋ assert_cfg.py 正向断言（续训、起点、学习率调度、终止 0.3、变体 A、课程第 7、8 段与门槛）；
#   3. 冒烟：fc_run_watched.sh 续训 4096000 步，判：日志里起始步数 349999104、W&B 汇总课程第 8 段与 immediate 1.0、replay ≥ 409600、
#      critic 与 actor 有更新、最终 policy 与起点不同；再 256 世界评一次（仓库里还没评过续训出来的 checkpoint，复核 6）；
#   4. 正式 4 条逐条训练、评测（4096 世界、第 419999744 步，经 scripts/dev/record 落 results/day2/）；任何非零退出都记失败、不重跑（复核 5）。
#   0. （配置里有 EVAL_PRE 时）先逐份评已有 run 的中间 checkpoint（4096 世界），不训练；
#   冒烟只在配置给了 SQ 时跑。
# 日志 runs/logs/<段>_mf4090.log，最后一行 FT_CHAIN_DONE（完成 x、失败 y）或 FT_CHAIN_FAILED <原因>。
# 用法：stages/policy/ft_chain.sh [段，默认 t2ft]   V2D_FT_DRYRUN=1 只做 1、2 两步
set -uo pipefail
cd "$(dirname "$0")/../.."
SEG=${1:-t2ft}; CFG="stages/policy/queues/mf4090_$SEG.ft.sh"
[ -f "$CFG" ] || { echo "ft_chain: 缺 $CFG" >&2; exit 2; }
LOG=runs/logs/${SEG}_mf4090.log; mkdir -p runs/logs
exec > >(tee -a "$LOG") 2>&1
say() { echo "=== $(date '+%F %T %Z') $*"; }
die() { say "FT_CHAIN_FAILED $*"; exit 1; }
TRAIN_SRC="$HOME/v2d/fc_src/v040_p185_nfg"; EVAL_SRC="$HOME/v2d/fc_src/fc_8c778f9"
SQ=""; EVAL_PRE=(); declare -A START_SHA=()
# 配置定义：SQ（可空）、FQ、SSTEP（有冒烟时）、START_SHA[<起点 state 相对 runs/flash_chord 的路径>]=sha256、EVAL_PRE=(<run>:<步数> …)，
#   以及函数 start_of <run>（起点 state 路径）、steps_of <run>（总步数）、p8_of <run>（课程第 8 段 immediate）、ot_of <run>（物体位置终止阈值），可选 extra_of <run>（额外断言）
# shellcheck source=/dev/null
source "$CFG"
check_start() {
  for k in "${!START_SHA[@]}"; do
    [ "$(sha256sum "runs/flash_chord/$k" | cut -d' ' -f1)" = "${START_SHA[$k]}" ] || die "起点 $k 的 sha256 不符"
  done
  say "起点 state 的 sha256 ${#START_SHA[@]} 份都符"
}
qline() { grep -v '^[[:space:]]*#' "$1" | awk -v r="$2" '{for (k = 1; k <= NF; k++) if ($k == r) {print; exit}}'; }
runs_of() { grep -v '^[[:space:]]*#' "$1" | awk '{for (k = 1; k <= NF; k++) if ($k ~ /^(smoke-)?t2-ep[0-9]+-/) {print $k; break}}'; }
# fc_config_check 的打印模式会建出只有 .lock 的空目录（fc_run.sh 第 49、54 行），只有 .lock 以外还有文件才算已有产物（同 arm_var_chain.sh）
has_output() { [ -n "$(find "runs/flash_chord/$1" -mindepth 1 ! -name .lock -print -quit 2>/dev/null)" ]; }
summary() { python3 -c 'import json, sys; d = json.load(open(sys.argv[1])); print(d.get(sys.argv[2]))' "runs/flash_chord/$1/wandb/latest-run/files/wandb-summary.json" "$2"; }

say "$SEG 开始，来历 $(cat .v2d-source-commit 2>/dev/null || echo unknown)"
check_start

# 2. 配置核对
V2D_ALLOW_FC_SRC=1 V2D_FC_SRC="$TRAIN_SRC" stages/policy/fc_config_check.sh $SQ "$FQ" | grep -E '^== |意外|共 |失败|通过|问题'   # 只打每条 run 的汇总与意外差异（复核 NIT 8）
[ "${PIPESTATUS[0]}" = 0 ] || die "fc_config_check.sh 不过"
cmds=""
for r in ${SQ:+$(runs_of "$SQ")} $(runs_of "$FQ"); do
  ot="$(ot_of "$r")"
  A="training.resume=true training.checkpoint_path=/repo/runs/flash_chord/$(start_of "$r")"
  A+=" training.optimization.learning_rate_schedule_updates=683592 training.total_environment_steps=$(steps_of "$r") training.replay.capacity=4000000"
  A+=" termination.object_position_threshold=$ot env.termination.object_position_threshold=$ot"
  p8="$(p8_of "$r")"
  declare -F extra_of > /dev/null && A+=" $(extra_of "$r")"   # 配置可加的额外断言（例如 reset.seed）
  A+=" objective.object_position_std=0.05 env.objective.object_position_std=0.05 termination.object_orientation_threshold=0.6 env.termination.object_orientation_threshold=0.6"
  for c in curriculum training.curriculum; do
    A+=" $c.stages.8.immediate_first_frame_probability=$p8 $c.stages.7.immediate_first_frame_probability=0.25 $c.thresholds.8=349999104 $c.thresholds.7=306249728"
  done
  cmds+="python /repo/stages/policy/assert_cfg.py /repo/runs/cfgcheck/$r.yaml $A || bad=1; "
done
V2D_FC_SRC="$TRAIN_SRC" scripts/dev/in-flash-chord "bad=0; ${cmds}exit \$bad" || die "配方断言不过"
say "配置核对与配方断言全过"
[ "${V2D_FT_DRYRUN:-0}" = 1 ] && { say "FT_CHAIN_DRYRUN_OK"; exit 0; }

# 0. 先评已有 run 的中间 checkpoint（只评不训）
for it in "${EVAL_PRE[@]}"; do
  er="${it%%:*}"; es="${it#*:}"
  [ -f "runs/flash_chord/$er/policy_$es.safetensors" ] || { say "缺 runs/flash_chord/$er/policy_$es.safetensors，跳过"; continue; }
  if [ -f "runs/flash_chord/$er/eval_${es}_w4096.json" ]; then say "$er 第 $es 步已评过，跳过"; continue; fi
  V2D_FC_SRC="$EVAL_SRC" scripts/dev/record "day2/eval_${er}_s${es}" stages/policy/fc_eval.sh "$er" "$es" || say "$er 第 $es 步评测失败"
done
[ "${#EVAL_PRE[@]}" = 0 ] || say "中间 checkpoint 评完（${#EVAL_PRE[@]} 份）"

# 3. 冒烟（配置给了 SQ 才跑）
if [ -n "$SQ" ]; then
SM="$(runs_of "$SQ")"
# 冒烟已训完（有 DONE）就不再训，只重做下面的检查（10/26 14:05 第一次的检查把整数 1 当成不等于 "1.0"）；有产物没 DONE 就停
if [ -f "runs/flash_chord/$SM/DONE" ]; then say "冒烟 $SM 已 DONE：不重训，只做检查"
else
  has_output "$SM" && die "runs/flash_chord/$SM 里已有产物但没有 DONE（不重跑，复核 5）"
  read -r -a tok <<< "$(qline "$SQ" "$SM")"; envs=(); i=0
  while [[ ${tok[$i]} =~ ^[A-Z_][A-Z0-9_]*= ]]; do envs+=("${tok[$i]}"); i=$((i + 1)); done
  env "${envs[@]}" V2D_ALLOW_FC_SRC=1 V2D_FC_SRC="$TRAIN_SRC" stages/policy/fc_run_watched.sh "${tok[@]:$i}" || die "冒烟 $SM 训练退出码非零"
fi
grep -m1 -E '^training progress: ' "runs/logs/$SM.s1.log" | grep -q '^training progress: 349,999,104/354,095,104' || die "冒烟的起始步数不是 349999104：$(grep -m1 '^training progress: ' "runs/logs/$SM.s1.log")"
num_is() { python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) == float(sys.argv[2]) else 1)' "$1" "$2" 2>/dev/null; }   # W&B 把 1.0 存成整数 1
num_is "$(summary "$SM" curriculum/stage_index)" 8 || die "冒烟课程段不是 8"
num_is "$(summary "$SM" curriculum/immediate_first_frame_probability)" 1.0 || die "冒烟 immediate_first_frame_probability 不是 1.0：$(summary "$SM" curriculum/immediate_first_frame_probability)"
python3 - "runs/flash_chord/$SM/wandb/latest-run/files/wandb-summary.json" <<'EOF' || die "冒烟 replay 没填满或没有更新"
import json, math, sys
d = json.load(open(sys.argv[1]))
ok = d.get("replay/size_transitions", 0) >= 409600 and all(
    isinstance(d.get(k), (int, float)) and math.isfinite(d[k]) and d[k] != 0
    for k in ("sac/critic_loss_mean_per_update", "sac/actor_loss_mean_per_update"))
print("冒烟汇总：replay", d.get("replay/size_transitions"), "critic", d.get("sac/critic_loss_mean_per_update"), "actor", d.get("sac/actor_loss_mean_per_update"))
sys.exit(0 if ok else 1)
EOF
a="$(sha256sum "runs/flash_chord/$SM/policy_$SSTEP.safetensors" | cut -d' ' -f1)"
b="$(sha256sum "runs/flash_chord/$SMOKE_START_POLICY" | cut -d' ' -f1)"
[ -n "$a" ] && [ "$a" != "$b" ] || die "冒烟最终 policy 与起点相同或缺失"
V2D_FC_SRC="$EVAL_SRC" V2D_EVAL_WORLDS=256 scripts/dev/record "t2ft/eval_${SM}_s${SSTEP}_w256" stages/policy/fc_eval.sh "$SM" "$SSTEP" || die "冒烟 256 世界评测失败"
say "冒烟全过"
fi

# 4. 正式
nd=0; nf=0
for r in $(runs_of "$FQ"); do
  if has_output "$r"; then say "$r 目录里已有产物：记失败、不重跑"; nf=$((nf + 1)); continue; fi
  read -r -a tok <<< "$(qline "$FQ" "$r")"; envs=(); i=0
  while [[ ${tok[$i]} =~ ^[A-Z_][A-Z0-9_]*= ]]; do envs+=("${tok[$i]}"); i=$((i + 1)); done
  say "训练 $r"
  env "${envs[@]}" V2D_ALLOW_FC_SRC=1 V2D_FC_SRC="$TRAIN_SRC" stages/policy/fc_run_watched.sh "${tok[@]:$i}"; rc=$?
  if [ "$rc" != 0 ] || [ ! -f "runs/flash_chord/$r/DONE" ]; then say "$r 训练失败（退出码 $rc）"; nf=$((nf + 1)); continue; fi
  STEP="$(steps_of "$r")"
  say "$r M：起始 $(grep -m1 '^training progress: ' "runs/logs/$r.s1.log" | cut -d' ' -f3)，课程段 $(summary "$r" curriculum/stage_index)，immediate $(summary "$r" curriculum/immediate_first_frame_probability)，回合平均 $(summary "$r" episode/length_mean_per_completed_episode)"
  V2D_FC_SRC="$EVAL_SRC" scripts/dev/record "day2/eval_${r}_s${STEP}" stages/policy/fc_eval.sh "$r" "$STEP" || { say "$r 评测失败"; nf=$((nf + 1)); continue; }
  nd=$((nd + 1))
done
check_start
say "FT_CHAIN_DONE（完成 ${nd}、失败 ${nf}）"
