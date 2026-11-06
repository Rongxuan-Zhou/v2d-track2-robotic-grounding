#!/usr/bin/env bash
# Day 2 评测链（计划 agent-doc/plan-day2-2026-09-28.md 第 2 版）：等训练链结束，再评期望表里每条有 DONE 的 run；单份评测失败不挡其余。训练机上执行，放 tmux。
#   0. V2D_WAIT_FOR_LOG=<训练链日志>：每分钟看一次最后一行，DAY2_TRAIN_DONE 或 DAY2_TRAIN_FAILED 之后开始；来历与 GPU 同训练链
#   1. 附加表里的 identity 行：把该 checkpoint 用 eval_playback_override.py --playback same 评一次，与已有的 fc_eval JSON 比
#      （|ΔADD-AUC| < 0.005、chord_sr |z| < 3、替换函数调用 1 次）；不过就不做任何交叉评测（匹配回放的评测照做）
#   2. 期望表每条 DONE 的 run：最后三个 checkpoint 按训练时的回放各评 4096 世界（fc_eval.sh，经 record 落成 day2/eval_<run>_s<步数>），
#      核对 step_count；xeval=yes 的另把最终 checkpoint 按另一种回放评一次（day2/xeval_<run>_s<步数>_<frame|real>）。
#      V2D_EVAL_STEPS="349999104" 时只评列出的步数（Tier 2 批量只评最终 checkpoint，计划 agent-doc/plan-t2-batch-2026-10-01.md）；
#      run 目录里有 NONFINITE（fc_run_watched.sh 写的：训完但最新 policy 含非有限参数）的不评
#   3. 附加表里的 eval / xeval 行（例如 ep20 两条满配的交叉评测、B 的对照）
# 期望表同 stages/policy/day2_train_chain.sh。附加表（TSV，无表头，# 开头是注释）：
#   identity <run> <step>
#   eval <run> <step> <playback> <src_frames>
#   xeval <run> <step> <另一种回放 frame|real> <src_frames> <pos_tol> <ori_tol>
# 用法（仓库根）：V2D_WAIT_FOR_LOG=runs/logs/<训练链>.log stages/evaluate/day2_eval_chain.sh <期望表|-> <附加表|-> <日志名>
#   期望表给 - 时只做附加表（开跑前的预检：恒等覆盖核对、ep20 的交叉评测、B 的对照）
# 日志 runs/logs/<日志名>.log，最后一行 DAY2_EVAL_DONE（成功 x、失败 y）或 DAY2_EVAL_FAILED（链本身出错）。
# V2D_GPU_WAIT_S（默认 0，即不等）：开始时与每份评测前见到 GPU 上有别的进程，每 V2D_POLL_S（默认 60）秒看一次，最多等这么久，
#   空了接着评，等不到就停（DAY2_EVAL_FAILED）。同 export_chain.sh 的 wait_gpu（计划 agent-doc/plan-t2-scaleout-2026-10-09.md，审查 M3）。
#   测试：stages/policy/test_day2_chain_waits.py。
set -uo pipefail
cd "$(dirname "$0")/../.."
EXPECT=${1:?期望表}; EXTRA=${2:?附加表或 -}; NAME=${3:?日志名}
mkdir -p runs/logs results/day2
exec >> "runs/logs/$NAME.log" 2>&1
MARK="runs/.chain-active-$NAME"   # 等待期间也挡住同步：scripts/dev/remote 见到 runs/.chain-active-* 就拒绝
echo "$NAME waiting-or-running $(date '+%F %T %Z')" > "$MARK"; trap 'rm -f "$MARK"' EXIT
say() { echo "=== $(date '+%F %T %Z') $*"; }
die() { say "$*"; say DAY2_EVAL_FAILED; exit 1; }
exp_frames() { python3 -c 'import sys; n = int(sys.argv[2]); print(n if sys.argv[1] == "frame" else int(round((n - 1) / 30.0 * 50.0)) + 1)' "$1" "$2"; }
jget() { python3 -c 'import json, sys; print(json.load(open(sys.argv[1])).get(sys.argv[2]))' "$1" "$2"; }
EVAL_STEPS="${V2D_EVAL_STEPS:-330012672 340013056 349999104}"
[[ $EVAL_STEPS =~ ^[0-9]+( [0-9]+)*$ ]] || { echo "V2D_EVAL_STEPS 只能是空格分隔的步数：$EVAL_STEPS" >&2; exit 2; }

if [ -n "${V2D_WAIT_FOR_LOG:-}" ]; then
  say "等 $V2D_WAIT_FOR_LOG 的最后一行变成 DAY2_TRAIN_DONE / DAY2_TRAIN_FAILED"
  until tail -1 "$V2D_WAIT_FOR_LOG" 2>/dev/null | grep -qE "DAY2_TRAIN_(DONE|FAILED)"; do sleep 60; done
  say "训练链已结束：$(tail -1 "$V2D_WAIT_FOR_LOG")"
fi
SRC0="$(cat .v2d-source-commit 2>/dev/null || echo unknown)"
say "start $NAME, source $SRC0, 期望表 $EXPECT, 附加表 $EXTRA, 评测步数 $EVAL_STEPS"
case "$SRC0" in *-dirty*|*-syncing*|unknown) die "来历 $SRC0 不干净，先提交再同步" ;; esac
check_src() { [ "$(cat .v2d-source-commit 2>/dev/null || echo unknown)" = "$SRC0" ] || die "来历变了（开头 $SRC0，现在 $(cat .v2d-source-commit 2>/dev/null)），停下"; }
GPU_WAIT_S="${V2D_GPU_WAIT_S:-0}"
[[ $GPU_WAIT_S =~ ^[0-9]+$ ]] || die "V2D_GPU_WAIT_S 只能是秒数：$GPU_WAIT_S"
gpu_pids() { nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ' | paste -sd, -; }
gpu_check() {  # 同 day2_train_chain.sh：最多等 GPU_WAIT_S 秒（0 就不等），等不到就停
  local o t0=$SECONDS
  o="$(gpu_pids)"; [ -n "$o" ] || return 0
  [ "$GPU_WAIT_S" -gt 0 ] || die "GPU 上有别的进程（$o），停下"
  say "GPU 上有别的进程（${o}），每 ${V2D_POLL_S:-60} 秒看一次，最多等 ${GPU_WAIT_S} 秒"
  while [ -n "$(gpu_pids)" ]; do
    [ $((SECONDS - t0)) -lt "$GPU_WAIT_S" ] || die "GPU 被别的进程占了 ${GPU_WAIT_S} 秒，停下"
    sleep "${V2D_POLL_S:-60}"
  done
  say "GPU 空了，接着评"
}
gpu_check
ok=0; bad=0

matched() {  # matched <run> <step> <playback> <src_frames>
  local run=$1 step=$2 pb=$3 src=$4 want got
  check_src; gpu_check
  want="$(exp_frames "$pb" "$src")"
  if scripts/dev/record "day2/eval_${run}_s${step}" stages/policy/fc_eval.sh "$run" "$step"; then
    got="$(jget "runs/flash_chord/$run/eval_${step}_w4096.json" step_count)"
    if [ "$got" = "$want" ]; then ok=$((ok + 1)); else say "$run 第 $step 步：step_count $got，期望 $want"; bad=$((bad + 1)); fi
  else
    say "$run 第 $step 步的评测失败"; bad=$((bad + 1))
  fi
}
cross() {  # cross <run> <step> <另一种回放> <src_frames> <pos_tol> <ori_tol>
  local run=$1 step=$2 pb=$3 src=$4 pos=$5 ori=$6 want side
  check_src; gpu_check
  [ "${XOK:-0}" = 1 ] || { say "恒等覆盖核对没过或没做：跳过 $run 的交叉评测"; bad=$((bad + 1)); return; }
  want="$(exp_frames "$pb" "$src")"
  side="runs/flash_chord/$run/xeval_${step}_w4096_${pb}.json.override.json"
  if scripts/dev/record "day2/xeval_${run}_s${step}_${pb}" env V2D_GPU=1 V2D_CONTAINER_NAME="xeval-$run" scripts/dev/in-flash-chord \
      "python /repo/stages/evaluate/eval_playback_override.py /repo/runs/flash_chord/$run/policy_${step}.safetensors /repo/runs/flash_chord/$run/xeval_${step}_w4096_${pb}.json --playback $pb --palm-position-tol $pos --palm-orientation-tol $ori" \
     && [ "$(jget "$side" step_count)" = "$want" ] && [ "$(jget "$side" calls)" = 1 ]; then
    ok=$((ok + 1))
  else
    say "$run 第 $step 步按 $pb 的交叉评测失败或 step_count 不是 $want"; bad=$((bad + 1))
  fi
}

# 1. 恒等覆盖核对
XOK=0
if [ "$EXTRA" != - ]; then
  while IFS=$'\t' read -r kind run step _; do
    [ "$kind" = identity ] || continue
    check_src
    J=runs/flash_chord/$run/identity_${step}_w4096.json
    if scripts/dev/record "day2/identity_${run}_s${step}_${NAME}" env V2D_GPU=1 V2D_CONTAINER_NAME="identity-$run" scripts/dev/in-flash-chord \
        "python /repo/stages/evaluate/eval_playback_override.py /repo/runs/flash_chord/$run/policy_${step}.safetensors /repo/$J --playback same" \
       && [ "$(jget "$J.override.json" calls)" = 1 ] && [ "$(jget "$J.override.json" changes)" = "{}" ] \
       && scripts/dev/record "day2/identity_compare_${run}_s${step}_${NAME}" python3 stages/policy/eval_compare.py "$J" "runs/flash_chord/$run/eval_${step}_w4096.json" \
            --max-add-auc-diff 0.005 --max-chord-z 3; then
      XOK=1; say "恒等覆盖核对通过（$run 第 $step 步）：交叉评测可用"
    else
      say "恒等覆盖核对没过（$run 第 $step 步）：本链不做交叉评测"
    fi
  done < <(grep -v '^[[:space:]]*#' "$EXTRA" | grep -v '^[[:space:]]*$')
fi

# 2. 期望表
[ "$EXPECT" = - ] || while IFS=$'\t' read -r run smoke seq pb src support pos ori cap seed xev; do
  if [ ! -f "runs/flash_chord/$run/DONE" ]; then say "$run 没有 DONE：不评"; continue; fi
  if [ -f "runs/flash_chord/$run/NONFINITE" ]; then say "$run 的最新 policy 含非有限参数（NONFINITE）：不评"; continue; fi
  for step in $EVAL_STEPS; do matched "$run" "$step" "$pb" "$src"; done
  if [ "$xev" = yes ]; then
    other=real; [ "$pb" = real ] && other=frame
    cross "$run" 349999104 "$other" "$src" "$pos" "$ori"
  fi
done < <(tail -n +2 "$EXPECT" | grep -v '^[[:space:]]*$')

# 3. 附加表
if [ "$EXTRA" != - ]; then
  while IFS=$'\t' read -r kind run step pb src pos ori; do
    case "$kind" in
      eval) matched "$run" "$step" "$pb" "$src" ;;
      xeval) cross "$run" "$step" "$pb" "$src" "$pos" "$ori" ;;
    esac
  done < <(grep -v '^[[:space:]]*#' "$EXTRA" | grep -v '^[[:space:]]*$')
fi
say "DAY2_EVAL_DONE（成功 $ok、失败 $bad）"
