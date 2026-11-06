#!/usr/bin/env bash
# 回放速度对照的分析链（计划 agent-doc/plan-playback-2026-09-28.md 第 2 版）：在一台训练机上跑完同一 buffer 两臂最后三个 checkpoint 的逐帧诊断。
# 100 万一对在 mf4090、满配一对在 ml40：checkpoint 都在本机，不经 Mac 中转（Mac → mf4090 的上传实测会卡住）。判定在 Mac 上用 playback_compare.py。
#   0. （可选）V2D_WAIT_FOR_LOG=<训练链日志>：每分钟看一次它的最后一行，PLAYBACK_TRAIN_DONE 就往下走，PLAYBACK_FAILED 就停
#   1. 两条 run 都已 DONE、六个 checkpoint 都在
#   2. 只在 mf4090（1m）：Day 1b 已有的两条对照最终 rollout 用新版报告重算，与旧报告逐行比对（diag_report_diff.py）；
#      100 万对照的重算报告就当作它第 349,999,104 步的诊断，不重录 rollout
#   3. 逐帧回放一臂：diag_day1b.sh stages/evaluate/playback_<buffer>_frame.runs（每个 checkpoint 一份报告，阶段 209 269 314 420）
#   4. 真实时长一臂：diag_day1b.sh stages/evaluate/playback_<buffer>_real.runs（阶段 348 448 523 700）
#   只有 G1 或 G3 没过时，登记的规则允许把那个 checkpoint 重跑一次；是否重跑由人定，这里不自动重跑。
# 用法（仓库根，放 tmux）：V2D_WAIT_FOR_LOG=runs/logs/mf4090_playback.log stages/evaluate/playback_analysis.sh 1m
# 日志 runs/logs/playback_analysis_<buffer>.log，最后一行 ANALYSIS_DONE、ANALYSIS_GATES_FAILED 或 ANALYSIS_FAILED。
set -Eeuo pipefail
cd "$(dirname "$0")/../.."
BUF=${1:?用法：playback_analysis.sh <1m|10m>}
case "$BUF" in
  1m) CTRL=t1-ep20-w2048-cap1m-mf4090; RT=t1-ep20-w2048-cap1m-rt-mf4090 ;;
  10m) CTRL=t1-ep20-w2048-cap10m-ml40; RT=t1-ep20-w2048-cap10m-rt-ml40 ;;
  *) echo "buffer 只能是 1m 或 10m" >&2; exit 2 ;;
esac
mkdir -p runs/logs results/playback
exec >> "runs/logs/playback_analysis_$BUF.log" 2>&1
say() { echo "=== $(date '+%F %T %Z') $*"; }
fail() { say "$*"; say ANALYSIS_FAILED; exit 1; }
trap 'say "ANALYSIS_FAILED（第 $LINENO 行命令失败）"' ERR
say "start $BUF, source $(cat .v2d-source-commit 2>/dev/null || echo unknown)"

# 0. 等训练链结束
if [ -n "${V2D_WAIT_FOR_LOG:-}" ]; then
  say "等 $V2D_WAIT_FOR_LOG 的最后一行变成 PLAYBACK_TRAIN_DONE"
  while :; do
    last="$(tail -1 "$V2D_WAIT_FOR_LOG" 2>/dev/null || true)"
    case "$last" in
      *PLAYBACK_TRAIN_DONE*) say "训练链已结束"; break ;;
      *PLAYBACK_FAILED*) fail "训练链失败（$last），不做分析" ;;
    esac
    sleep 60
  done
fi

# 1. 两条 run 与六个 checkpoint
STEPS3="330012672 340013056 349999104"
for r in "$CTRL" "$RT"; do
  [ -f "runs/flash_chord/$r/DONE" ] || fail "runs/flash_chord/$r 没有 DONE"
  for s in $STEPS3; do [ -f "runs/flash_chord/$r/policy_$s.safetensors" ] || fail "缺 runs/flash_chord/$r/policy_$s.safetensors"; done
done

# 2. mf4090：Day 1b 的旧 rollout 用新版报告重算，并与旧报告逐行比对
if [ "$BUF" = 1m ]; then
  for r in t1-ep20-w2048-cap1m-mf4090 t1-ep20-w2048-cap10m-ml40; do
    D=runs/diag/$r; O=runs/flash_chord/$r
    [ -f "$D/rollout_w4096.npz" ] || fail "缺 Day 1b 的 $D/rollout_w4096.npz"
    CROSS=""; [ -f "$O/eval_349999104_w4096.ml40.json" ] && CROSS="--cross-card-json /repo/$O/eval_349999104_w4096.ml40.json"
    if [ "$r" = "$CTRL" ]; then NAME=diag_${r}_s349999104; else NAME=rereport_${r}_day1b_rollout; fi
    if [ ! -f "results/playback/$NAME.txt" ]; then
      scripts/dev/record "playback/$NAME" scripts/dev/in-flash-chord \
        "python /repo/stages/evaluate/diag_report.py /repo/$D/rollout_w4096.npz --diag-json /repo/$D/rollout_w4096.json --same-card-json /repo/$O/eval_349999104_w4096.json $CROSS --segments 209 420 --expect-ref 209 420 269 --predict-frame 359 --phases 209 269 314 420 --svg /repo/results/playback/$NAME.svg --title '$r 第 349999104 步（4096 世界，Day 1b 的 rollout）'" \
        || fail "$r：用新版报告重算 Day 1b 的 rollout 没过"
    fi
    scripts/dev/record "playback/reportdiff_$r" python3 stages/evaluate/diag_report_diff.py "results/day1/diag_$r.txt" "results/playback/$NAME.txt" \
      || fail "$r：新旧报告有意料之外的差异"
  done
fi

# 3–4. 两臂各一张运行表
gates=0
for pb in frame real; do
  if [ "$pb" = frame ]; then PH="209 269 314 420"; else PH="348 448 523 700"; fi
  if V2D_DIAG_PER_STEP=1 V2D_DIAG_RECORD_DIR=playback V2D_DIAG_PHASES="$PH" stages/evaluate/diag_day1b.sh "stages/evaluate/playback_${BUF}_$pb.runs"; then
    say "$BUF $pb 一臂：三个 checkpoint 的诊断闸全过"
  else
    rc=$?
    [ "$rc" = 3 ] || fail "$BUF $pb 一臂的诊断出错（退出码 $rc）"
    gates=$((gates + 1)); say "$BUF $pb 一臂：有 checkpoint 的闸没过（见上）"
  fi
done
if [ "$gates" = 0 ]; then say ANALYSIS_DONE; else say "ANALYSIS_GATES_FAILED（$gates 臂有闸没过）"; exit 3; fi
