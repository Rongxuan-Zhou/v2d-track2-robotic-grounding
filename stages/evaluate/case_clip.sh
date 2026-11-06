#!/usr/bin/env bash
# SOP“成功与失败案例”视频：录一次 rollout、挑世界，再渲染成左右两栏的成片（训练机上、仓库根执行）。计划 agent-doc/plan-site-cases-video-2026-10-08.md 第 2 版。
# 用法：V2D_FC_SRC=<8c778f9 副本> stages/evaluate/case_clip.sh record <run> <类别,类别…>   （256 世界，V2D_CASE_WORLDS 可改）
#       V2D_FC_SRC=<8c778f9 副本> stages/evaluate/case_clip.sh render <run> <类别,类别…>   （同一条 run 的参考栏只渲一次）
# 输出 runs/site_cases/<run>/：record.npz、record.json（评测指标）、panels.mp4（各左栏与参考栏顺序录在一段里）、render.json（帧号范围）；
#   成片 runs/site_cases/out/<run>_<类别>.{mp4,jpg}（左：录下的 rollout；右：参考）。日志 runs/logs/<run>.case_<模式>.log。
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"; cd "$ROOT"
die() { echo "case_clip: $*" >&2; exit "${2:-2}"; }
MODE=${1:?record|render}; RUN=${2:?run}; KINDS=${3:?类别}; STEP=349999104; W=${V2D_CASE_WORLDS:-256}
[[ $MODE =~ ^(record|render)$ && $RUN =~ ^t[12]-[a-z0-9_-]{2,56}$ && $KINDS =~ ^[a-z,]+$ ]] || die "参数非法"
[ -n "${V2D_FC_SRC:-}" ] && head -1 "$V2D_FC_SRC/SOURCE_V2D.txt" 2>/dev/null | grep -q 8c778f905c || die "V2D_FC_SRC 必须是 8c778f9 副本"
O="runs/flash_chord/$RUN"; P="$O/policy_${STEP}.safetensors"; [ -f "$P" ] || die "没有 $P"
[ -f "$O/NONFINITE" ] && die "$RUN 有 NONFINITE" 12
free_mb="$(df -m . | awk 'NR==2{print $4}')"; [ "${free_mb:-0}" -ge 800 ] || die "磁盘只剩 ${free_mb} MB（< 800）" 7
D="runs/site_cases/$RUN"; mkdir -p "$D" runs/site_cases/out runs/logs
others="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')"
[ -z "$others" ] || die "GPU 上有别的进程（PID ${others//$'\n'/,}）" 6
PARQ="$(python3 -c 'import json,sys
for l in open(sys.argv[1]):
    r=json.loads(l)
    if r.get("event")=="start": print(r["parquet"]); break' "$O/run_record.jsonl")"
[[ $PARQ == /repo/runs/motion/tier* ]] || die "读不出参考 parquet：$PARQ"
LOG="runs/logs/$RUN.case_$MODE.log"
echo "=== $(date '+%F %T %Z') case_clip $MODE $RUN $KINDS ===" >> "$LOG"
if [ "$MODE" = record ]; then
  V2D_GPU=1 V2D_CONTAINER_NAME="fc-case-$RUN" scripts/dev/in-flash-chord \
    "cd /workspace && V2D_CASE_RUN=$RUN V2D_CASE_OUT=/repo/$D/record.npz V2D_CASE_KINDS=$KINDS python /repo/stages/evaluate/case_clip.py record $(printf '%q ' \
     "evaluation.checkpoint=/repo/$P" "evaluation.world_count=$W" "evaluation.metrics_output=/repo/$D/record.json" "hydra.run.dir=/repo/$O/hydra/case-record")" >> "$LOG" 2>&1
  rc=$?; [ "$rc" = 0 ] && [ -s "$D/record.npz" ] || die "录制失败（退出码 $rc），见 $LOG" 3
  grep -a "^case_clip:" "$LOG" | tail -6
  exit 0
fi
V2D_GPU=1 V2D_CONTAINER_NAME="fc-case-$RUN-render" scripts/dev/in-flash-chord \
  "cd /workspace && PYGLET_HEADLESS=1 V2D_CASE_RUN=$RUN V2D_CASE_IN=/repo/$D/record.npz V2D_CASE_PICKS=$KINDS V2D_CASE_DIR=/repo/$D V2D_CASE_PARQUET=$PARQ \
   python /repo/stages/evaluate/case_clip.py render $(printf '%q ' "evaluation.checkpoint=/repo/$P" "evaluation.world_count=1" \
   "evaluation.metrics_output=/repo/$D/render_unused.json" "hydra.run.dir=/repo/$O/hydra/case-render")" >> "$LOG" 2>&1
rc=$?; [ "$rc" = 0 ] && [ -s "$D/panels.mp4" ] && [ -s "$D/render.json" ] || die "渲染失败（退出码 $rc），见 $LOG" 3
for K in ${KINDS//,/ }; do
  OUT="runs/site_cases/out/${RUN}_${K}"
  V2D_CONTAINER_NAME="fc-case-ff-$RUN-$K" scripts/dev/in-flash-chord \
    "cd /repo && python stages/evaluate/case_compose.py $D/record.npz $K $D/panels.mp4 $D/render.json $OUT" >> "$LOG" 2>&1 || die "ffmpeg 失败，见 $LOG" 3
  [ -s "$OUT.mp4" ] && [ -s "$OUT.jpg" ] || die "没有成片或封面 $OUT" 3
  echo "case_clip: $RUN $K → $OUT.mp4（$(stat -c %s "$OUT.mp4") B）"
done
grep -a "^case_clip: 渲染完成" "$LOG" | tail -1
