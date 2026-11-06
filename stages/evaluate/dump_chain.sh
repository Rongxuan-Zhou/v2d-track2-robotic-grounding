#!/usr/bin/env bash
# 逐帧终止分析的批量评测（训练机上、仓库根执行，放 tmux）：按清单逐条跑 stages/evaluate/fc_eval_dump.sh（8c778f9 评测器、4096 世界，
# 另存逐步、逐世界的跟踪误差）。不训练、不导出、不上传。计划 agent-doc/plan-frame-onset-2026-10-14.md 第 3 节第 0 步；骨架同 export_chain.sh。
# 用法：V2D_FC_SRC=<8c778f9 副本> stages/evaluate/dump_chain.sh <清单> <日志名>
#       预检：V2D_DUMP_CHAIN_DRYRUN=1 加上同样的环境与参数（不写日志、结果表与标记）。另查：别的长链标记、残留的 fc-evaldump-* 容器（有就 FAIL）
# 清单每行空白分隔：run 步数；# 开头与空行跳过。
# 0. 日志 runs/logs/<日志名>.log 或结果表 runs/frames/<日志名>.tsv 已存在、清单读不了、没有行、有一行格式不对或 run 重复：拒绝（退出码 2，不写日志）。
# 1. 起链就写长链标记 runs/.chain-active-<日志名>（写日志名、来历与时间：scripts/dev/remote 只认非空标记），退出时删。先核 V2D_FC_SRC（没设或
#    核不过就停），再等 GPU：有进程就每 V2D_POLL_S 秒（默认 60）看一次，最多等 V2D_GPU_WAIT_S 秒（默认 7200），等不到就停；只等，不碰别人的进程。
#    本日志里不出现链结束标记的原文。
# 2. 逐行（某行不过不停）：eval_dump 下 JSON 与 npz 都在 → exists（不重评）；有 NONFINITE → nonfinite；没有 policy_<步数>.safetensors → no_policy；
#    否则跑 fc_eval_dump.sh：退出码 0 且 JSON、npz 都在 → ok；退出码 6（GPU 刚被别人占了）→ 等 GPU 空了重跑这一行（同样最多等
#    V2D_GPU_WAIT_S，等不到就停整条链）；其他 → failed（结果表记退出码）。每次调用经 scripts/dev/record 落盘到 results/frames/<日志名>/<run>_s<步数>.txt。
#    结果表 runs/frames/<日志名>.tsv：run 步数 状态 退出码 JSON npz npz 的 sha256。
#    nvidia-smi 出错按 GPU 被占处理；调用 fc_eval_dump.sh 时固定 V2D_EVAL_WORLDS=4096；退出码 6 的重跑之间停 V2D_POLL_S 秒，一行累计等满 V2D_GPU_WAIT_S 就停。
# 3. 日志最后一行 DUMP_CHAIN_DONE（成功 x、失败 y）（成功 = ok + exists），停在中途时是 DUMP_CHAIN_FAILED。
set -uo pipefail
cd "$(dirname "$0")/../.."
LIST=${1:?用法见文件头}; NAME=${2:?用法见文件头}
[[ $NAME =~ ^[A-Za-z0-9_][A-Za-z0-9_.-]*$ && $NAME != *..* ]] || { echo "dump_chain: 日志名非法：$NAME" >&2; exit 2; }
LOG="runs/logs/$NAME.log"; OUT="runs/frames/$NAME.tsv"; MARK="runs/.chain-active-$NAME"
DRY="${V2D_DUMP_CHAIN_DRYRUN:-0}"; GPU_WAIT_S="${V2D_GPU_WAIT_S:-7200}"; POLL_S="${V2D_POLL_S:-60}"; W=4096
[ -e "$LOG" ] || [ -e "$OUT" ] && { echo "dump_chain: $LOG 或 $OUT 已存在，拒绝（确认后改名或删掉再起）" >&2; exit 2; }
[ -r "$LIST" ] || { echo "dump_chain: 读不了清单 $LIST" >&2; exit 2; }
[[ $GPU_WAIT_S =~ ^[0-9]+$ && $POLL_S =~ ^[1-9][0-9]*$ ]] || { echo "dump_chain: V2D_GPU_WAIT_S、V2D_POLL_S 必须是整数" >&2; exit 2; }
ROWS=()
while read -r run step extra || [ -n "${run:-}" ]; do   # 最后一行没有换行也读
  case "$run" in ''|'#'*) continue ;; esac
  [[ $run =~ ^[a-z0-9][a-z0-9_-]{2,59}$ && $step =~ ^[0-9]+$ && -z ${extra:-} ]] || { echo "dump_chain: 清单行格式不对：$run $step ${extra:-}" >&2; exit 2; }
  for r in "${ROWS[@]+"${ROWS[@]}"}"; do [ "${r% *}" = "$run" ] && { echo "dump_chain: 清单里 $run 重复" >&2; exit 2; }; done
  ROWS+=("$run $step")
done < "$LIST"
NROW=${#ROWS[@]}
[ "$NROW" -gt 0 ] || { echo "dump_chain: 清单 $LIST 没有行" >&2; exit 2; }
demark() { sed -E 's/DUMP_CHAIN_(DONE|FAILED)/批量评测链结束/g; s/DUMP_DRYRUN_(OK|FAIL)/批量评测预检结束/g'; }
dumped() { [ -s "runs/flash_chord/$1/eval_dump/eval_${2}_w${W}.json" ] && [ -s "runs/flash_chord/$1/eval_dump/eval_${2}_w${W}.npz" ]; }

if [ "$DRY" = 1 ]; then
  bad=0
  chk() { if eval "$2" >/dev/null 2>&1; then echo "  ok   $1"; else echo "  FAIL $1"; bad=$((bad + 1)); fi; }
  echo "== 预检 ${NAME}：清单 ${LIST}（${NROW} 行），V2D_FC_SRC=${V2D_FC_SRC:-未设}，来历 $(cat .v2d-source-commit 2>/dev/null || echo unknown)"
  echo "  ok   $LOG 与 $OUT 都不存在"
  chk "nvidia-smi 能跑" "nvidia-smi --query-compute-apps=pid --format=csv,noheader"
  chk "V2D_FC_SRC 是核对过的 8c778f9 副本" "[ -n '${V2D_FC_SRC:-}' ] && python3 stages/policy/fc_src_check.py '${V2D_FC_SRC:-/nonexistent}' third_party/patches/flash_chord-8c778f9.sha256"
  others="$(ls runs/.chain-active-* 2>/dev/null | paste -sd, -)"
  chk "没有别的长链标记（${others:-无}）" "[ -z '$others' ]"
  left="$(docker ps -a --filter name=fc-evaldump- --format '{{.Names}}' 2>&1 | paste -sd, -)"
  chk "没有残留的 fc-evaldump-* 容器（${left:-无}）" "[ -z '$left' ]"
  need=$((NROW * 60 + 1024)); free=$(df -Pm runs | awk 'NR == 2 {print $4}')
  chk "runs 所在分区可用 ${free} MB，不少于 ${need} MB（行数 × 60 MB + 1 GB）" "[ '${free:-0}' -ge '$need' ]"
  for row in "${ROWS[@]}"; do
    run=${row% *}; step=${row#* }; d="runs/flash_chord/$run"; e="$d/eval_${step}_w${W}.json"
    if dumped "$run" "$step"; then echo "  ok   ${run}：已有 dump，开跑时记 exists"; continue; fi
    chk "${run}：有 policy_${step}.safetensors" "[ -s '$d/policy_${step}.safetensors' ]"
    chk "${run}：没有 NONFINITE" "[ ! -e '$d/NONFINITE' ]"
    chk "${run}：原评测 ${e} 是 4096 世界、没有非有限世界" \
      "python3 -c 'import json, sys; e = json.load(open(sys.argv[1])); sys.exit(0 if e.get(\"world_count\") == 4096 and e.get(\"non_finite_worlds\") == [] else 1)' '$e'"
  done
  if [ "$bad" = 0 ]; then echo "DUMP_DRYRUN_OK"; exit 0; else echo "DUMP_DRYRUN_FAIL $bad"; exit 1; fi
fi

mkdir -p runs/logs runs/frames
echo "$NAME $(cat .v2d-source-commit 2>/dev/null || echo unknown) $(date '+%F %T %Z')" > "$MARK"; trap 'rm -f "$MARK"' EXIT
say() { echo "=== $(date '+%F %T %Z') $*" | demark | tee -a "$LOG"; }
stop() { say "$*"; echo "=== $(date '+%F %T %Z') DUMP_CHAIN_FAILED" | tee -a "$LOG"; exit 1; }
gpu_pids() {  # nvidia-smi 出错按“GPU 被占”处理（审查第 3 条：不能把出错读成卡空）
  local out
  out="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" || { echo "nvidia-smi 出错"; return; }
  printf '%s\n' "$out" | tr -d ' ' | grep . | paste -sd, -
}
wait_gpu() {  # GPU 上有进程就每 POLL_S 秒看一次，最多 GPU_WAIT_S 秒；空了返回 0。只看，不动别人的进程
  local t0=$SECONDS p
  p="$(gpu_pids)"; [ -n "$p" ] || return 0
  say "GPU 上有别的进程（${p}），每 ${POLL_S} 秒看一次，最多等 ${GPU_WAIT_S} 秒"
  while [ -n "$(gpu_pids)" ]; do
    [ $((SECONDS - t0)) -lt "$GPU_WAIT_S" ] || return 1
    sleep "$POLL_S"
  done
  say "GPU 空了，接着做"
}
printf 'run\tstep\tstatus\trc\tjson\tnpz\tnpz_sha256\n' > "$OUT"
say "start ${NAME}：清单 ${LIST}（${NROW} 行），V2D_FC_SRC=${V2D_FC_SRC:-未设}，GPU 最多等 ${GPU_WAIT_S} 秒，来历 $(cat .v2d-source-commit 2>/dev/null || echo unknown)"
[ -n "${V2D_FC_SRC:-}" ] || stop "没设 V2D_FC_SRC，停下"
scripts/dev/record "frames/${NAME}/fc_src" python3 stages/policy/fc_src_check.py "$V2D_FC_SRC" third_party/patches/flash_chord-8c778f9.sha256 >> "$LOG" 2>&1 \
  || stop "V2D_FC_SRC=${V2D_FC_SRC} 不是核对过的 8c778f9 副本（results/frames/${NAME}/fc_src.txt.failed），停下"
wait_gpu || stop "GPU 被别的进程占了 ${GPU_WAIT_S} 秒，停下"

ok=0; bad=0
row() {  # row <run> <步数> <状态> <退出码>
  local j="runs/flash_chord/$1/eval_dump/eval_${2}_w${W}.json" n="runs/flash_chord/$1/eval_dump/eval_${2}_w${W}.npz" s=""
  [ -s "$j" ] || j=""; [ -s "$n" ] && s="$(sha256sum "$n" | cut -d' ' -f1)" || n=""
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$j" "$n" "$s" >> "$OUT"
  case "$3" in ok|exists) ok=$((ok + 1)) ;; *) bad=$((bad + 1)) ;; esac
}
for row_ in "${ROWS[@]}"; do
  run=${row_% *}; step=${row_#* }; d="runs/flash_chord/$run"
  if dumped "$run" "$step"; then row "$run" "$step" exists 0; say "${run}：已有 dump，不重评"; continue; fi
  if [ -e "$d/NONFINITE" ]; then row "$run" "$step" nonfinite -; say "${run}：有 NONFINITE，不评"; continue; fi
  if [ ! -s "$d/policy_${step}.safetensors" ]; then row "$run" "$step" no_policy -; say "${run}：没有 policy_${step}.safetensors"; continue; fi
  t_row=$SECONDS
  while :; do
    rc=0
    V2D_EVAL_WORLDS="$W" scripts/dev/record "frames/${NAME}/${run}_s${step}" stages/evaluate/fc_eval_dump.sh "$run" "$step" >> "$LOG" 2>&1 < /dev/null || rc=$?
    [ "$rc" = 6 ] || break
    # 退出码 6：等 GPU 空了再重跑；每次重跑前先停 POLL_S 秒，这一行累计等满 GPU_WAIT_S 就停整条链（审查第 4 条）
    [ $((SECONDS - t_row)) -lt "$GPU_WAIT_S" ] || stop "${run} 这一行累计等 GPU 超过 ${GPU_WAIT_S} 秒，停下；之前的行已写进 $OUT"
    wait_gpu || stop "GPU 被别的进程占了 ${GPU_WAIT_S} 秒，停在 ${run}；之前的行已写进 $OUT"
    sleep "$POLL_S"
  done
  if [ "$rc" = 0 ] && dumped "$run" "$step"; then row "$run" "$step" ok 0; say "${run}：评完，dump 已存"
  else row "$run" "$step" failed "$rc"; say "${run}：fc_eval_dump 失败（退出码 ${rc}），见 results/frames/${NAME}/${run}_s${step}.txt.failed 与 runs/logs/${run}.eval_dump_${step}_w${W}.log"; fi
done
echo "=== $(date '+%F %T %Z') DUMP_CHAIN_DONE（成功 ${ok}、失败 ${bad}）" | tee -a "$LOG"
