#!/usr/bin/env bash
# v0.4.0 训出的 checkpoint 能否在 8c778f9 评测器里加载、导出（训练机上、仓库根执行，放 tmux）。计划 agent-doc/plan-box6-and-eval-runtime-2026-10-01.md 第 3.3、3.6 节。
# 用法：V2D_FC_SRC=<8c778f9 副本> [V2D_WAIT_FOR_EVAL_LOG=<上一段评测链日志>] stages/evaluate/compat_8c778f9.sh <清单> <日志名>
#       预检：V2D_COMPAT_DRYRUN=1 加上同样的环境与参数（见文末“预检”）
# 清单（stages/evaluate/compat/<机器>.tsv）每行 tab 分隔：<类型> <run> <步数> <世界数>；# 开头为注释。四种类型：
#   export  fc_export.sh 按 kit 的做法导出（metrics JSON + 轨迹 parquet），再用 check_export.sh（宿主机缺库时进容器跑 check_export.py）核结构与来历（判据 P1–P3）
#   eval    fc_export.sh 只写 metrics JSON（V2D_EXPORT_TRAJ=0），与同一 checkpoint 在 v0.4.0 评测器下的 eval_<步数>_w<世界数>.json
#           用 eval_compare.py 比：同一 sha256、ΔADD-AUC < 0.005、chord_sr |z| < 3（判据 P4，与 Day 2 恒等覆盖核对同一道闸）
#   guard、negctl  含 NaN 的 policy 的两道对照，判定见 compat_expect.sh
# 每行经 scripts/dev/record 落盘到 results/submission/compat/<清单名>_<类型>_<run>_w<世界数>*.txt（不过的是 .txt.failed）。
# 0. 日志 runs/logs/<日志名>.log 或门槛文件 runs/logs/<日志名>.gate 已经存在就拒绝（退出码 2，不写日志）：旧的结尾行会让
#    chain_after.sh 立刻开方盒链。清单读不了、里面没有 export 行，也拒绝。
# 1. 设了 V2D_WAIT_FOR_EVAL_LOG：每分钟看它的最后一行，变成 DAY2_EVAL_DONE / DAY2_EVAL_FAILED 才开始。从启动起就写长链标记
#    runs/.chain-active-<日志名>，等待期间也挡住同步。
# 2. GPU 上有进程、没设 V2D_FC_SRC、或它核不过（fc_src_check.py + flash_chord-8c778f9.sha256）：COMPAT_FAIL，停。
# 3. 逐行执行；某行不过不停，接着做下一行。
# 4. 门槛文件写一行 “DAY2_TRAIN_DONE（完成 1、失败 0、跳过 0、发散 0）”或“（完成 0、失败 1、…）”：清单里每个 export 行的
#    fc_export.sh 都退出码 0（评测器加载了 checkpoint、写出了 JSON 与轨迹）才是完成 1。方盒训练链的 chain_after.sh 把门槛文件
#    当“上一段训练日志”，门槛 1。check_export.py、eval、guard、negctl 不进门槛：它们管的是导出的格式与来历、已有结果能否沿用、
#    守卫灵不灵，与方盒怎么训无关；不过的照样记 COMPAT_FAIL，人来看。
# 5. 日志倒数第二行 COMPAT_PASS / COMPAT_FAIL，最后一行 DAY2_EVAL_DONE（成功 x、失败 y）：沿用评测链的结束行，chain_after.sh 靠它判断这一段结束。
# 预检（V2D_COMPAT_DRYRUN=1）：不等、不查 GPU、不写标记、日志与门槛，只往标准输出打。核：清单与日志名的拒绝条件、check_export 能跑
#   （宿主机有 numpy 与 pyarrow，或有 flash_chord:latest 镜像）、V2D_WAIT_FOR_EVAL_LOG 指的文件已存在、V2D_FC_SRC 核得过、eval 行的旧 JSON 存在；
#   每行调 fc_export.sh 的 V2D_EXPORT_DRYRUN=1（export、eval、negctl 要退出码 0，guard 要 12）。最后一行 COMPAT_DRYRUN_OK 或 COMPAT_DRYRUN_FAIL n。
set -uo pipefail
cd "$(dirname "$0")/../.."
LIST=${1:?用法见文件头}; NAME=${2:?用法见文件头}
TAG="$(basename "$LIST" .tsv)"
LOG="runs/logs/$NAME.log"; GATE="runs/logs/$NAME.gate"; MARK="runs/.chain-active-$NAME"
DRY="${V2D_COMPAT_DRYRUN:-0}"
[ -e "$LOG" ] || [ -e "$GATE" ] && { echo "compat: $LOG 或 $GATE 已存在，拒绝（旧的结尾行会让 chain_after.sh 立刻开方盒链；确认后改名或删掉再起）" >&2; exit 2; }
[ -r "$LIST" ] || { echo "compat: 读不了清单 $LIST" >&2; exit 2; }
NEXP="$(grep -c "^export	" "$LIST")"
[ "$NEXP" -gt 0 ] || { echo "compat: 清单 $LIST 里没有 export 行" >&2; exit 2; }

if [ "$DRY" = 1 ]; then
  bad=0
  chk() { if eval "$2" >/dev/null 2>&1; then echo "  ok   $1"; else echo "  FAIL $1"; bad=$((bad + 1)); fi; }
  echo "== 预检 ${NAME}：清单 ${LIST}（export 行 ${NEXP}），V2D_FC_SRC=${V2D_FC_SRC:-未设}，来历 $(cat .v2d-source-commit 2>/dev/null || echo unknown)"
  echo "  ok   $LOG 与 $GATE 都不存在"
  chk "check_export 能跑（宿主机 python3 有 numpy、pyarrow，或经 flash_chord:latest 容器）：$(stages/evaluate/check_export.sh --probe 2>&1)" "stages/evaluate/check_export.sh --probe"
  if [ -n "${V2D_WAIT_FOR_EVAL_LOG:-}" ]; then chk "要等的评测链日志 $V2D_WAIT_FOR_EVAL_LOG 已存在" "[ -f '$V2D_WAIT_FOR_EVAL_LOG' ]"; fi
  chk "V2D_FC_SRC 是核对过的 8c778f9 副本" "[ -n '${V2D_FC_SRC:-}' ] && python3 stages/policy/fc_src_check.py '${V2D_FC_SRC:-/nonexistent}' third_party/patches/flash_chord-8c778f9.sha256"
  chk "third_party/patches/flash_chord-8c778f9.eval.json 存在" "[ -s third_party/patches/flash_chord-8c778f9.eval.json ]"
  exec 3< "$LIST"
  while IFS=$'\t' read -r -u 3 kind run step w _; do
    case "$kind" in ''|'#'*) continue ;; esac
    case "$kind" in
      export|eval) want=0; unguarded=0 ;;
      guard) want=12; unguarded=0 ;;
      negctl) want=0; unguarded=1 ;;
      *) echo "  FAIL 未知类型 ${kind}（${run}）"; bad=$((bad + 1)); continue ;;
    esac
    traj=1; [ "$kind" = eval ] && traj=0
    out="$(V2D_EXPORT_DRYRUN=1 V2D_EXPORT_WORLDS="$w" V2D_EXPORT_TRAJ="$traj" V2D_EXPORT_UNGUARDED="$unguarded" stages/evaluate/fc_export.sh "$run" "$step" 2>&1 < /dev/null)"; rc=$?
    if [ "$rc" = "$want" ]; then echo "  ok   $kind ${run}（${w} 世界）：fc_export 预检退出码 ${rc}"; else echo "  FAIL $kind ${run}（${w} 世界）：fc_export 预检退出码 ${rc}，应为 ${want}：$(tail -1 <<< "$out")"; bad=$((bad + 1)); fi
    [ "$kind" = eval ] && chk "eval $run 的 v0.4.0 结果 runs/flash_chord/$run/eval_${step}_w${w}.json 存在" "[ -s 'runs/flash_chord/$run/eval_${step}_w${w}.json' ]"
  done
  exec 3<&-
  if [ "$bad" = 0 ]; then echo "COMPAT_DRYRUN_OK"; exit 0; else echo "COMPAT_DRYRUN_FAIL $bad"; exit 1; fi
fi

mkdir -p runs/logs
touch "$MARK"; trap 'rm -f "$MARK"' EXIT
say() { echo "=== $(date '+%F %T %Z') $*" | tee -a "$LOG"; }
ok=0; bad=0; xok=0; xbad=0
gate() {
  if [ "$xok" = "$NEXP" ] && [ "$xbad" = 0 ]; then echo "=== $(date '+%F %T %Z') DAY2_TRAIN_DONE（完成 1、失败 0、跳过 0、发散 0）" > "$GATE"
  else echo "=== $(date '+%F %T %Z') DAY2_TRAIN_DONE（完成 0、失败 1、跳过 0、发散 0）" > "$GATE"; fi
}
stop() { say "$*"; gate; say "COMPAT_FAIL（没做完）"; say "DAY2_EVAL_FAILED"; exit 1; }
say "start ${NAME}：清单 ${LIST}（export 行 ${NEXP}），V2D_FC_SRC=${V2D_FC_SRC:-未设}，来历 $(cat .v2d-source-commit 2>/dev/null || echo unknown)"
if [ -n "${V2D_WAIT_FOR_EVAL_LOG:-}" ]; then
  # 本日志里只有最后那一行可以出现评测链结束标记的原文：chain_after.sh 盯着本日志的最后一行认它（10/1 起链时，等待提示里带了
  # 这两个词，chain_after 当场误判 compat 已结束，见 agent-doc/experiments/2026-10-02-box6-eval-runtime.md）。
  say "等 $V2D_WAIT_FOR_EVAL_LOG 结束（它的最后一行变成评测链的结束行）"
  until grep -qE "DAY2_EVAL_(DONE|FAILED)" < <(tail -1 "$V2D_WAIT_FOR_EVAL_LOG" 2>/dev/null); do sleep "${V2D_POLL_S:-60}"; done
  say "上一段评测链已结束：$(tail -1 "$V2D_WAIT_FOR_EVAL_LOG" | sed -E 's/DAY2_EVAL_DONE/评测链完成/; s/DAY2_EVAL_FAILED/评测链失败/')"
fi
others="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')"
[ -z "$others" ] || stop "GPU 上有别的进程（${others}），停下"
[ -n "${V2D_FC_SRC:-}" ] || stop "没设 V2D_FC_SRC（要 make_fc_src_8c778f9.sh 生成的副本），停下"
scripts/dev/record "submission/compat/${TAG}_fc_src" python3 stages/policy/fc_src_check.py "$V2D_FC_SRC" third_party/patches/flash_chord-8c778f9.sha256 >> "$LOG" 2>&1 \
  || stop "V2D_FC_SRC=${V2D_FC_SRC} 不是核对过的 8c778f9 副本（results/submission/compat/${TAG}_fc_src.txt.failed），停下"

exec 3< "$LIST"
while IFS=$'\t' read -r -u 3 kind run step w _; do
  case "$kind" in ''|'#'*) continue ;; esac
  key="submission/compat/${TAG}_${kind}_${run}_w${w}"
  case "$kind" in
    export)
      if V2D_EXPORT_WORLDS="$w" scripts/dev/record "$key" stages/evaluate/fc_export.sh "$run" "$step" >> "$LOG" 2>&1 < /dev/null; then
        xok=$((xok + 1))
        if scripts/dev/record "${key}_check" stages/evaluate/check_export.sh "$run" "$step" "$w" >> "$LOG" 2>&1 < /dev/null; then
          ok=$((ok + 1)); say "export ${run}（${w} 世界）：导出、核对都通过"
        else
          bad=$((bad + 1)); say "export ${run}（${w} 世界）：导出成功，核对不过（不进门槛），见 results/${key}_check.txt.failed"
        fi
      else
        xbad=$((xbad + 1)); bad=$((bad + 1)); say "export ${run}（${w} 世界）：导出失败（进门槛），见 results/${key}.txt.failed"
      fi ;;
    eval)
      d="runs/flash_chord/$run/export_8c778f9/s${step}_w${w}"
      if V2D_EXPORT_WORLDS="$w" V2D_EXPORT_TRAJ=0 scripts/dev/record "$key" stages/evaluate/fc_export.sh "$run" "$step" >> "$LOG" 2>&1 < /dev/null \
         && j="$(ls "$d"/episode_*.json 2>/dev/null | head -1)" && [ -n "$j" ] \
         && scripts/dev/record "${key}_compare" python3 stages/policy/eval_compare.py "$j" "runs/flash_chord/$run/eval_${step}_w${w}.json" \
              --max-add-auc-diff 0.005 --max-chord-z 3 >> "$LOG" 2>&1 < /dev/null; then
        ok=$((ok + 1)); say "eval ${run}（${w} 世界）：与 v0.4.0 评测器的结果一致（CONSISTENT）"
      else
        bad=$((bad + 1)); say "eval ${run}（${w} 世界）：不一致或没评完，见 results/${key}*"
      fi ;;
    guard|negctl)
      if V2D_RECORD_TRACEBACK_OK=1 scripts/dev/record "$key" stages/evaluate/compat_expect.sh "$kind" "$run" "$step" "$w" >> "$LOG" 2>&1 < /dev/null; then
        ok=$((ok + 1)); say "$kind ${run}：符合期望（$(grep -h 'COMPAT_EXPECT_OK' "results/$key.txt" | tail -1)）"
      else
        bad=$((bad + 1)); say "$kind ${run}：不符合期望，见 results/${key}.txt.failed"
      fi ;;
    *) bad=$((bad + 1)); say "清单里有未知类型 ${kind}（${run}），跳过" ;;
  esac
done
exec 3<&-
gate
if [ "$bad" = 0 ]; then say "COMPAT_PASS（${ok} 行全部符合）"
else say "COMPAT_FAIL（${bad} 行不符合，共 $((ok + bad)) 行；export 导出成功 ${xok} / ${NEXP}，门槛$([ "$xok" = "$NEXP" ] && [ "$xbad" = 0 ] && echo '过' || echo '不过')）"; fi
say "DAY2_EVAL_DONE（成功 ${ok}、失败 ${bad}）"
