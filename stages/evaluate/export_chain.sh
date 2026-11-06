#!/usr/bin/env bash
# 提交用的批量导出（训练机上、仓库根执行，放 tmux）：按导出清单把每条 episode 的 policy 用 8c778f9 评测器导成 1 个世界的轨迹。不上传。
# 计划 agent-doc/plan-submission-exports-2026-10-02.md 第 3.2 节（第 2 版，按 Fable 5.1 的审查修改）。
# 用法：V2D_FC_SRC=<8c778f9 副本> [V2D_WAIT_FOR_EVAL_LOG=<上一段评测链日志>] [V2D_COMPAT_GATE=<compat 门槛文件>] \
#         stages/evaluate/export_chain.sh <导出清单> <日志名>
#       预检：V2D_EXPORT_CHAIN_DRYRUN=1 加上同样的环境与参数（见“预检”一段）
#       导出都已在、只缺结果表、训练占着 GPU 时：再加 V2D_GPU_WAIT_START=0 V2D_GPU_WAIT_S=0（见第 2 步）
# 清单（stages/evaluate/exports/<机器>.tsv，export_map.py 生成）每行 tab 分隔：档位 episode run 步数；# 开头为注释。
# 0. 日志 runs/logs/<日志名>.log 或结果表 runs/exports/<日志名>.tsv 已存在、清单读不了或没有行：拒绝（退出码 2，不写日志）。
# 1. 设了 V2D_WAIT_FOR_EVAL_LOG：每分钟看它的最后一行，变成 DAY2_EVAL_DONE / DAY2_EVAL_FAILED 才开始。从启动起就写长链标记
#    runs/.chain-active-<日志名>。本日志里不出现任何链结束标记的原文（抄进来的行把标记换成中文；10/1 起链事故的教训）。
# 2. 设了 V2D_COMPAT_GATE：门槛不是“完成 1”只记警告、照样导出（真加载不了的话每行会自己很快失败；只有个别行失败时，其余行不受连累）。
#    GPU 上有进程就每分钟看一次，最多等 V2D_GPU_WAIT_S 秒（默认 7200），等不到就停；没设 V2D_FC_SRC 或它核不过，停。
#    V2D_GPU_WAIT_START=0：开头这次不等（默认 1 = 等）。fc_export 遇到已有完整结果（JSON 与要求的 parquet）直接返回、不查 GPU，
#    所以导出都已在、只缺结果表时，训练占着 GPU 也能做；要真导出的行仍由 fc_export 查 GPU（退出码 6），按第 3 步等，V2D_GPU_WAIT_S=0 就立刻停。
# 3. 逐行（某行不过不停）：
#    run 没有 DONE → no_done；有 NONFINITE，或最终一步 fc_export 退出码 12（规矩一：含 NaN 的 policy 不导出；不回退，记缺，等用户定）→ nonfinite。
#    否则 fc_export.sh 导出（1 个世界、写轨迹），再 check_export.sh 核结构与来历，再核“候选”条件（box6 计划第 4.5 节）：
#    这一步有 4096 世界评测（runs/flash_chord/<run>/eval_<步数>_w4096.json），world_count 4096、non_finite_worlds 为空、
#    step_count 与 checkpoint_sha256 和这次导出相同。
#      都过 → ok；候选条件不过 → no_eval（轨迹留着，拉回时进 held/，人来看）；
#      fc_export 退出码 13（评测器报 must disable task failure termination）→ 规矩二：按步数从大到小试同一 run 里更早的 policy，
#        最多 V2D_FALLBACK_MAX 份（默认 8）：退出码 12 跳过，13 继续往前，导出与核对都过的那一步就用；别的失败停在这一条（fallback_failed）。
#        用上的那一步在 8c778f9 下补评 4096 世界（只写 JSON，在 export_8c778f9/s<步数>_w4096/ 下）：补评成功且满足同样的候选条件 → fallback；
#        否则 → fallback_noeval（不进打包目录，人来看）；
#      fc_export 其他退出码 → export_failed；check_export 不过 → check_failed；导出文件的 episode 号与清单不符 → ep_mismatch。
#    fc_export 退出码 6（GPU 被别的进程占了）：等 GPU 空了重跑这一行（同样最多等 V2D_GPU_WAIT_S 秒，等不到就停整条链）。
#    每行写进结果表 runs/exports/<日志名>.tsv：档位 episode run 用的步数 状态 JSON parquet JSON 的 sha256 parquet 的 sha256。
#    每次导出、核对、补评经 scripts/dev/record 落盘到 results/submission/exports/<日志名>_<run>_s<步数>[_check|_w4096].txt。
#    fc_export 的退出码就是结论（摘要脚本出错不改变结论），所以对它设 V2D_RECORD_TRACEBACK_OK=1。
# 4. 日志最后一行 EXPORT_CHAIN_DONE（成功 x、失败 y）（成功 = ok + fallback），停在中途时是 EXPORT_CHAIN_FAILED。
set -uo pipefail
cd "$(dirname "$0")/../.."
LIST=${1:?用法见文件头}; NAME=${2:?用法见文件头}
LOG="runs/logs/$NAME.log"; OUT="runs/exports/$NAME.tsv"; MARK="runs/.chain-active-$NAME"
DRY="${V2D_EXPORT_CHAIN_DRYRUN:-0}"; FMAX="${V2D_FALLBACK_MAX:-8}"; GPU_WAIT_S="${V2D_GPU_WAIT_S:-7200}"; WAIT_START="${V2D_GPU_WAIT_START:-1}"
[ -e "$LOG" ] || [ -e "$OUT" ] && { echo "export_chain: $LOG 或 $OUT 已存在，拒绝（确认后改名或删掉再起）" >&2; exit 2; }
[ -r "$LIST" ] || { echo "export_chain: 读不了清单 $LIST" >&2; exit 2; }
NROW="$(grep -cvE '^(#|$)' "$LIST")"
[ "$NROW" -gt 0 ] || { echo "export_chain: 清单 $LIST 没有行" >&2; exit 2; }
[[ $FMAX =~ ^[0-9]+$ && $GPU_WAIT_S =~ ^[0-9]+$ ]] || { echo "export_chain: V2D_FALLBACK_MAX、V2D_GPU_WAIT_S 必须是非负整数" >&2; exit 2; }
[[ $WAIT_START =~ ^[01]$ ]] || { echo "export_chain: V2D_GPU_WAIT_START 只能是 0 或 1" >&2; exit 2; }
demark() { sed -E 's/DAY2_EVAL_DONE/评测链完成/g; s/DAY2_EVAL_FAILED/评测链失败/g; s/DAY2_TRAIN_DONE/训练链完成/g; s/DAY2_TRAIN_FAILED/训练链失败/g; s/EXPORT_CHAIN_(DONE|FAILED)/导出链结束/g'; }
eval_ok() {  # eval_ok <4096 世界评测 JSON> <这次导出的 JSON>：打一行说明；满足候选条件返回 0
  python3 - "$1" "$2" <<'PY'
import json, sys
try:
    e, x = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
except Exception as err:   # noqa: BLE001  没评、读不了都算不满足
    print(f"没有可用的 4096 世界评测（{type(err).__name__}: {err}）"); sys.exit(1)
nf = e.get("non_finite_worlds")
ok = (e.get("world_count") == 4096 and nf == [] and e.get("step_count") == x.get("step_count")
      and e.get("checkpoint_sha256") == x.get("checkpoint_sha256"))
print(f"4096 世界评测：world_count {e.get('world_count')}，non_finite_worlds {len(nf) if isinstance(nf, list) else nf} 个，"
      f"step_count {e.get('step_count')}（导出 {x.get('step_count')}），checkpoint {'相同' if e.get('checkpoint_sha256') == x.get('checkpoint_sha256') else '不同'}")
sys.exit(0 if ok else 1)
PY
}

if [ "$DRY" = 1 ]; then
  bad=0; pend=0
  chk() { if eval "$2" >/dev/null 2>&1; then echo "  ok   $1"; else echo "  FAIL $1"; bad=$((bad + 1)); fi; }
  echo "== 预检 ${NAME}：清单 ${LIST}（${NROW} 行），V2D_FC_SRC=${V2D_FC_SRC:-未设}，来历 $(cat .v2d-source-commit 2>/dev/null || echo unknown)"
  echo "  ok   $LOG 与 $OUT 都不存在"
  chk "check_export 能跑：$(stages/evaluate/check_export.sh --probe 2>&1)" "stages/evaluate/check_export.sh --probe"
  [ "$WAIT_START" = 1 ] || echo "  --   开头不等 GPU（V2D_GPU_WAIT_START=0）；要真导出的行遇到 GPU 被占，最多等 ${GPU_WAIT_S} 秒"
  if [ -n "${V2D_WAIT_FOR_EVAL_LOG:-}" ]; then chk "要等的评测链日志 $V2D_WAIT_FOR_EVAL_LOG 已存在" "[ -f '$V2D_WAIT_FOR_EVAL_LOG' ]"; fi
  [ -n "${V2D_COMPAT_GATE:-}" ] && echo "  --   门槛 ${V2D_COMPAT_GATE}：$( (cat "$V2D_COMPAT_GATE" 2>/dev/null || echo 还没写，开跑时再看) | demark)"
  chk "V2D_FC_SRC 是核对过的 8c778f9 副本" "[ -n '${V2D_FC_SRC:-}' ] && python3 stages/policy/fc_src_check.py '${V2D_FC_SRC:-/nonexistent}' third_party/patches/flash_chord-8c778f9.sha256"
  exec 3< "$LIST"
  while IFS=$'\t' read -r -u 3 tier ep run step _; do
    case "$tier" in ''|'#'*) continue ;; esac
    d="runs/flash_chord/$run"
    if [ ! -f "$d/DONE" ]; then echo "  --   Tier $tier ep${ep} ${run}：还没有 DONE（还在训或排队），开跑时再看"; pend=$((pend + 1)); continue; fi
    if [ -f "$d/NONFINITE" ]; then echo "  --   Tier $tier ep${ep} ${run}：有 NONFINITE，开跑时按规矩一记缺"; pend=$((pend + 1)); continue; fi
    out="$(V2D_EXPORT_DRYRUN=1 V2D_EXPORT_WORLDS=1 stages/evaluate/fc_export.sh "$run" "$step" 2>&1 < /dev/null)"; rc=$?
    case "$rc" in
      0) echo "  ok   Tier $tier ep${ep} ${run}：fc_export 预检退出码 0（$(head -1 <<< "$out" | sed 's/^fc_export: DRYRUN //')）" ;;
      12) echo "  --   Tier $tier ep${ep} ${run}：fc_export 预检退出码 12（policy 非有限或读不了），开跑时按规矩一记缺"; pend=$((pend + 1)); continue ;;
      *) echo "  FAIL Tier $tier ep${ep} ${run}：fc_export 预检退出码 ${rc}：$(tail -1 <<< "$out")"; bad=$((bad + 1)); continue ;;
    esac
    e="$d/eval_${step}_w4096.json"
    if [ -s "$e" ] && python3 -c 'import json, sys; e = json.load(open(sys.argv[1])); sys.exit(0 if e.get("world_count") == 4096 and e.get("non_finite_worlds") == [] else 1)' "$e" 2>/dev/null; then
      echo "       有干净的 4096 世界评测 $e"
    else
      echo "  --   Tier $tier ep${ep} ${run}：还没有干净的 4096 世界评测 ${e}，开跑时若仍没有就记 no_eval"
    fi
  done
  exec 3<&-
  echo "  --   还没训完、有 NONFINITE 或 policy 非有限的 ${pend} 行，开跑时按 no_done / nonfinite 记"
  if [ "$bad" = 0 ]; then echo "EXPORT_DRYRUN_OK"; exit 0; else echo "EXPORT_DRYRUN_FAIL $bad"; exit 1; fi
fi

mkdir -p runs/logs runs/exports
touch "$MARK"; trap 'rm -f "$MARK"' EXIT
say() { echo "=== $(date '+%F %T %Z') $*" | tee -a "$LOG"; }
stop() { say "$*"; say "EXPORT_CHAIN_FAILED"; exit 1; }
gpu_pids() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | paste -sd, -; }
wait_gpu() {  # GPU 上有进程就每分钟看一次，最多 GPU_WAIT_S 秒；空了返回 0
  local t0=$SECONDS p
  p="$(gpu_pids)"; [ -n "$p" ] || return 0
  say "GPU 上有别的进程（${p}），每 ${V2D_POLL_S:-60} 秒看一次，最多等 ${GPU_WAIT_S} 秒"
  while [ -n "$(gpu_pids)" ]; do
    [ $((SECONDS - t0)) -lt "$GPU_WAIT_S" ] || return 1
    sleep "${V2D_POLL_S:-60}"
  done
  say "GPU 空了，接着做"
}
printf 'tier\tepisode\trun\tstep\tstatus\tjson\tparquet\tjson_sha256\tparquet_sha256\n' > "$OUT"
say "start ${NAME}：清单 ${LIST}（${NROW} 行），V2D_FC_SRC=${V2D_FC_SRC:-未设}，回退最多 ${FMAX} 份，GPU 最多等 ${GPU_WAIT_S} 秒，来历 $(cat .v2d-source-commit 2>/dev/null || echo unknown)"
if [ -n "${V2D_WAIT_FOR_EVAL_LOG:-}" ]; then
  say "等 $V2D_WAIT_FOR_EVAL_LOG 结束（它的最后一行变成评测链的结束行）"
  until grep -qE "DAY2_EVAL_(DONE|FAILED)" < <(tail -1 "$V2D_WAIT_FOR_EVAL_LOG" 2>/dev/null); do sleep "${V2D_POLL_S:-60}"; done
  say "上一段评测链已结束：$(tail -1 "$V2D_WAIT_FOR_EVAL_LOG" | demark)"
fi
if [ -n "${V2D_COMPAT_GATE:-}" ]; then
  if grep -q '完成 1、' "$V2D_COMPAT_GATE" 2>/dev/null; then say "compat 门槛是“完成 1”"
  else say "警告：compat 门槛 $V2D_COMPAT_GATE 不是“完成 1”，照样导出，逐行看结果：$( (cat "$V2D_COMPAT_GATE" 2>/dev/null || echo 没有门槛文件) | demark)"; fi
fi
if [ "$WAIT_START" = 1 ]; then
  wait_gpu || stop "GPU 被别的进程占了 ${GPU_WAIT_S} 秒，停下"
else
  say "开头不等 GPU（V2D_GPU_WAIT_START=0）：已有完整导出的行不碰 GPU；要真导出的行遇到 GPU 被占，最多等 ${GPU_WAIT_S} 秒"
fi
[ -n "${V2D_FC_SRC:-}" ] || stop "没设 V2D_FC_SRC，停下"
scripts/dev/record "submission/exports/${NAME}_fc_src" python3 stages/policy/fc_src_check.py "$V2D_FC_SRC" third_party/patches/flash_chord-8c778f9.sha256 >> "$LOG" 2>&1 \
  || stop "V2D_FC_SRC=${V2D_FC_SRC} 不是核对过的 8c778f9 副本（results/submission/exports/${NAME}_fc_src.txt.failed），停下"

ok=0; bad=0
shaof() { [ -f "$1" ] && sha256sum "$1" | cut -d' ' -f1; }
xjson() { echo "runs/flash_chord/$1/export_8c778f9/s${2}_w1/episode_$(printf %06d "$3").json"; }   # xjson <run> <步数> <ep>
row() {  # row <档位> <ep> <run> <步数> <状态>：步数是这一行最后用的（或试过的）那一步
  local j q="" st=$5
  j="$(xjson "$3" "$4" "$2")"
  if [ -f "$j" ]; then
    [ -f "${j%.json}.parquet" ] && q="${j%.json}.parquet"
  else
    if ls "runs/flash_chord/$3/export_8c778f9/s${4}_w1"/episode_*.json >/dev/null 2>&1 && [ "$st" != no_done ] && [ "$st" != nonfinite ]; then
      st=ep_mismatch; say "  Tier $1 ep${2} ${3}：导出文件的 episode 号与清单不符（$(ls "runs/flash_chord/$3/export_8c778f9/s${4}_w1"/episode_*.json | head -1)），记 ep_mismatch"
    fi
    j=""
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$st" "$j" "$q" "$(shaof "$j")" "$(shaof "$q")" >> "$OUT"
  case "$st" in ok|fallback) ok=$((ok + 1)) ;; *) bad=$((bad + 1)) ;; esac
}
fcx() {  # fcx <record 名> <世界数> <轨迹 0|1> <run> <步数>：跑 fc_export；退出码 6（GPU 被占）就等 GPU 空了重跑，等不到就停整条链
  local rc
  while :; do
    rc=0
    V2D_RECORD_TRACEBACK_OK=1 V2D_EXPORT_WORLDS="$2" V2D_EXPORT_TRAJ="$3" scripts/dev/record "$1" \
      stages/evaluate/fc_export.sh "$4" "$5" >> "$LOG" 2>&1 < /dev/null || rc=$?
    [ "$rc" = 6 ] || return "$rc"
    wait_gpu || stop "GPU 被别的进程占了 ${GPU_WAIT_S} 秒，停在 $4 第 $5 步；之前的行已写进 $OUT"
  done
}
export_one() {  # export_one <run> <步数>：0 导出与核对都过；12、13 是 fc_export 的退出码；3 导出其他失败；4 核对不过
  local rc=0
  fcx "submission/exports/${NAME}_${1}_s${2}" 1 1 "$1" "$2" || rc=$?
  case "$rc" in 0) ;; 12|13) return "$rc" ;; *) return 3 ;; esac
  scripts/dev/record "submission/exports/${NAME}_${1}_s${2}_check" stages/evaluate/check_export.sh "$1" "$2" 1 >> "$LOG" 2>&1 < /dev/null || return 4
  return 0
}
exec 3< "$LIST"
while IFS=$'\t' read -r -u 3 tier ep run step _; do
  case "$tier" in ''|'#'*) continue ;; esac
  d="runs/flash_chord/$run"
  if [ ! -f "$d/DONE" ]; then row "$tier" "$ep" "$run" "$step" no_done; say "Tier $tier ep${ep} ${run}：没有 DONE，不导出"; continue; fi
  if [ -f "$d/NONFINITE" ]; then row "$tier" "$ep" "$run" "$step" nonfinite; say "Tier $tier ep${ep} ${run}：有 NONFINITE（规矩一），不导出，这条记缺"; continue; fi
  export_one "$run" "$step"; rc=$?
  case "$rc" in
    0)
      if msg="$(eval_ok "$d/eval_${step}_w4096.json" "$(xjson "$run" "$step" "$ep")")"; then
        row "$tier" "$ep" "$run" "$step" ok; say "Tier $tier ep${ep} ${run}：第 ${step} 步导出、核对都通过；${msg}"
      else
        row "$tier" "$ep" "$run" "$step" no_eval; say "Tier $tier ep${ep} ${run}：第 ${step} 步导出、核对都通过，但不满足候选条件（${msg}），记 no_eval"
      fi
      continue ;;
    4) row "$tier" "$ep" "$run" "$step" check_failed; say "Tier $tier ep${ep} ${run}：第 ${step} 步导出了，但 check_export 不过"; continue ;;
    12) row "$tier" "$ep" "$run" "$step" nonfinite; say "Tier $tier ep${ep} ${run}：第 ${step} 步的 policy 非有限或读不了（规矩一），不导出，这条记缺"; continue ;;
    13) ;;
    *) row "$tier" "$ep" "$run" "$step" export_failed; say "Tier $tier ep${ep} ${run}：第 ${step} 步导出失败"; continue ;;
  esac
  say "Tier $tier ep${ep} ${run}：第 ${step} 步报 must disable task failure termination，按规矩二往前退（最多 ${FMAX} 份）"
  used=""; tried=0; last="$step"
  for s in $(ls "$d"/policy_*.safetensors 2>/dev/null | sed -E 's/.*policy_([0-9]+)\.safetensors$/\1/' | sort -rn); do
    [ "$s" -lt "$step" ] || continue
    [ "$tried" -lt "$FMAX" ] || { say "  已试 ${FMAX} 份，不再往前"; break; }
    tried=$((tried + 1)); last="$s"
    export_one "$run" "$s"; r2=$?
    case "$r2" in
      0) used="$s"; break ;;
      12) say "  第 ${s} 步的 policy 非有限或读不了，跳过" ;;
      13) say "  第 ${s} 步同样报终止错误，继续往前" ;;
      *) say "  第 ${s} 步导出或核对失败（${r2}），停在这一条"; break ;;
    esac
  done
  if [ -z "$used" ]; then row "$tier" "$ep" "$run" "$last" fallback_failed; say "Tier $tier ep${ep} ${run}：回退没有找到能用的 checkpoint"; continue; fi
  w="runs/flash_chord/$run/export_8c778f9/s${used}_w4096/episode_$(printf %06d "$ep").json"
  msg=""
  if fcx "submission/exports/${NAME}_${run}_s${used}_w4096" 4096 0 "$run" "$used" && msg="$(eval_ok "$w" "$(xjson "$run" "$used" "$ep")")"; then
    row "$tier" "$ep" "$run" "$used" fallback; say "Tier $tier ep${ep} ${run}：按规矩二改用第 ${used} 步；${msg}"
  else
    row "$tier" "$ep" "$run" "$used" fallback_noeval; say "Tier $tier ep${ep} ${run}：按规矩二改用第 ${used} 步，但 4096 世界补评失败或不满足候选条件（${msg:-补评失败}），不进打包目录，人来看"
  fi
done
exec 3<&-
say "EXPORT_CHAIN_DONE（成功 ${ok}、失败 ${bad}）"
