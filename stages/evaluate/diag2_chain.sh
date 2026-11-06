#!/usr/bin/env bash
# MP-SR 为 0 的序列诊断链（训练机上、仓库根执行，放 tmux）。计划 agent-doc/plan-diag-mpsr0-2026-10-02.md 第 5 节第 4 步。
# 用法：[V2D_WAIT_FOR_LOG=<本机导出链日志>] stages/evaluate/diag2_chain.sh <运行表> <日志名>
#       运行表 stages/evaluate/diag2/<机器>.tsv，每行 tab 分隔：run 步数 角色（control | main）；# 开头为注释。
# 0. 日志 runs/logs/<日志名>.log 已存在、运行表读不了或没有行、某行格式不对：拒绝（退出码 2，不写日志）。
# 1. 设了 V2D_WAIT_FOR_LOG：等它的最后一行变成导出链结束行（EXPORT_CHAIN_DONE / EXPORT_CHAIN_FAILED）。从启动起写长链标记 runs/.chain-active-<日志名>。
#    本日志里不出现任何链结束标记的原文（抄进来的行把标记换成中文）。
# 2. GPU 上有进程就每分钟看一次，最多等 V2D_GPU_WAIT_S 秒（默认 7200），等不到就停；每次用 GPU（试跑、每行的诊断 rollout）之前都这样等一次（第 2 版）。
#    诊断与已有评测用同一运行时：默认挂的上游 v0.4.0，本脚本开头 unset V2D_FC_SRC。
#    例外（计划 agent-doc/plan-t2dg-2026-10-20.md）：设了 V2D_DIAG_FC_SRC=<源码副本的绝对路径> 时，诊断与试跑都挂这份副本，
#    且只做那份 4096 世界评测日志的挂载说明行正好指向同一份副本（“/workspace ← <路径>（”）的行；没有挂载行或指向别的副本就跳过、记出错。
# 3. 第一行先试跑 16 个世界（diag_rollout.py --palms），再用 diag2_sync.py 看同步闸；不过就停（DIAG2_FAILED），后面的行都不跑。
# 4. 逐行：policy、本机已有的 eval_<步数>_w4096.json 与它的本机日志都要在（同卡比对靠它），且那份日志里没有 in-flash-chord 的挂载说明行
#    （只有设了 V2D_FC_SRC 才打这行；没有就说明是默认的 v0.4.0 算的，与这次诊断同一运行时）；diag_rollout.py --palms 4096 世界
#    → runs/diag2/<run>/rollout_w4096.{npz,json}；diag_phase.py（--role 按运行表）→ results/diag2/phase_<run>.txt。
#    diag_phase 退出码 0 记完成，3 记闸不过，4 记对照不过（都继续下一行）；其他退出码或 rollout 失败记出错，也继续下一行。
#    每行跑完删掉 hydra 目录，npz 留着（viser 复看用）。每一步经 scripts/dev/record 落盘到 results/diag2/。
# 5. 日志最后一行 DIAG2_DONE（完成 a、闸不过 b、对照不过 c、出错 d），停在开头或试跑时是 DIAG2_FAILED。
set -uo pipefail
cd "$(dirname "$0")/../.."
unset V2D_FC_SRC
DIAG_SRC="${V2D_DIAG_FC_SRC:-}"
[ -z "$DIAG_SRC" ] || [[ $DIAG_SRC == /* ]] || { echo "diag2_chain: V2D_DIAG_FC_SRC 必须是绝对路径：$DIAG_SRC" >&2; exit 2; }
if [ -n "$DIAG_SRC" ]; then   # 与 in-flash-chord 打印的挂载路径同样解析（pwd -P），末尾斜杠、软链接都不影响匹配
  DIAG_SRC="$(cd "$DIAG_SRC" 2>/dev/null && pwd -P)" || { echo "diag2_chain: V2D_DIAG_FC_SRC 不是目录：$V2D_DIAG_FC_SRC" >&2; exit 2; }
fi
TABLE=${1:?用法见文件头}; NAME=${2:?用法见文件头}
LOG="runs/logs/$NAME.log"; MARK="runs/.chain-active-$NAME"; GPU_WAIT_S="${V2D_GPU_WAIT_S:-7200}"
[ ! -e "$LOG" ] || { echo "diag2_chain: $LOG 已存在，拒绝（确认后改名或删掉再起）" >&2; exit 2; }
[ -r "$TABLE" ] || { echo "diag2_chain: 读不了运行表 $TABLE" >&2; exit 2; }
rows=0
while IFS=$'\t' read -r run step role _; do
  case "$run" in ''|'#'*) continue ;; esac
  [[ $run =~ ^t[12]-[a-z0-9_-]+$ && $step =~ ^[0-9]+$ && $role =~ ^(control|main)$ ]] || { echo "diag2_chain: 运行表的行不对：$run $step $role" >&2; exit 2; }
  rows=$((rows + 1))
done < "$TABLE"
[ "$rows" -gt 0 ] || { echo "diag2_chain: 运行表 $TABLE 没有行" >&2; exit 2; }
[[ $GPU_WAIT_S =~ ^[0-9]+$ ]] || { echo "diag2_chain: V2D_GPU_WAIT_S 必须是非负整数" >&2; exit 2; }
mkdir -p runs/logs runs/diag2 results/diag2
touch "$MARK"; trap 'rm -f "$MARK"' EXIT
say() { echo "=== $(date '+%F %T %Z') $*" | tee -a "$LOG"; }
stop() { say "$*"; say "DIAG2_FAILED"; exit 1; }
demark() { sed -E 's/EXPORT_CHAIN_DONE/导出链完成/g; s/EXPORT_CHAIN_FAILED/导出链失败/g; s/DAY2_(EVAL|TRAIN)_(DONE|FAILED)/上一段链结束/g; s/DIAG2_(DONE|FAILED)/诊断链结束/g'; }
gpu_pids() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | paste -sd, -; }
say "start ${NAME}：运行表 ${TABLE}（${rows} 行），运行时为${DIAG_SRC:+源码副本 }${DIAG_SRC:-默认的上游 v0.4.0}，GPU 最多等 ${GPU_WAIT_S} 秒，来历 $(cat .v2d-source-commit 2>/dev/null || echo unknown)"
if [ -n "${V2D_WAIT_FOR_LOG:-}" ]; then
  say "等 $V2D_WAIT_FOR_LOG 结束（它的最后一行变成导出链的结束行）"
  until grep -qE "EXPORT_CHAIN_(DONE|FAILED)" < <(tail -1 "$V2D_WAIT_FOR_LOG" 2>/dev/null); do sleep "${V2D_POLL_S:-60}"; done
  say "上一段已结束：$(tail -1 "$V2D_WAIT_FOR_LOG" | demark)"
fi
wait_gpu() {  # GPU 上有进程就每分钟看一次，最多 GPU_WAIT_S 秒；等不到就停整条链
  local t0=$SECONDS
  [ -z "$(gpu_pids)" ] && return 0
  say "GPU 上有别的进程（$(gpu_pids)），每 ${V2D_POLL_S:-60} 秒看一次"
  while [ -n "$(gpu_pids)" ]; do
    [ $((SECONDS - t0)) -lt "$GPU_WAIT_S" ] || stop "GPU 被别的进程占了 ${GPU_WAIT_S} 秒，停下"
    sleep "${V2D_POLL_S:-60}"
  done
  say "GPU 空了"
}
wait_gpu
fc() {  # fc <record 名> <GPU 0|1> <容器名> <命令>
  if [ -n "$DIAG_SRC" ]; then
    V2D_FC_SRC="$DIAG_SRC" V2D_GPU="$2" V2D_CONTAINER_NAME="$3" scripts/dev/record "diag2/$1" scripts/dev/in-flash-chord "$4" >> "$LOG" 2>&1 < /dev/null
  else
    V2D_GPU="$2" V2D_CONTAINER_NAME="$3" scripts/dev/record "diag2/$1" scripts/dev/in-flash-chord "$4" >> "$LOG" 2>&1 < /dev/null
  fi
}
done_n=0; gate_n=0; ctl_n=0; err_n=0; first=1
while IFS=$'\t' read -r run step role _; do
  case "$run" in ''|'#'*) continue ;; esac
  P="runs/flash_chord/$run/policy_$step.safetensors"; E="runs/flash_chord/$run/eval_${step}_w4096.json"; D="runs/diag2/$run"
  mline="$(grep -a -m1 "in-flash-chord: /workspace ← " "runs/logs/$run.eval_${step}_w4096.log" 2>/dev/null)"
  if [ -z "$DIAG_SRC" ] && [ -n "$mline" ]; then
    err_n=$((err_n + 1)); say "${run}：本机那份 4096 世界评测挂的是源码副本（日志里有挂载说明行），不是默认的 v0.4.0，同卡比对不成立，跳过"; continue
  fi
  if [ -n "$DIAG_SRC" ] && [[ $mline != *"/workspace ← ${DIAG_SRC}（"* ]]; then
    err_n=$((err_n + 1)); say "${run}：本机那份 4096 世界评测挂的不是 ${DIAG_SRC}（挂载行：${mline:-没有}），同卡比对不成立，跳过"; continue
  fi
  if [ ! -f "$P" ] || [ ! -f "$E" ] || [ ! -f "runs/logs/$run.eval_${step}_w4096.log" ]; then
    err_n=$((err_n + 1)); say "${run}：缺 policy、本机的 4096 世界评测 JSON 或它的本机日志，跳过"; continue
  fi
  mkdir -p "$D"
  if [ "$first" = 1 ]; then
    first=0
    wait_gpu
    say "${run}：试跑 16 个世界（--palms）"
    fc "smoke_$run" 1 "diag2-smoke-$run" "cd /workspace && python /repo/stages/evaluate/diag_rollout.py /repo/$P /repo/$D/smoke_w16 --worlds 16 --palms" \
      || stop "${run}：试跑失败，见 results/diag2/smoke_${run}.txt.failed"
    fc "smoke_sync_$run" 0 "diag2-sync-$run" "python /repo/stages/evaluate/diag2_sync.py /repo/$D/smoke_w16.npz" \
      || stop "${run}：试跑的同步闸不过，见 results/diag2/smoke_sync_${run}.txt.failed"
    rm -rf "$D/smoke_w16_hydra"
    say "${run}：试跑通过（$(grep -h '同步最大差' "results/diag2/smoke_sync_$run.txt" | head -1)）"
  fi
  wait_gpu
  say "${run}：诊断 rollout（4096 世界，--palms）"
  if ! fc "rollout_$run" 1 "diag2-$run" "cd /workspace && python /repo/stages/evaluate/diag_rollout.py /repo/$P /repo/$D/rollout_w4096 --worlds 4096 --palms"; then
    err_n=$((err_n + 1)); say "${run}：诊断 rollout 失败，见 results/diag2/rollout_${run}.txt.failed"; rm -rf "$D/rollout_w4096_hydra"; continue
  fi
  rm -rf "$D/rollout_w4096_hydra"
  say "${run}：分析（diag_phase.py --role ${role}）"
  rc=0
  fc "phase_$run" 0 "diag2-phase-$run" "python /repo/stages/evaluate/diag_phase.py /repo/$D/rollout_w4096.npz --diag-json /repo/$D/rollout_w4096.json --same-card-json /repo/$E --role $role" || rc=$?
  case "$rc" in
    0) done_n=$((done_n + 1)); say "${run}：完成，$(grep -h '主要失败环节' "results/diag2/phase_$run.txt" | sed 's/^ *//' | head -1)" ;;
    3) gate_n=$((gate_n + 1)); say "${run}：分析的闸没过，只有数字，见 results/diag2/phase_${run}.txt.failed" ;;
    4) ctl_n=$((ctl_n + 1)); say "${run}：对照不过（CONTROL_FAIL），分类门限不可信，见 results/diag2/phase_${run}.txt.failed" ;;
    *) err_n=$((err_n + 1)); say "${run}：分析出错（退出码 ${rc}），见 results/diag2/phase_${run}.txt.failed" ;;
  esac
done < "$TABLE"
say "DIAG2_DONE（完成 ${done_n}、闸不过 ${gate_n}、对照不过 ${ctl_n}、出错 ${err_n}）"
