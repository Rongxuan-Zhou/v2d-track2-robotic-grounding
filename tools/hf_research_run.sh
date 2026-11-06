#!/usr/bin/env bash
# 赛后研究归档在一台机器上的完整流程（机器上执行，由 tools/hf_research_remote.sh 经 ssh 启动，setsid 后台跑）。
# 计划 agent-doc/plan-hf-research-archive-2026-11-04.md。只传不删：本脚本不删除、不改动任何源文件，只写工作目录 $W。
# stdin 依次给：HF_TOKEN、代理地址（Mac 上为 direct）、成员用户名（逗号分隔）、已知凭据原值若干行，最后一行 END。读完立刻关掉 stdin；凭据只放在这个进程的变量里，
# 经管道交给 python，不进命令行、不进日志、不落盘。
# 用法：bash hf_research_run.sh <机器名> <工作目录> <python> <步骤 prepare|upload|all> -- <plan 的 --src/--pack 参数...>
# 步骤：prepare = plan、scan、复扫（必须 0 个新挡下）、stage，停下等核对；upload = 上传公开与私有、verify、spotcheck；all = 两段连着跑。
# 最后一行 HF_RESEARCH_PREPARED、HF_RESEARCH_DONE <机器> 或 HF_RESEARCH_STOPPED <哪一步>。
set -uo pipefail
M=${1:?机器名}; W=${2:?工作目录}; PY=${3:?python}; STEPS=${4:?步骤}; shift 4
[ "${1:-}" = -- ] && shift
PUB=Rongxuan-Zhou/v2d-track2-research-archive
PRI=Rongxuan-Zhou/v2d-track2-research-code
T="$W/tools/hf_research_archive.py"
if [ "${V2D_RS_CHILD:-}" != 1 ]; then
  # 前台：读完 stdin 就把自己放到后台（setsid，断开 ssh 也不受影响），凭据经环境变量交给子进程（只有本用户和 root 能读）
  read -r TOK; read -r PROXY; read -r MEMBERS
  S=""; while IFS= read -r x; do [ "$x" = END ] && break; S+="$x"$'\n'; done
  exec 0</dev/null
  [ -n "$TOK" ] && [ -n "$PROXY" ] && [ -n "$S" ] || { echo "没读到凭据"; echo "HF_RESEARCH_STOPPED stdin"; exit 1; }
  export HF_TOKEN="$TOK" V2D_RS_SECRETS="$S" V2D_RS_MEMBERS="$MEMBERS" V2D_RS_CHILD=1
  [ "$PROXY" = direct ] || export https_proxy="$PROXY" HTTPS_PROXY="$PROXY"  # Mac 直连 HF，代理只在训练机上用
  unset TOK PROXY S
  if command -v setsid >/dev/null; then setsid nohup bash "$0" "$M" "$W" "$PY" "$STEPS" -- "$@" < /dev/null & else
    nohup bash "$0" "$M" "$W" "$PY" "$STEPS" -- "$@" < /dev/null & fi
  echo "已转到后台（pid $!）"; exit 0
fi
S="$V2D_RS_SECRETS"; MEMBERS="$V2D_RS_MEMBERS"; unset V2D_RS_SECRETS
export HF_ENDPOINT=https://huggingface.co HF_HUB_DISABLE_PROGRESS_BARS=1
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0 HF_HOME="$W/hf_home" PYTHONDONTWRITEBYTECODE=1
MAXPRI=2.5; [ -s "$W/max_private_gb" ] && MAXPRI="$(cat "$W/max_private_gb")"  # 默认每台 2.5 GB；某台要放宽就在工作目录写这个文件
ts() { date '+%F %T %Z'; }
stop() { echo "=== $(ts) HF_RESEARCH_STOPPED $1"; exit 1; }
secrets() { printf '%s' "$S"; }
echo "=== $(ts) $M 开始（步骤 ${STEPS}）；工具 md5 $( (md5sum "$T" 2>/dev/null || md5 -r "$T") | cut -d' ' -f1)；可用 $(df -Pk "$W" | awk 'NR==2{print int($4/1048576)}') GB"
case "$STEPS" in prepare|all)
  REUSE=(); [ -s "$W/manifest.prev.jsonl" ] && REUSE=(--reuse "$W/manifest.prev.jsonl")
  "$PY" "$T" plan --machine "$M" --work "$W" ${REUSE[@]+"${REUSE[@]}"} "$@" || stop plan  # bash 3.2 在 set -u 下展开空数组会报错
  secrets | "$PY" "$T" scan --fast --work "$W" --members "$MEMBERS" --max-private-gb "$MAXPRI" || stop scan
  out="$(secrets | "$PY" "$T" scan --fast --work "$W" --members "$MEMBERS" --max-private-gb "$MAXPRI")" || stop rescan
  echo "$out"
  [ "$(grep '^SCAN_DONE ' <<< "$out" | awk '{print $4}')" = 0 ] || stop "rescan（复扫又挡下文件）"
  secrets | "$PY" "$T" stage --work "$W" --members "$MEMBERS" --max-private-gb "$MAXPRI" || stop stage
  [ "$STEPS" = prepare ] && { echo "=== $(ts) HF_RESEARCH_PREPARED"; exit 0; } ;;
esac
case "$STEPS" in upload|all)
  for v in public private; do
    repo=$PUB; [ $v = private ] && repo=$PRI
    "$PY" "$T" upload --work "$W" --repo "$repo" --vis $v || stop "upload $v"
  done
  for v in public private; do
    repo=$PUB; [ $v = private ] && repo=$PRI
    "$PY" "$T" verify --work "$W" --repo "$repo" --vis $v || stop "verify $v"
    "$PY" "$T" spotcheck --work "$W" --repo "$repo" --vis $v || stop "spotcheck $v"
  done
  echo "=== $(ts) HF_RESEARCH_DONE $M" ;;
  prepare) ;;
  *) stop "步骤只能是 prepare、upload、all" ;;
esac
