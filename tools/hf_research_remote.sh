#!/usr/bin/env bash
# 在一台机器上后台启动 tools/hf_research_run.sh（Mac 上、仓库根执行）。计划 agent-doc/plan-hf-research-archive-2026-11-04.md。
# 凭据从 .env（scripts/dev/_load-env）和 ~/.kaggle/access_token 读，经 ssh 的 stdin 传过去，不进命令行、不进日志。
# 已知凭据原值：HF_TOKEN、WANDB_API_KEY、V2D_PROXY 整串、其中的“用户名:密码”与密码（不短于 8 个字符时）、Kaggle token。
# 用法：tools/hf_research_remote.sh <ssh 主机> <机器名> <工作目录> <python> <步骤> -- <plan 参数...>
#   主机为 local 时在 Mac 本机跑（不经 ssh）。日志写在 <工作目录>/run_<步骤>.log。
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
H=${1:?主机}; M=${2:?机器名}; W=${3:?工作目录}; PY=${4:?python}; STEPS=${5:?步骤}; shift 5
. "$ROOT/scripts/dev/_load-env" >/dev/null 2>&1
[ -f "$ROOT/.env.members" ] && . "$ROOT/.env.members"   # V2D_MEMBERS：其他成员的用户名，不进 git
[ -n "${V2D_MEMBERS:-}" ] || { echo "缺 .env.members 里的 V2D_MEMBERS" >&2; exit 2; }
[ -n "${HF_TOKEN:-}" ] && [ -n "${V2D_PROXY:-}" ] && [ -n "${WANDB_API_KEY:-}" ] || { echo "缺 HF_TOKEN、V2D_PROXY 或 WANDB_API_KEY" >&2; exit 2; }
creds() {
  local u="${V2D_PROXY#*://}"; u="${u%%@*}"
  if [ "$H" = local ]; then printf '%s\ndirect\n' "$HF_TOKEN"; else printf '%s\n%s\n' "$HF_TOKEN" "$V2D_PROXY"; fi
  printf '%s\n' "$V2D_MEMBERS"
  printf '%s\n' "$HF_TOKEN" "$WANDB_API_KEY" "$V2D_PROXY"
  case "$V2D_PROXY" in *@*) local p="${u#*:}"; printf '%s\n' "$u"; [ ${#p} -ge 8 ] && printf '%s\n' "$p" ;; esac
  [ -s "$HOME/.kaggle/access_token" ] && head -1 "$HOME/.kaggle/access_token"
  echo END
}
q() { printf '%q ' "$@"; }
ARGS="$(q "$M" "$W" "$PY" "$STEPS" "$@")"  # 已按 shell 转义，远端命令行与本机 eval 都按原样拆分
if [ "$H" = local ]; then
  mkdir -p "$W/tools" && cp "$ROOT/tools/hf_research_archive.py" "$ROOT/tools/hf_research_run.sh" "$W/tools/"
  creds | eval "bash $(q "$W/tools/hf_research_run.sh") $ARGS" >> "$W/run_$STEPS.log" 2>&1
else
  ssh -o BatchMode=yes "$H" "mkdir -p $(printf %q "$W/tools")" || exit 1
  scp -q "$ROOT/tools/hf_research_archive.py" "$ROOT/tools/hf_research_run.sh" "$H:$W/tools/" || exit 1
  creds | ssh -o BatchMode=yes "$H" "cd $(q "$W") && bash tools/hf_research_run.sh $ARGS >> run_$STEPS.log 2>&1; tail -1 run_$STEPS.log"
fi
echo "日志：$H:$W/run_$STEPS.log"
