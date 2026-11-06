#!/usr/bin/env bash
# 在训练机上带 Hugging Face 凭据跑一条命令（Mac 上、仓库根执行）。HF_TOKEN 与 V2D_PROXY 从 .env 读（scripts/dev/_load-env），
# 经 ssh 的 stdin 传过去；远端读进来后 export 成 HF_TOKEN、https_proxy、HTTPS_PROXY，HF_ENDPOINT 固定为 https://huggingface.co
# （服务器 .env 里的 HF_ENDPOINT 指向下载镜像，带 token 的请求不能经它转发）。凭据不进命令行、不进日志（CLAUDE.md 安全一节）。
# 输出里含 @ 或 token 样式的行一律滤掉（代理地址带用户名密码）。主机按 V2D_ENV 选（同 scripts/dev/remote），命令在远端仓库副本根目录执行。
# 远端 PATH 前面加 ~/.local/bin：hf 命令装在那里，ssh 的非登录 shell 默认找不到它。
# 10/1 归档时是手敲的同一套做法，这里固定成脚本。
# 用法：tools/hf_remote.sh '<远端命令>'
#   例：tools/hf_remote.sh 'python3 runs/tools/hf_archive.py whoami --repo Rongxuan-Zhou/v2d-track2-archive'
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd -P)"
. "$ROOT/scripts/dev/_load-env" >/dev/null
CMD=${1:?用法：tools/hf_remote.sh '<远端命令>'}
[ -n "${HF_TOKEN:-}" ] && [ -n "${V2D_PROXY:-}" ] || { echo "hf_remote: .env 里缺 HF_TOKEN 或 V2D_PROXY" >&2; exit 2; }
printf '%s\n%s\n' "$HF_TOKEN" "$V2D_PROXY" | ssh -o BatchMode=yes -o ConnectTimeout=20 "$V2D_HOST" \
  "cd '$V2D_REMOTE' && read -r T && read -r P && [ -n \"\$T\" ] && [ -n \"\$P\" ] || { echo 没读到凭据; exit 1; }
export HF_TOKEN=\"\$T\" https_proxy=\"\$P\" HTTPS_PROXY=\"\$P\" HF_ENDPOINT=https://huggingface.co HF_HUB_DISABLE_PROGRESS_BARS=1 PATH=\"\$HOME/.local/bin:\$PATH\"
unset T P
$CMD" 2>&1 | { grep -v -E '@|hf_[A-Za-z0-9]{20}' || true; }
