#!/usr/bin/env bash
# 开链前的环境核对（训练机上、仓库根执行；不占 GPU、不起容器）。计划 agent-doc/plan-t2-batch-2026-10-01.md 第 7 节第 3 步。
# 逐项打印、逐项判：
#   来历 .v2d-source-commit 干净（不带 -dirty / -syncing / unknown）；GPU 上没有进程；没有 runs/.chain-active-*；环境里没有 V2D_FC_SRC；
#   flash_chord:latest 的镜像指纹（runs/cache/image-<ID>.json 里 fc_run.sh 缓存的 base11 / freeze）等于第 1 批的值；
#   third_party/video_to_data（挂成容器 /workspace 的上游源码）工作树干净、HEAD 等于期望、scene/support.py 的 md5 等于期望。
#   另报：磁盘可用空间、镜像 ID、上游 HEAD。
# 用法：stages/policy/chain_preflight.sh [期望的上游短哈希，默认 33129dd] [期望的 support.py md5，默认 95e5e33f6f635310817130fe22477307]
# 全过退出码 0，最后一行 PREFLIGHT_OK；否则退出码 1，最后一行 PREFLIGHT_FAIL 与不过的项数。
set -uo pipefail
cd "$(dirname "$0")/../.."
UP_WANT=${1:-33129dd}; SUP_WANT=${2:-95e5e33f6f635310817130fe22477307}
B11_WANT=e2934912858820b8; FRZ_WANT=df0cf35711a68fb3   # 同 day2_train_chain.sh：两台机器 Day 1 起的镜像指纹
bad=0
ok() { echo "  ok   $*"; }
no() { echo "  FAIL $*"; bad=$((bad + 1)); }
echo "== $(hostname) $(date '+%F %T %Z')"
src="$(cat .v2d-source-commit 2>/dev/null || echo unknown)"
case "$src" in *-dirty*|*-syncing*|unknown) no "来历 $src" ;; *) ok "来历 $src" ;; esac
apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null)"
[ -z "$apps" ] && ok "GPU 上没有进程" || no "GPU 上有进程：$apps"
marks="$(ls runs/.chain-active-* 2>/dev/null)"
[ -z "$marks" ] && ok "没有 runs/.chain-active-*" || no "有长链标记：$marks"
[ -z "${V2D_FC_SRC:-}" ] && ok "环境里没有 V2D_FC_SRC" || no "环境里有 V2D_FC_SRC=${V2D_FC_SRC}"
echo "  info 磁盘可用 $(df -Pk . | awk 'NR == 2 {print int($4 / 1024)}') MB"
img="$(docker image inspect flash_chord:latest --format '{{.Id}}' 2>/dev/null)"
if [ -z "$img" ]; then no "没有 flash_chord:latest"
else
  echo "  info 镜像 $img"
  c="runs/cache/image-${img#sha256:}.json"
  if [ -s "$c" ]; then
    read -r b11 frz <<< "$(python3 -c 'import json, sys; d = json.load(open(sys.argv[1])); print(d["base11_sha"], d["freeze_sha"])' "$c")"
    [ "$b11" = "$B11_WANT" ] && [ "$frz" = "$FRZ_WANT" ] && ok "镜像指纹 $b11 / $frz" || no "镜像指纹 $b11 / ${frz}，应为 $B11_WANT / $FRZ_WANT"
  else
    no "没有 ${c}（这个镜像 ID 没在本机训练过）"
  fi
fi
UP=third_party/video_to_data
head="$(git -C "$UP" rev-parse --short=7 HEAD 2>/dev/null || echo none)"
[ "$head" = "$UP_WANT" ] && ok "上游 HEAD $head" || no "上游 HEAD ${head}，应为 $UP_WANT"
dirty="$(git -C "$UP" status --porcelain --untracked-files=no 2>/dev/null | head -5)"
[ -z "$dirty" ] && ok "上游工作树（已跟踪的文件）没有改动" || no "上游工作树有改动：$(tr '\n' ' ' <<< "$dirty")"
sp="$UP/robotic_grounding/flash_chord/src/flash_chord/scene/support.py"
sup="$( (md5sum "$sp" 2>/dev/null || echo none) | cut -d' ' -f1)"
[ "$sup" = "$SUP_WANT" ] && ok "support.py md5 $sup" || no "support.py md5 ${sup}，应为 $SUP_WANT"
if [ "$bad" = 0 ]; then echo "PREFLIGHT_OK"; else echo "PREFLIGHT_FAIL $bad"; exit 1; fi
