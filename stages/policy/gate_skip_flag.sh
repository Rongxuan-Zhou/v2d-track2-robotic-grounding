#!/usr/bin/env bash
# 跳过闸（给 arm_after.sh 的 V2D_PRE_CMD 用）：最多等 <分钟> 分钟，期间出现 <标记文件> 就不放行（退出码 1），一直没有就放行（退出码 0）。
# 计划 agent-doc/plan-c10-extend-2026-10-16.md 第 3 版：mf4090 今夜先起 Tier 2 的 t2m，t3ps 挪到 t2m 后面；先起 t2m 要 Mac 在 t3pk 结束时在线做整体同步。
# Mac 在线时由它放标记，这次 t3ps 不起（之后挂在 t2m 后面）；Mac 断线时没有标记，等满时间照旧起 t3ps，卡不空着。
# 用法：bash stages/policy/gate_skip_flag.sh <标记文件> <分钟>；V2D_GATE_POLL_S（默认 60 秒）。最后一行 GATE_SKIP_STOP 或 GATE_SKIP_PASS。
# 测试：stages/policy/test_gate_skip_flag.py。
set -uo pipefail
F=${1:?标记文件}; M=${2:?分钟}
[[ $M =~ ^[0-9]+$ ]] || { echo "gate_skip_flag: 分钟要是非负整数：${M}" >&2; exit 2; }
S=${V2D_GATE_POLL_S:-60}
end=$(( $(date +%s) + M * 60 ))
while :; do
  if [ -e "$F" ]; then echo "GATE_SKIP_STOP 有 ${F}"; exit 1; fi
  [ "$(date +%s)" -lt "$end" ] || break
  sleep "$S"
done
echo "GATE_SKIP_PASS 等了 ${M} 分钟没有 ${F}"; exit 0
