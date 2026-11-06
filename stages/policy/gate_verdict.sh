#!/usr/bin/env bash
# 判读闸（给 arm_after.sh 的 V2D_PRE_CMD 用）：等判读记录出现，最后一行是允许的总判才放行（退出码 0），否则退出码 1。
# 计划 agent-doc/plan-c10-extend-2026-10-16.md 路 A（独立复核 MINOR 3）：只看最后一行、整行匹配（r3o_verdict.py 的说明行里也写着“有用”“看不清”，
# 整文件 grep 会在“没用”时也放行）；判读失败时 scripts/dev/record 落成 <记录>.failed，见到就立刻不放行，不白等。
# 写入侧没有竞态：record 先写 .partial 再改名。
# 用法：bash stages/policy/gate_verdict.sh <判读记录> <允许的总判，竖线分隔，如 有用|看不清>
#   V2D_GATE_POLLS（默认 60 次）、V2D_GATE_POLL_S（默认 30 秒）
# 最后一行 GATE_VERDICT_PASS <总判> 或 GATE_VERDICT_STOP <原因>。测试：stages/policy/test_gate_verdict.py。
set -uo pipefail
F=${1:?判读记录}; OK=${2:?允许的总判}
[[ $OK =~ ^[^[:space:]\'\"\;\&\$\`\(\)]+$ ]] || { echo "gate_verdict: 允许的总判写法非法：${OK}" >&2; exit 2; }
N=${V2D_GATE_POLLS:-60}; S=${V2D_GATE_POLL_S:-30}
i=0
while [ ! -f "$F" ]; do
  if [ -e "$F.failed" ]; then echo "GATE_VERDICT_STOP 判读失败（${F}.failed）"; exit 1; fi
  i=$((i + 1))
  if [ "$i" -gt "$N" ]; then echo "GATE_VERDICT_STOP 等了 ${N} 次还没有 ${F}"; exit 1; fi
  sleep "$S"
done
v="$(tail -1 "$F")"
if printf '%s\n' "$v" | grep -qxE "R3O_VERDICT (${OK})"; then echo "GATE_VERDICT_PASS ${v#R3O_VERDICT }"; exit 0; fi
echo "GATE_VERDICT_STOP 总判是：${v:-空}"; exit 1
