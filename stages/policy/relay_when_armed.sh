#!/usr/bin/env bash
# 等前一条接力起链之后再挂下一条接力（计划 agent-doc/plan-t1r3o-2026-10-31.md 第 3 版）。
# arm_after.sh 等待期间看到仓库副本的来历变了就停；前一条接力带 V2D_AFTER_SYNC_FROM 时会改来历，所以下一条要等它同步、起链之后才挂。
# 每 V2D_POLL_S 秒（默认 60）看一次 <前一条接力的日志> 的最后一行：是 arm_after.sh 的结束行 “=== <时间> ARM_AFTER_ARMED” 就 exec 后面的命令；ARM_AFTER_STOPPED、
# 或 arm_after.sh 参数错时的 "arm_after:" 报错行就退出 1；
# 超过 V2D_RELAY_MAX_POLLS 次（默认 720，约 12 小时）退出 3。在仓库根跑；后面的命令照原样 exec（环境变量由调用方给）。
# 用法：bash runs/tools/relay_when_armed.sh <前一条接力的日志> <命令> [参数 ...]
set -u
[ $# -ge 2 ] || { echo "用法：relay_when_armed.sh <前一条接力的日志> <命令> [参数 ...]" >&2; exit 2; }
LOG=$1; shift
POLL=${V2D_POLL_S:-60}; MAXP=${V2D_RELAY_MAX_POLLS:-720}
say() { echo "=== $(date '+%F %T %Z') $*"; }
say "等 ${LOG} 的最后一行变成 ARM_AFTER_ARMED，再跑：$*"
i=0
while :; do
  last="$(tail -n 1 "$LOG" 2>/dev/null || true)"
  # 只认 arm_after.sh 自己写的结束行“=== <时间> ARM_AFTER_ARMED”：前一条若也是本脚本，它的等待行与“已起链”行里都带这个词（11/03 final1 的教训）
  if [[ $last =~ ^===\ [0-9-]+\ [0-9:]+\ [A-Z]+\ ARM_AFTER_ARMED$ ]]; then say "前一条接力已起链：${last}"; exec "$@"; fi
  case "$last" in
    *ARM_AFTER_STOPPED*|arm_after:*) say "前一条接力停了，不挂：${last}"; exit 1 ;;
  esac
  i=$((i + 1)); [ "$i" -le "$MAXP" ] || { say "等了 ${MAXP} 次还没起链，退出"; exit 3; }
  sleep "$POLL"
done
