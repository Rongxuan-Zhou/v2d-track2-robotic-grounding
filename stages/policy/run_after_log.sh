#!/usr/bin/env bash
# 等某个链日志的最后一行匹配结束行，再经 scripts/dev/record 跑一条命令（训练机上、仓库根执行，setsid 后台跑；Mac 断网不影响）。
# 用途：评测链结束后自动跑判读（如 r3o_verdict.py），不靠 Mac 轮询（用户 10/16：Mac 随时可能断网）。
# 链在跑时 scripts/dev/remote 拒绝同步，本脚本可经 scripts/dev/push-new-files 放上去。
# 用法：V2D_POLL_S=60 V2D_MAX_POLLS=720 setsid nohup stages/policy/run_after_log.sh <日志> <结束行的正则> <记录名> <命令...> \
#         > runs/logs/after_<名>.log 2>&1 < /dev/null &
#   每 V2D_POLL_S 秒（默认 60）看一次日志最后一行（grep -E）；匹配就跑 scripts/dev/record <记录名> <命令...>，最后一行 RUN_AFTER_DONE <退出码>，
#   本脚本的退出码就是 record 的退出码；起来时日志文件还不存在就先打一行警告（复核 NIT 4、5）；
#   等满 V2D_MAX_POLLS 次（默认 720，约 12 小时）还没匹配就打 RUN_AFTER_TIMEOUT，不跑命令，退出码 1。
# 测试：stages/policy/test_run_after_log.py。
set -uo pipefail
cd "$(dirname "$0")/../.."
LOG=${1:?用法见文件头}; PAT=${2:?用法见文件头}; NAME=${3:?用法见文件头}; shift 3
[ $# -gt 0 ] || { echo "run_after_log: 没有要跑的命令" >&2; exit 2; }
[[ $NAME =~ ^[A-Za-z0-9_/.-]+$ && $NAME != *..* && $NAME != /* ]] || { echo "run_after_log: 记录名非法：$NAME" >&2; exit 2; }
POLL=${V2D_POLL_S:-60}; MAX=${V2D_MAX_POLLS:-720}
[[ $POLL =~ ^[1-9][0-9]*$ && $MAX =~ ^[1-9][0-9]*$ ]] || { echo "run_after_log: V2D_POLL_S、V2D_MAX_POLLS 要是正整数" >&2; exit 2; }
echo "=== $(date '+%F %T %Z') 等 ${LOG} 的最后一行匹配 /${PAT}/，然后记录 ${NAME}：$*"
[ -f "$LOG" ] || echo "=== $(date '+%F %T %Z') 警告：${LOG} 现在还不存在（路径写错的话会一直等到超时）"
for ((i = 0; i < MAX; i++)); do
  if [ -f "$LOG" ] && tail -n 1 "$LOG" | grep -a -q -E "$PAT"; then
    echo "=== $(date '+%F %T %Z') 匹配：$(tail -n 1 "$LOG")"
    scripts/dev/record "$NAME" "$@"; rc=$?
    echo "RUN_AFTER_DONE ${rc}"
    exit "$rc"
  fi
  sleep "$POLL"
done
echo "RUN_AFTER_TIMEOUT"
exit 1
