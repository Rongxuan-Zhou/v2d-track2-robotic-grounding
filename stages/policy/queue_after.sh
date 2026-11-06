#!/usr/bin/env bash
# 等另一条训练链结束（它的 runs/.chain-active-<名字> 标记消失）、卡上没有进程后，再跑一个队列（训练机上，放 tmux 里）。
# 链在跑时 scripts/dev/remote 拒绝同步，服务器上的仓库副本里可能还没有本脚本：先复制到 runs/tools/ 再跑，复制时记下 md5。
# 用法（仓库根）：bash <本脚本> <等的标记名> <本链标记名> <队列文件> <日志>
#   例：tmux new-session -d -s v2d-diag5 -c <仓库根> 'bash runs/tools/queue_after.sh diag diag5 stages/policy/queues/diag_ml40_ext4750.txt runs/logs/diag_ml40_ext4750.log'
# 本链的标记一开始就写（内容非空），等待与运行期间 scripts/dev/remote 都拒绝同步。最多等 6 小时；卡上 5 分钟内一直有进程就不跑。
set -uo pipefail
WAIT=${1:?用法：queue_after.sh <等的标记名> <本链标记名> <队列文件> <日志>}; MINE=${2:?}; QUEUE=${3:?}; LOG=${4:?}
for n in "$WAIT" "$MINE"; do [[ $n =~ ^[A-Za-z0-9_-]+$ ]] || { echo "标记名非法：$n" >&2; exit 2; }; done
[ -f "$QUEUE" ] || { echo "缺队列 $QUEUE" >&2; exit 2; }
[ -x stages/policy/fc_queue.sh ] || { echo "不在仓库根" >&2; exit 2; }
ts() { date '+%F %T %Z'; }
MARK="runs/.chain-active-$MINE"
echo "$MINE $(cat .v2d-source-commit 2>/dev/null) $(date)" > "$MARK"
echo "=== $(ts) 等 runs/.chain-active-$WAIT 消失后跑 $QUEUE" >> "$LOG"
for _ in $(seq 360); do [ -e "runs/.chain-active-$WAIT" ] || break; sleep 60; done
if [ -e "runs/.chain-active-$WAIT" ]; then echo "=== $(ts) 等了 6 小时还没结束，不跑" >> "$LOG"; rm -f "$MARK"; exit 1; fi
for _ in $(seq 10); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 30; done
apps="$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader)"
if [ -n "$apps" ]; then echo "=== $(ts) 卡上还有进程，不跑：$apps" >> "$LOG"; rm -f "$MARK"; exit 1; fi
stages/policy/fc_queue.sh "$QUEUE" >> "$LOG" 2>&1; rc=$?
rm -f "$MARK"
echo "=== $(ts) $QUEUE 结束，退出码 $rc" >> "$LOG"
exit $rc
