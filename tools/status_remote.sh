#!/usr/bin/env bash
# 训练机上的只读状态快照（经 stdin 送过去执行，不改任何文件）：时间、GPU、仓库所在盘、本用户的 tmux 会话、长链标记，
# 近 24 小时改过的 arm_after 日志与最近改动的 8 份链日志各自的最后一行。不打印主机名（mf4090 的主机名里有别人的用户名），机器名由参数给。
# 用法（Mac，仓库根）：scripts/dev/record ops/status_<机器>_<时间> ssh <机器> 'bash -s <机器>' < tools/status_remote.sh
set -u   # 不开 pipefail：下面 ls | head 会让 ls 收到 SIGPIPE，管道整体算失败
M=${1:-?}
if [ -d ~/v2d/maniphys_lab_v2d_challenge ]; then cd ~/v2d/maniphys_lab_v2d_challenge; else cd /lab_data/rxzhou/v2d/maniphys_lab_v2d_challenge || exit 1; fi
echo "== $M $(date '+%F %T %Z')"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
df -h . | tail -1
echo "-- tmux"; tmux ls 2>/dev/null || echo "（没有会话）"
echo "-- 长链标记"; ls runs/.chain-active-* 2>/dev/null || echo "（没有）"
echo "-- arm_after 日志（近 24 小时改过的）"
find runs/logs -maxdepth 1 -name 'arm_after_*.log' -mmin -1440 -printf '%T@ %p\n' 2>/dev/null | sort -n | cut -d' ' -f2 | while read -r f; do
  echo "$f：$(tail -n 1 "$f" | cut -c1-200)"
done
echo "-- 最近改动的 8 份链日志"
ls -t runs/logs/*.log 2>/dev/null | grep -vE '/arm_after_' | head -8 | while read -r f; do
  echo "$f：$(tail -n 1 "$f" | cut -c1-200)"
done
