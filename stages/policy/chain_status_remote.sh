# 训练机上的只读状态快照：批量链各段的训练与评测日志要点、各 run 的看守结局、已完成正式 run 的 W&B 摘要、正在训练的 run、GPU、磁盘、长链标记。
# 不改任何文件、不起容器。Mac 上经 stdin 送过去执行，批量训练期间不用同步仓库：
#   scripts/dev/record t2/status_mf4090_2000 ssh mf4090 "cd <仓库> && bash -s -- mf4090 t2 t2a t2b t2c" < stages/policy/chain_status_remote.sh
# 参数：<机器标签> <run 名前缀，如 t2> <段名…>。日志名约定 runs/logs/<段名>_{train,eval}_<机器标签>.log。
set -uo pipefail
HOST=${1:?机器标签}; PFX=${2:?run 名前缀}; shift 2
echo "== $(hostname) $(date '+%F %T %Z')；来历 $(cat .v2d-source-commit 2>/dev/null || echo unknown)"
for seg in "$@"; do
  for kind in train eval; do
    f="runs/logs/${seg}_${kind}_${HOST}.log"
    echo "## ${seg} ${kind}（${f}）"
    if [ -f "$f" ]; then grep -aE '^===|fc_run_watched:|SMOKE|^chain_after' "$f" | tail -n 40; else echo "（没有这个日志）"; fi
  done
done
echo "## 看守日志：run、含非有限的扫描行数、最后的 NAN_WATCH 结局、扫描份数"
for f in runs/logs/${PFX}-ep*.nanwatch.log; do
  [ -f "$f" ] || continue
  r="$(basename "$f" .nanwatch.log)"
  printf '%s 非有限行 %s 结局 %s 扫描 %s\n' "$r" "$(grep -acE '非有限 [1-9]' "$f")" \
    "$( (grep -aoE 'NAN_WATCH_[A-Z]+' "$f" | tail -1) || true)" "$(grep -ac '步：非有限' "$f")"
done
echo "## 已完成正式 run 的 W&B 摘要（train_nan_scan.py --log，最后一段日志）"
for d in runs/flash_chord/${PFX}-ep*-"$HOST"; do
  [ -f "$d/DONE" ] || continue
  r="$(basename "$d")"; s="$(ls runs/logs/"$r".s*.log 2>/dev/null | sort -V | tail -1)"
  [ -n "$s" ] && python3 stages/policy/train_nan_scan.py --log "$s"
done
echo "## 没有 DONE 的正式 run：最新完整 policy 的步数与写入时间"
for d in runs/flash_chord/${PFX}-ep*-"$HOST"; do
  [ -d "$d" ] && [ ! -f "$d/DONE" ] || continue
  p="$(ls "$d"/policy_*.safetensors 2>/dev/null | sed -E 's/.*policy_([0-9]+)\.safetensors/\1/' | sort -n | tail -1)"
  echo "$(basename "$d")：${p:-（还没有 policy）}$( [ -n "$p" ] && date -r "$d/policy_$p.safetensors" '+ 写于 %F %T %Z')$( [ -f "$d/NONFINITE" ] && echo '；有 NONFINITE')"
done
echo "## 机器"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
df -Pk . | awk 'NR == 2 {print int($4 / 1024), "MB free"}'
ls runs/.chain-active-* 2>/dev/null || echo "（没有长链标记）"
