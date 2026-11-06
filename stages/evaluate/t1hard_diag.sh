#!/usr/bin/env bash
# Tier 1 五个硬集的手部诊断（agent-doc/experiments/2026-10-30-t1hard-diag.md 第 2 件事；训练机仓库根执行，占 GPU 几分钟一集；只读 checkpoint，不训练）。
# 每集：diag_rollout.py --palms --fingers（--worlds 默认 1024）→ diag_onset.py（失败世界分"参考无接触时离开 / 没贴住 / 贴住后掉"）→ finger_facts.py（手指合了多少）。
# 用法：bash stages/evaluate/t1hard_diag.sh [世界数]   日志 runs/logs/t1hard_diag.log，最后一行 T1HARD_DIAG_DONE（成功 n、失败 m）
set -u
W=${1:-1024}
LOG=runs/logs/t1hard_diag.log
mkdir -p runs/logs runs/diag_t1hard
ok=0; bad=0
echo "=== $(date '+%F %T %Z') 开始，世界数 $W" >> "$LOG"
for e in 25 9 17 1 4; do
  run=t1-ep$e-w2048-cap10m-r3a-ml40
  pol=/repo/runs/flash_chord/$run/policy_349999104.safetensors
  out=/repo/runs/diag_t1hard/${run}_w$W
  if V2D_GPU=1 V2D_CONTAINER_NAME="fc-diag-t1hard-ep$e" scripts/dev/record t1hard/diag_rollout_ep${e} scripts/dev/in-flash-chord \
       "python /repo/stages/evaluate/diag_rollout.py $pol $out --worlds $W --palms --fingers" >> "$LOG" 2>&1 \
     && scripts/dev/record t1hard/onset_ep${e} scripts/dev/in-flash-chord "python /repo/stages/evaluate/diag_onset.py $out.npz" >> "$LOG" 2>&1 \
     && scripts/dev/record t1hard/finger_facts_ep${e} scripts/dev/in-flash-chord "python /repo/stages/evaluate/finger_facts.py $out.npz" >> "$LOG" 2>&1; then
    ok=$((ok + 1)); echo "=== $(date '+%F %T %Z') ep$e 完成" >> "$LOG"
  else
    bad=$((bad + 1)); echo "=== $(date '+%F %T %Z') ep$e 失败" >> "$LOG"
  fi
done
echo "=== $(date '+%F %T %Z') T1HARD_DIAG_DONE（成功 $ok、失败 $bad）" >> "$LOG"
