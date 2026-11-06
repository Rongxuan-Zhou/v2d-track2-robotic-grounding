#!/usr/bin/env bash
# 12 个 Track 2 Kaggle 榜各列出本队的提交记录（只读，Mac 上执行），用来数每个榜最近 7 天用了几次额度。
# 额度原文见 results/ops/kaggle_rules_submission_1014.txt 第 13–14 行。凭据由 kaggle CLI 自己读 ~/.kaggle/access_token。
# 用法：scripts/dev/record ops/submission_history_<MMDD_HHMM>（Kaggle 的时间是 UTC） bash tools/submission_history.sh
set -uo pipefail
K=${V2D_KAGGLE:-$HOME/.cache/uv/archive-v0/uvfKxZAYaAMiPZYi/bin/kaggle}
[ -x "$K" ] || K=$(command -v kaggle 2>/dev/null || true)
if [ -z "$K" ] || [ ! -x "$K" ]; then echo "submission_history: 找不到 kaggle CLI，用 V2D_KAGGLE 指定" >&2; exit 2; fi
rc=0
for t in 1 2 3; do
  for m in auc sp-sr mp-sr mppe; do
    s="v2d-challenge-track2-tier${t}-${m}"
    echo "== ${s}"
    out=$(perl -e 'alarm 90; exec @ARGV' "$K" competitions submissions "$s" --csv 2>&1); r=$?
    echo "$out" | grep -v '^Warning:' | python3 -c 'import csv,sys
for r in csv.DictReader(sys.stdin): print(r["date"][:19], r["fileName"], r["status"].split(".")[-1], r["publicScore"], sep="\t")'
    [ "$r" -eq 0 ] || rc=1
  done
done
echo "SUBMISSION_HISTORY_DONE rc=$rc"
exit "$rc"
