#!/usr/bin/env bash
# 第 3 步配方试验的闸 G1（训练机上、仓库根执行，放 tmux；Mac 离线不影响）。计划 agent-doc/plan-step3-recipe-trial-2026-10-04.md 第 2 版第 7 节第 3 步。
#   1. fc_config_check.sh 展开两段全部 run 与冒烟的配置（变体 A 的两项与 env 下的副本应判“允许”，意外差异 0）
#   2. A 段第一条冒烟（变体 A，400 万步）用 v040_p185_nfg 副本训；之后训练链见它 DONE 就不重训
#   3. fc_8c778f9 副本按 4096 世界评这条冒烟的 policy：评测器能读变体 A 的 checkpoint，non_finite_worlds 为空，指标有限
#   4. fc_8c778f9 重评本档基线 seed 42（试验 5 条与对照 1 条；fc_export.sh，4096 世界，只写指标）
#   5. 对照的基线再用 fc_eval.sh 评一次（V2D_EVAL_TAG=rep），与第 4 步比，量同版评测器的重复噪声（只报）
# 每步经 scripts/dev/record 落到 results/r3/g1_*（拉回 Mac 后在 results/<主机>/r3/ 下）。前一步不过就停。
# 日志 runs/logs/r3_g1_<机器>.log，最后一行 R3_G1_DONE 或 R3_G1_FAILED。
# 用法：stages/policy/trial_g1.sh mf4090|ml40
set -uo pipefail
cd "$(dirname "$0")/../.."
M=${1:?用法：trial_g1.sh mf4090|ml40}
case "$M" in
  mf4090) T=1; SRC="$HOME/v2d/fc_src" ;;
  ml40)   T=2; SRC=/lab_data/rxzhou/v2d/fc_src ;;
  *) echo "机器只能是 mf4090 或 ml40" >&2; exit 2 ;;
esac
Q=stages/policy/queues; SPEC=stages/evaluate/trials/r3_t${T}_${M}.json; A=$Q/${M}_r3t${T}a; BC=$Q/${M}_r3t${T}bc
mkdir -p runs/logs; exec >> "runs/logs/r3_g1_$M.log" 2>&1
say() { echo "=== $(date '+%F %T %Z') $*"; }
die() { say "$*"; say R3_G1_FAILED; exit 1; }
say "start 闸 G1（$M，Tier $T），来历 $(cat .v2d-source-commit 2>/dev/null)"
others="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')"
[ -z "$others" ] || die "GPU 上有别的进程（${others//$'\n'/,}）"

# 1. 展开配置
scripts/dev/record "r3/g1_cfgcheck_$M" stages/policy/fc_config_check.sh "${A}_smoke.txt" "$A.txt" "${BC}_smoke.txt" "$BC.txt" \
  || die "fc_config_check 不过，见 results/r3/g1_cfgcheck_$M.txt*"
say "1. 展开配置：两段 32 份全过"

# 2. 变体 A 的冒烟（A 段第一条）
SMOKE="$(awk -F'\t' 'NR == 2 {print $2}' "$A.expect.tsv")"
[[ $SMOKE == smoke-*-r3a-* ]] || die "A 段期望表第一条的冒烟名不对：$SMOKE"
V2D_ALLOW_FC_SRC=1 V2D_FC_SRC="$SRC/v040_p185_nfg" scripts/dev/record "r3/g1_smoke_$SMOKE" stages/policy/run_queue_line.sh "${A}_smoke.txt" "$SMOKE" \
  || die "冒烟 $SMOKE 没跑完"
[ -f "runs/flash_chord/$SMOKE/DONE" ] && [ -f "runs/flash_chord/$SMOKE/policy_4096000.safetensors" ] || die "冒烟 $SMOKE 没有 DONE 或 policy_4096000"
say "2. 冒烟 $SMOKE 训完"

# 3. 8c778f9 评冒烟
V2D_FC_SRC="$SRC/fc_8c778f9" scripts/dev/record "r3/g1_smoke_eval_$SMOKE" stages/policy/fc_eval.sh "$SMOKE" 4096000 || die "8c778f9 评不了 $SMOKE"
grep -aF 'in-flash-chord: /workspace ← ' "runs/logs/$SMOKE.eval_4096000_w4096.log" | tail -1 | grep -q 8c778f905c || die "评 $SMOKE 挂的不是 8c778f9 副本"
scripts/dev/record "r3/g1_smoke_evalcheck_$SMOKE" python3 stages/policy/eval_check.py "runs/flash_chord/$SMOKE/eval_4096000_w4096.json" \
  || die "冒烟评测的 non_finite_worlds 不空或指标非有限"
say "3. 8c778f9 能读变体 A 的 checkpoint，非有限世界 0"

# 4. 基线 seed 42 重评
mapfile -t BASES < <(python3 -c 'import json, sys; s = json.load(open(sys.argv[1])); [print(e["base42"]) for e in s["episodes"] + [s["canary"]]]' "$SPEC")
for b in "${BASES[@]}"; do
  V2D_FC_SRC="$SRC/fc_8c778f9" V2D_EXPORT_WORLDS=4096 V2D_EXPORT_TRAJ=0 scripts/dev/record "r3/g1_reeval_$b" stages/evaluate/fc_export.sh "$b" 349999104 \
    || die "重评 $b 失败"
done
say "4. 基线 seed 42 重评 ${#BASES[@]} 条"

# 5. 同版重复噪声：对照的基线再评一次
CB="${BASES[-1]}"
V2D_FC_SRC="$SRC/fc_8c778f9" V2D_EVAL_TAG=rep scripts/dev/record "r3/g1_repeat_$CB" stages/policy/fc_eval.sh "$CB" 349999104 || die "重复评 $CB 失败"
CEP="$(python3 -c 'import json, sys; print("%06d" % json.load(open(sys.argv[1]))["canary"]["ep"])' "$SPEC")"
scripts/dev/record "r3/g1_repeat_compare_$CB" python3 stages/policy/eval_compare.py "runs/flash_chord/$CB/eval_349999104_w4096.rep.json" \
  "runs/flash_chord/$CB/export_8c778f9/s349999104_w4096/episode_$CEP.json" --max-add-auc-diff 1 || die "重复噪声比较失败（两次评的不是同一份 checkpoint？）"
say "5. 重复噪声见 results/r3/g1_repeat_compare_$CB.txt"
say R3_G1_DONE
