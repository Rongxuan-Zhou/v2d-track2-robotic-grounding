# 一批 run 的健康记录：列出计划第 4 节 R1–R3 要看的行，只读，不改任何文件、不起容器。Mac 上经 stdin 送过去执行，训练机不用同步仓库：
#   scripts/dev/record t3p/health_mf4090 ssh -o BatchMode=yes mf4090 "cd <仓库> && bash -s -- t3p_train_mf4090 t3p_eval_mf4090 cand_t3_t3p_mf4090_gate cand_t3_t3p_mf4090 t3-ep8-w2048-cap4m-r3a-mf4090 t3-ep24-w2048-cap4m-r3a-mf4090" < stages/policy/chain_health_remote.sh
# 参数：<训练链日志名> <评测链日志名> <导出门日志名> <导出链日志名> <run>...（日志名不带 runs/logs/ 与 .log；导出结果表是 runs/exports/<导出链日志名>.tsv）。
# 列法同 results/r3x/health_mf4090.txt：训练链的 === 行（冒烟行另列）与含“发散”的行数；每条 run 的 DONE、NONFINITE、NONFINITE_GUARD_STOP 与
# NONFINITE_GUARD 的行数、目录大小、policy 与 state 份数、看守日志；评测链、导出门、导出链的 === 行与导出结果表。过不过由人按计划的判据读。
set -uo pipefail
TRAIN=${1:?训练链日志名}; EVAL=${2:?评测链日志名}; GATE=${3:?导出门日志名}; EXP=${4:?导出链日志名}; shift 4
[ $# -ge 1 ] || { echo "chain_health: 至少要一条 run" >&2; exit 2; }
echo "== $(hostname) $(date '+%F %T %Z')；来历 $(cat .v2d-source-commit 2>/dev/null || echo unknown)"
f="runs/logs/$TRAIN.log"
echo "## 训练链 $TRAIN（不含冒烟行）"
grep -a '^=== ' "$f" | grep -av '冒烟' | tail -n 40
echo "## $TRAIN 的冒烟行"
grep -a '^=== .*冒烟' "$f" | tail -n 40
echo "DIVERGE_IN_CHAIN $TRAIN $(grep -ac '发散' "$f")"
echo "## 各 run"
for r in "$@"; do
  d="runs/flash_chord/$r"
  logs=$(ls runs/logs/"$r".log runs/logs/"$r".s*.log 2>/dev/null)
  echo "$r DONE=$([ -f "$d/DONE" ] && echo 1 || echo 0) NONFINITE=$([ -f "$d/NONFINITE" ] && echo 1 || echo 0)" \
    "GUARD_STOP=$(cat $logs /dev/null | grep -ac NONFINITE_GUARD_STOP) GUARD=$(cat $logs /dev/null | grep -ac NONFINITE_GUARD)" \
    "MB=$(du -sm "$d" 2>/dev/null | cut -f1) policy=$(ls "$d"/policy_*.safetensors 2>/dev/null | wc -l) state=$(ls "$d"/state_*.safetensors 2>/dev/null | wc -l)"
  w="runs/logs/$r.nanwatch.log"
  [ -f "$w" ] && echo "  看守：含非有限的扫描行 $(grep -acE '非有限 [1-9]' "$w")，最后一行：$(tail -n 1 "$w")"
done
for L in "$EVAL" "$GATE" "$EXP"; do
  echo "## $L"
  if [ -f "runs/logs/$L.log" ]; then grep -a '^=== ' "runs/logs/$L.log" | tail -n 40; else echo "（没有这个日志）"; fi
done
echo "## 导出结果表 runs/exports/$EXP.tsv"
cat "runs/exports/$EXP.tsv" 2>/dev/null || echo "（没有这个表）"
echo "CHAIN_HEALTH_DONE"
