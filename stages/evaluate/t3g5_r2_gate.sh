#!/usr/bin/env bash
# t3g5 判据 R2 的闸（训练机上、仓库根执行，放 tmux）。计划 agent-doc/plan-t3g5-2026-10-21.md 判据 R2 与 10/21 08:1x 的执行记录。
# R2：A 段 7 集里 ≥ 5 集没跟上，就在 B 段起之前停掉 B 段的等待（只停本计划自己挂的 arm_after.sh）。mf4090 的 B 段在它自己的 A 段导出完后几分钟就会起，
# 光靠人看来不及，这里把这条登记的规则自动做掉：
#   1. 放闸：建 runs/.chain-active-t3g5r2-<B 段>（arm_after.sh 要等 runs/.chain-active-* 都消失才起下一段），已有就拒绝；
#   2. 每 V2D_POLL_S 秒（默认 30）看 results/t3g5/eval_<A 段>_<机器>.txt 有没有 “T3G5_R1_DONE 跟上 k / 共 n”；
#      读数做完（results/t3g5/readout_<A 段>_<机器>.txt 或 .failed 出现）却没有这一行：保留闸、报 T3G5_R2_GATE_FAILED，等人来判（不猜）；
#   3. 没跟上的合计 = 另一台已知的没跟上集数 + 这台的 n − k；合计 ≥ 门槛：停 B 段的等待进程（命令行是 runs/tools/arm_after.sh <B 段> …，只认我们自己的进程），
#      查它确实没了再撤闸；合计 < 门槛：撤闸放行；
#   4. 等了 V2D_GATE_MAX_S 秒（默认 21600 = 6 小时）还没有读数：撤闸放行，记“R2 没判”（不让卡一直空着），之后只能人工判。
# 用法：stages/evaluate/t3g5_r2_gate.sh <A 段> <机器> <另一台已知没跟上的集数> <门槛> <B 段>   例：… t3pg5a mf4090 2 5 t3pg5b
# 日志写到标准输出（调用方重定向到 runs/logs/t3g5_r2_gate_<机器>.log），最后一行 T3G5_R2_GATE_DONE <结论> 或 T3G5_R2_GATE_FAILED <原因>。
# 测试：stages/evaluate/test_t3g5_r2_gate.py。
set -uo pipefail
cd "${V2D_GATE_ROOT:-$(dirname "$0")/../..}"
SEG=${1:?A 段}; MACH=${2:?机器}; OTHER=${3:?另一台没跟上的集数}; THR=${4:?门槛}; NEXT=${5:?B 段}
POLL=${V2D_POLL_S:-30}; MAX=${V2D_GATE_MAX_S:-21600}
HOLD="runs/.chain-active-t3g5r2-$NEXT"
EVAL="results/t3g5/eval_${SEG}_${MACH}.txt"; READ="results/t3g5/readout_${SEG}_${MACH}.txt"
say() { echo "=== $(date '+%F %T %Z') $*"; }
[[ $OTHER =~ ^[0-9]+$ && $THR =~ ^[0-9]+$ ]] || { echo "T3G5_R2_GATE_FAILED 集数与门槛要是整数"; exit 2; }
[ -e "$HOLD" ] && { echo "T3G5_R2_GATE_FAILED 闸文件 $HOLD 已存在"; exit 2; }
mkdir -p runs
echo "t3g5 判据 R2 的闸（stages/evaluate/t3g5_r2_gate.sh $*）：等 $EVAL 的 R1 结论再决定 $NEXT 起不起" > "$HOLD"
say "放闸 $HOLD；等 $EVAL；另一台已知没跟上 $OTHER 集，门槛 $THR"
waited=0
line=""
while :; do
  line="$(grep -h -o 'T3G5_R1_DONE 跟上 [0-9]* / 共 [0-9]*' "$EVAL" 2>/dev/null | tail -1)"
  [ -n "$line" ] && break
  if [ -e "$READ" ] || [ -e "$READ.failed" ]; then
    say "读数做完了，但 $EVAL 里没有 R1 结论；闸留着，等人来判"
    echo "T3G5_R2_GATE_FAILED 读不到 R1"; exit 1
  fi
  if [ "$waited" -ge "$MAX" ]; then
    rm -f "$HOLD"; say "等了 ${waited} 秒没有读数：撤闸放行，R2 没判"
    echo "T3G5_R2_GATE_DONE 超时放行（R2 没判）"; exit 0
  fi
  sleep "$POLL"; waited=$((waited + POLL))
done
k="$(sed -E 's/.*跟上 ([0-9]+) \/ 共 ([0-9]+).*/\1/' <<< "$line")"; n="$(sed -E 's/.*跟上 ([0-9]+) \/ 共 ([0-9]+).*/\2/' <<< "$line")"
fails=$((OTHER + n - k))
say "R1：这台 $line；没跟上合计 ${fails}（另一台 $OTHER + 这台 $((n - k))），门槛 $THR"
if [ "$fails" -ge "$THR" ]; then
  pids="$(pgrep -u "$(id -u)" -f "[r]uns/tools/arm_after.sh $NEXT " || true)"
  if [ -z "$pids" ]; then
    say "R2 成立，但没找到 $NEXT 的等待进程（可能已经起了）；闸留着，等人来看"
    echo "T3G5_R2_GATE_FAILED 没找到等待进程"; exit 1
  fi
  say "R2 成立：停 $NEXT 的等待进程 $(tr '\n' ' ' <<< "$pids")"
  kill $pids
  for _ in 1 2 3 4 5 6 7 8 9 10; do pgrep -u "$(id -u)" -f "[r]uns/tools/arm_after.sh $NEXT " > /dev/null || break; sleep 1; done
  if pgrep -u "$(id -u)" -f "[r]uns/tools/arm_after.sh $NEXT " > /dev/null; then
    say "等待进程没停下；闸留着"; echo "T3G5_R2_GATE_FAILED 等待进程没停下"; exit 1
  fi
  rm -f "$HOLD"; say "等待进程已停、撤闸；B 段不起，照 R2 另写计划"
  echo "T3G5_R2_GATE_DONE R2 成立，$NEXT 不起"
else
  rm -f "$HOLD"; say "R2 不成立：撤闸，$NEXT 照常起"
  echo "T3G5_R2_GATE_DONE R2 不成立，$NEXT 放行"
fi
