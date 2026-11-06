#!/usr/bin/env bash
# 一集的 checkpoint 扫描：粗扫、细扫、按 MPPE 选点、导出（训练机上、仓库根执行；由 arm_after.sh 的 V2D_PRE_CMD 调用）。
# 计划 agent-doc/plan-ep25-sweep-seeds-2026-10-18.md 第 2 节。评测与导出都用现成的链，本脚本只负责先后顺序：
# 1. 粗扫：day2_eval_chain.sh - <粗扫表> <名字>_c（附加表 eval 行，4096 世界，记录 results/day2/eval_<run>_s<步数>.txt）。
# 2. 细扫：ckpt_sweep_pick.py refine 写 runs/ckpt_sweep/<名字>_refine.tsv（每条 run 选一步，前后各补一份）；不空就
#    day2_eval_chain.sh - <细扫表> <名字>_r。选法：MPPE 不超过最低值 + 1 cm 的点里取步数最大的（ckpt_sweep_pick.py 的 choose）。
# 3. 选点：ckpt_sweep_pick.py final（经 record 落成 results/ckpt_sweep/<名字>_pick.txt），每档本机按同样的选法选一点，低于上限就写进
#    runs/ckpt_sweep/<名字>_export.tsv。
# 4. 导出：清单不空就 export_chain.sh <清单> <名字>_x（1 世界、check_export、候选条件；V2D_FALLBACK_MAX=0）。
# 5. 只报：清单里每一点再跑一次逐世界 dump（fc_eval_dump.sh，4096 世界，results/ckpt_sweep/dump_<run>_s<步数>.txt），
#    dump_knock.py 报物体平均位置误差超过 0.8 m 的世界比例（瓶子被带离很远的概率），results/ckpt_sweep/knock_<run>_s<步数>.txt。
# 用法：V2D_FC_SRC=<8c778f9 副本> stages/evaluate/ckpt_sweep.sh <名字> <粗扫表> <MPPE 上限 cm>
#   V2D_GPU_WAIT_S（默认 7200）原样交给两条链。V2D_SWEEP_DRYRUN=1：只核参数、表格与副本，打印将要执行的命令，不写任何文件。
# 从开始到结束写长链标记 runs/.chain-active-<名字>（挡同步）。日志 runs/logs/<名字>.log，最后一行 CKPT_SWEEP_DONE（…）。
# 参数对就总是退出码 0：评测、选点或导出失败只记进日志，不挡 arm_after 后面要起的训练段。参数不对退出码 2，不写日志。
set -uo pipefail
cd "$(dirname "$0")/../.."
NAME=${1:?用法见文件头}; COARSE=${2:?用法见文件头}; CAP=${3:?用法见文件头}
die() { echo "ckpt_sweep: $*" >&2; exit 2; }
[[ $NAME =~ ^[a-z0-9_]+$ ]] || die "名字只许小写字母、数字、下划线：$NAME"
[[ $CAP =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "MPPE 上限要是正数：$CAP"
[ -s "$COARSE" ] || die "粗扫表 $COARSE 不在或是空的"
[ -n "${V2D_FC_SRC:-}" ] && head -1 "$V2D_FC_SRC/SOURCE_V2D.txt" 2>/dev/null | grep -q 8c778f905c \
  || die "V2D_FC_SRC 必须是 8c778f9 副本（现在是 ${V2D_FC_SRC:-未设}）"
LOG="runs/logs/$NAME.log"; MARK="runs/.chain-active-$NAME"; D="runs/ckpt_sweep"
REF="$D/${NAME}_refine.tsv"; EXL="$D/${NAME}_export.tsv"
for f in "$LOG" "runs/logs/${NAME}_c.log" "runs/logs/${NAME}_r.log" "runs/logs/${NAME}_x.log" "runs/exports/${NAME}_x.tsv" "$REF" "$EXL"; do
  [ ! -e "$f" ] || die "$f 已存在：不沿用旧的扫描，确认后改名再起"
done
python3 stages/evaluate/ckpt_sweep_pick.py refine "$COARSE" /dev/null > /dev/null || die "粗扫表 $COARSE 读不了（见 ckpt_sweep_pick.py 的报错）"
while IFS=$'\t' read -r kind run step _; do
  [ "$kind" = eval ] || continue
  [ -f "runs/flash_chord/$run/DONE" ] || die "$run 没有 DONE"
  [ -f "runs/flash_chord/$run/policy_$step.safetensors" ] || die "没有 runs/flash_chord/$run/policy_$step.safetensors"
done < "$COARSE"
GW="${V2D_GPU_WAIT_S:-7200}"
[[ $GW =~ ^[0-9]+$ ]] || die "V2D_GPU_WAIT_S 只能是秒数：$GW"
if [ "${V2D_SWEEP_DRYRUN:-0}" = 1 ]; then
  echo "DRYRUN 粗扫 $(grep -c '^eval' "$COARSE") 份：V2D_GPU_WAIT_S=$GW stages/evaluate/day2_eval_chain.sh - $COARSE ${NAME}_c"
  echo "DRYRUN 细扫：ckpt_sweep_pick.py refine $COARSE ${REF}，不空就 day2_eval_chain.sh - $REF ${NAME}_r"
  echo "DRYRUN 选点：ckpt_sweep_pick.py final $COARSE $EXL ${CAP}（记录 results/ckpt_sweep/${NAME}_pick.txt）"
  echo "DRYRUN 导出：清单不空就 V2D_FALLBACK_MAX=0 stages/evaluate/export_chain.sh $EXL ${NAME}_x"
  echo "DRYRUN 只报：清单里每一点 fc_eval_dump.sh <run> <步数>，再 dump_knock.py"
  echo "CKPT_SWEEP_DRYRUN_OK"; exit 0
fi
mkdir -p runs/logs "$D"
echo "$NAME running $(date '+%F %T %Z')" > "$MARK"; trap 'rm -f "$MARK"' EXIT
exec >> "$LOG" 2>&1
say() { echo "=== $(date '+%F %T %Z') $*"; }
demark() { sed -E 's/DAY2_EVAL_(DONE|FAILED)/评测链结束（\1）/g; s/EXPORT_CHAIN_(DONE|FAILED)/导出链结束（\1）/g; s/CKPT_SWEEP_PICK_DONE/选点做完/g'; }
md5of() { { md5sum "$1" 2>/dev/null || true; } | cut -c1-12; }
say "start ${NAME}：粗扫表 ${COARSE}（$(grep -c '^eval' "$COARSE") 份），MPPE 上限 ${CAP} cm；来历 $(cat .v2d-source-commit 2>/dev/null || echo unknown)"
say "脚本 md5：ckpt_sweep.sh $(md5of stages/evaluate/ckpt_sweep.sh)，ckpt_sweep_pick.py $(md5of stages/evaluate/ckpt_sweep_pick.py)，粗扫表 $(md5of "$COARSE")"

V2D_GPU_WAIT_S="$GW" stages/evaluate/day2_eval_chain.sh - "$COARSE" "${NAME}_c" < /dev/null
say "粗扫：$(tail -1 "runs/logs/${NAME}_c.log" 2>/dev/null | demark)"

if python3 stages/evaluate/ckpt_sweep_pick.py refine "$COARSE" "$REF" < /dev/null; then
  if [ -s "$REF" ]; then
    V2D_GPU_WAIT_S="$GW" stages/evaluate/day2_eval_chain.sh - "$REF" "${NAME}_r" < /dev/null
    say "细扫：$(tail -1 "runs/logs/${NAME}_r.log" 2>/dev/null | demark)"
  else
    say "细扫：没有要补的"
  fi
else
  say "细扫选点出错，跳过细扫"
fi

pick=失败; xs=没导出
if scripts/dev/record "ckpt_sweep/${NAME}_pick" python3 stages/evaluate/ckpt_sweep_pick.py final "$COARSE" "$EXL" "$CAP" < /dev/null; then
  pick="导出清单 $(grep -c . "$EXL" 2>/dev/null || echo 0) 行"
  grep -E '^Tier [0-9]+ 本机选点' "results/ckpt_sweep/${NAME}_pick.txt"
  if [ -s "$EXL" ]; then
    V2D_GPU_WAIT_S="$GW" V2D_FALLBACK_MAX=0 stages/evaluate/export_chain.sh "$EXL" "${NAME}_x" < /dev/null
    xs="$(tail -1 "runs/logs/${NAME}_x.log" 2>/dev/null | demark)"
  fi
fi

nd=0
if [ -s "$EXL" ]; then
  while IFS=$'\t' read -r _ _ run step; do
    [ -n "${run:-}" ] || continue
    if scripts/dev/record "ckpt_sweep/dump_${run}_s${step}" stages/evaluate/fc_eval_dump.sh "$run" "$step" < /dev/null \
       && scripts/dev/record "ckpt_sweep/knock_${run}_s${step}" python3 stages/evaluate/dump_knock.py "runs/flash_chord/$run/eval_dump/eval_${step}_w4096.npz" < /dev/null; then
      nd=$((nd + 1)); say "只报：${run} 第 ${step} 步 $(grep -E '^平均位置误差超过' "results/ckpt_sweep/knock_${run}_s${step}.txt")"
    else
      say "只报：${run} 第 ${step} 步的逐世界 dump 没做成（不影响别的）"
    fi
  done < "$EXL"
fi
say "CKPT_SWEEP_DONE（粗扫 $(tail -1 "runs/logs/${NAME}_c.log" 2>/dev/null | demark | sed 's/^=== [^ ]* [^ ]* [^ ]* //')；选点 ${pick}；导出 ${xs:-没导出}；dump ${nd} 份）"
exit 0
