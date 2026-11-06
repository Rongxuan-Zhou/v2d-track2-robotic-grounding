#!/usr/bin/env bash
# ep5 seed 42 原样补跑（mf4090 上、仓库根执行）。计划 agent-doc/plan-t1-r3x-2026-10-08.md 第 7 节。
# 10/8 19:38（MDT）t1-ep5-w2048-cap4m-r3a-mf4090 开跑时 wandb.init 连 W&B 超时（CommError），队列行 V2D_MAX_RETRY=0 不重试，训练链记失败、
# 接着跑下一条。计划第 4 节登记过：原因在基础设施（W&B 网络等）时，可以用同一配置原样重跑一次。原来的评测链只评有 DONE 的 run，导出门见到
# “失败 0”就导出，这条会以 no_done 进导出结果表，导出名也就用掉了。所以趁原评测链与导出门还在等，把它们改挂到补跑之后：
#   v2d-r3x-train2   stages/policy/train_after.sh：等 r3x_train_mf4090 正常结束，再用只含这一条的期望表与两份队列（从登记的三个文件里
#                    原样取出，放 runs/r3x_rerun/）起训练链 r3x2_train_mf4090，环境与原训练链相同。冒烟已在原链判过、state 已删，
#                    训练链会重判一次（不重训）；重判会覆盖 results/day2/bar_<冒烟>.txt 与 runs/diag/<冒烟>/，原来的先复制到 runs/r3x_rerun/
#   v2d-r3x-eval2    评测链 r3x_eval2_mf4090：等 r3x2_train_mf4090 结束，评原期望表里 6 条有 DONE 的 run（fc_8c778f9，4096 世界，只评最终 checkpoint）
#   v2d-r3x-export2  导出门：等 r3x_eval2_mf4090，“成功 N、失败 0”才用原导出名 cand_t1_r3x_mf4090 与原清单跑导出链
# 改挂前确认（任何一条不过就停，什么也不动）：
#   - 这台是 mf4090；来历等于原链的 d8b309e9d52c；train_after.sh 已放上来；两个副本在
#   - 原训练链会话在、日志最后一行还不是结束行；原评测链日志与导出门日志都只有等待那一行；导出链的日志与结果表都不存在
#   - 失败原因：原链日志里有这条“失败（退出码 2）”，它的 s1 段日志里有 wandb 的 CommError；run 目录里没有 policy、state、DONE、NONFINITE，
#     run_record 里正好 1 个 start；冒烟有 DONE，原链里有它通过的记录
#   - 新的日志、标记、会话都不存在；可用 ≥ 2000 MB
#   - 取出的期望表 1 行（表头与原表相同、冒烟列对得上）、冒烟队列与正式队列各 1 行；precheck_expect.sh 全过；两份配置的配方断言过
# 然后紧挨着再核一次（原训练链没结束、两份旧日志仍只有一行），复制取出的文件；停原评测链与导出门的会话（只在等），确认进程都没了、
# 旧评测链日志仍只有一行，才删它们留下的标记，在两份旧日志末尾各记一行说明；先起 v2d-r3x-train2、确认它的等待标记在，再起另外两个。
# 停会话之后任何一步不过：打印收尾步骤与三条命令（人来删残留标记、手动起会话），退出码 1。这时重跑本脚本会停在“原会话不在”。
# 在训练机上 detached 执行，Mac 断线不影响（审查 m2），例如：
#   tmux new-session -d -s v2d-r3x-arm -c <仓库根> 'scripts/dev/record r3x/arm_rerun_mf4090 stages/policy/arm_r3x_rerun.sh'
# 用法：stages/policy/arm_r3x_rerun.sh      V2D_ARM_DRYRUN=1：核完只打印要做的事，不复制、不停、不起
set -euo pipefail
cd "$(dirname "$0")/../.."
SRC="$HOME/v2d/fc_src"; PEAK=24564; SRC_WANT=d8b309e9d52c
Q=stages/policy/queues/mf4090_r3x; LIST=stages/evaluate/exports/cand_t1_r3x_mf4090.tsv
RUN=t1-ep5-w2048-cap4m-r3a-mf4090; SMOKE=smoke-t1-ep5-w2048-cap4m-r3a-mf4090
LT=r3x_train_mf4090; LE=r3x_eval_mf4090; LX=cand_t1_r3x_mf4090; LT2=r3x2_train_mf4090; LE2=r3x_eval2_mf4090
D=runs/r3x_rerun
stop() { echo "arm_r3x_rerun: $*，停" >&2; exit 1; }
lines() { grep -c '' "$1" 2>/dev/null || true; }

[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" = "NVIDIA GeForce RTX 4090" ] || stop "这台不是 mf4090（GPU 不是 RTX 4090）"
[ "$(cat .v2d-source-commit 2>/dev/null)" = "$SRC_WANT" ] || stop "来历不是 $SRC_WANT"
[ -x stages/policy/train_after.sh ] || stop "缺 stages/policy/train_after.sh（先用 scripts/dev/push-new-files 放上来）"
for f in "$Q.expect.tsv" "${Q}_smoke.txt" "$Q.txt" "$LIST"; do [ -f "$f" ] || stop "缺 $f"; done
[ -f "$SRC/v040_p185_nfg/SOURCE_V2D.txt" ] && [ -f "$SRC/fc_8c778f9/SOURCE_V2D.txt" ] || stop "$SRC 下缺副本"

# 原链的状态
for s in v2d-r3x-train v2d-r3x-eval v2d-r3x-export; do tmux has-session -t "=$s" 2>/dev/null || stop "原会话 $s 不在"; done
if grep -qE 'DAY2_TRAIN_(DONE|FAILED)' < <(tail -1 "runs/logs/$LT.log"); then stop "原训练链已经结束（原评测链可能已开始）"; fi
[ "$(lines "runs/logs/$LE.log")" = 1 ] && grep -q "等 runs/logs/$LT.log" "runs/logs/$LE.log" || stop "原评测链的日志不只等待那一行"
[ "$(lines "runs/logs/${LX}_gate.log")" = 1 ] && grep -q "等 runs/logs/$LE.log" "runs/logs/${LX}_gate.log" || stop "导出门的日志不只等待那一行"
[ ! -e "runs/logs/$LX.log" ] && [ ! -e "runs/exports/$LX.tsv" ] || stop "导出链的日志或结果表已存在"

# 失败原因与 run 目录
grep -q "$RUN 失败（退出码 2）" "runs/logs/$LT.log" || stop "原链日志里没有 $RUN 的失败行"
grep -q 'wandb.errors.errors.CommError' "runs/logs/$RUN.s1.log" || stop "runs/logs/$RUN.s1.log 里没有 wandb 的 CommError"
[ -z "$(find "runs/flash_chord/$RUN" -maxdepth 1 \( -name 'policy_*' -o -name 'state_*' -o -name DONE -o -name NONFINITE \) -print -quit)" ] \
  || stop "runs/flash_chord/$RUN 里已有训练产物"
[ "$(grep -c '"event": "start"' "runs/flash_chord/$RUN/run_record.jsonl")" = 1 ] || stop "$RUN 的 run_record 里 start 不是 1 个"
[ -f "runs/flash_chord/$SMOKE/DONE" ] || stop "冒烟 $SMOKE 没有 DONE"
grep -q "冒烟 $SMOKE 通过" "runs/logs/$LT.log" || stop "原链日志里没有冒烟 $SMOKE 通过的记录"
[ -f "results/day2/bar_$SMOKE.txt" ] && [ -f "runs/diag/$SMOKE/rollout_w256.json" ] || stop "缺冒烟 $SMOKE 原来的不动基线记录或 rollout 摘要"

# 新名字
for L in "$LT2" "$LE2"; do
  [ ! -e "runs/logs/$L.log" ] || stop "已存在 runs/logs/$L.log"
  [ ! -e "runs/.chain-active-$L" ] || stop "已存在 runs/.chain-active-$L"
done
[ ! -e "runs/.chain-active-${LT2}-wait" ] || stop "已存在 runs/.chain-active-${LT2}-wait"
for s in v2d-r3x-train2 v2d-r3x-eval2 v2d-r3x-export2; do
  if tmux has-session -t "=$s" 2>/dev/null; then stop "tmux 会话 $s 已存在"; fi
done
free_mb="$(df -Pk . | awk 'NR == 2 {print int($4 / 1024)}')"
[ "$free_mb" -ge 2000 ] || stop "可用空间只有 ${free_mb} MB"

# 从登记的三个文件里原样取出这一条（写到临时目录，核过才放进 runs/r3x_rerun/）
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
head -1 "$Q.expect.tsv" > "$T/expect.tsv"
awk -F'\t' -v r="$RUN" 'NR > 1 && $1 == r' "$Q.expect.tsv" >> "$T/expect.tsv"
grep -v '^[[:space:]]*#' "${Q}_smoke.txt" | awk -v r="$SMOKE" '{for (i = 1; i <= NF; i++) if ($i == r) {print; next}}' > "$T/smoke.txt"
grep -v '^[[:space:]]*#' "$Q.txt" | awk -v r="$RUN" '{for (i = 1; i <= NF; i++) if ($i == r) {print; next}}' > "$T/queue.txt"
[ "$(lines "$T/expect.tsv")" = 2 ] && [ "$(lines "$T/smoke.txt")" = 1 ] && [ "$(lines "$T/queue.txt")" = 1 ] || stop "取出的行数不对"
[ "$(awk -F'\t' 'NR == 2 {print $2}' "$T/expect.tsv")" = "$SMOKE" ] || stop "期望表这一行的冒烟列不是 $SMOKE"
for f in expect.tsv smoke.txt queue.txt; do echo "取出 $f（md5 $(md5sum < "$T/$f" | cut -d' ' -f1)）："; sed 's/^/    /' "$T/$f"; done
stages/policy/precheck_expect.sh "$T/expect.tsv" "$T/smoke.txt" "$T/queue.txt" | tail -1 | grep -q '^PRECHECK_OK' || stop "precheck_expect.sh 不过"
VAR="objective.object_position_std=0.05 env.objective.object_position_std=0.05 termination.object_orientation_threshold=0.6 env.termination.object_orientation_threshold=0.6"
scripts/dev/in-flash-chord "bad=0; for r in $RUN $SMOKE; do python /repo/stages/policy/assert_cfg.py /repo/runs/cfgcheck/\$r.yaml $VAR || bad=1; done; exit \$bad" \
  || stop "配方断言不过（见上）"
echo "核对：来历 $SRC_WANT，可用 ${free_mb} MB，原链状态、失败原因、新名字、取出的三份文件、precheck 与配方断言全过"

CMD_T2="V2D_ALLOW_FC_SRC=1 V2D_FC_SRC=$SRC/v040_p185_nfg V2D_PRUNE_SMOKE_STATE=1 V2D_MIN_FREE_MB=1200 stages/policy/train_after.sh runs/logs/$LT.log $D/expect.tsv $D/smoke.txt $D/queue.txt $PEAK $LT2"
CMD_E2="V2D_FC_SRC=$SRC/fc_8c778f9 V2D_EVAL_STEPS=349999104 V2D_WAIT_FOR_LOG=runs/logs/$LT2.log stages/evaluate/day2_eval_chain.sh $Q.expect.tsv - $LE2"
CMD_X2="V2D_FC_SRC=$SRC/fc_8c778f9 V2D_FALLBACK_MAX=0 stages/evaluate/export_after_eval.sh runs/logs/$LE2.log $LIST $LX"
if [ "${V2D_ARM_DRYRUN:-0}" = 1 ]; then
  printf '预演，不复制、不停、不起：\n  停：v2d-r3x-eval、v2d-r3x-export\n  train2: %s\n  eval2:  %s\n  export2: %s\n' "$CMD_T2" "$CMD_E2" "$CMD_X2"
  echo DRYRUN_OK; exit 0
fi

# 紧挨着停会话再核一次：上面的容器断言要几十秒，期间原训练链若结束，原评测链会醒来、起 fc_eval 容器（审查 m1）
if grep -qE 'DAY2_TRAIN_(DONE|FAILED)' < <(tail -1 "runs/logs/$LT.log"); then stop "原训练链刚结束，不改挂"; fi
[ "$(lines "runs/logs/$LE.log")" = 1 ] && [ "$(lines "runs/logs/${LX}_gate.log")" = 1 ] || stop "原评测链或导出门的日志多了行，不改挂"
[ ! -e "$D" ] || stop "$D 已存在"
mkdir -p "$D"; cp "$T/expect.tsv" "$T/smoke.txt" "$T/queue.txt" "$D/"
cp -p "results/day2/bar_$SMOKE.txt" "$D/bar_${SMOKE}_first.txt"
cp -p "runs/diag/$SMOKE/rollout_w256.json" "$D/rollout_w256_${SMOKE}_first.json"
echo "取出的三份文件与原冒烟记录已复制到 $D/"

# 从这里起动了原链：后面任何一步不过都要人来收尾（审查 m3）
half() {
  echo "arm_r3x_rerun: $*。原评测链与导出门已停，新会话没起全。收尾：tmux ls 看 v2d-r3x-*2 会话，杀掉没起全的；删掉没有进程对应的" >&2
  echo "  runs/.chain-active-${LE}、runs/.chain-active-${LX}-gate、runs/.chain-active-r3x2_*、runs/.chain-active-${LE2}；再在仓库根按顺序手动起：" >&2
  local x
  for x in "v2d-r3x-train2|$CMD_T2" "v2d-r3x-eval2|$CMD_E2" "v2d-r3x-export2|$CMD_X2"; do
    printf '    tmux new-session -d -s %s -c %q %q\n' "${x%%|*}" "$PWD" "${x#*|}" >&2
  done
  exit 1
}
tmux kill-session -t =v2d-r3x-eval || true; tmux kill-session -t =v2d-r3x-export || true
alive() { pgrep -f "[d]ay2_eval_chain.sh .*$LE\$" >/dev/null || pgrep -f "[e]xport_after_eval.sh runs/logs/$LE.log " >/dev/null; }
for _ in $(seq 20); do alive || break; sleep 1; done
! alive || half "原评测链或导出门的进程还在（会话已停）"
[ "$(lines "runs/logs/$LE.log")" = 1 ] || half "原评测链的日志在停之前多了行，它可能已经开始评测（用 docker ps 看有没有 fc_eval 容器）"
rm -f "runs/.chain-active-$LE" "runs/.chain-active-${LX}-gate"
echo "=== $(date '+%F %T %Z') arm_r3x_rerun：本评测链在等待中停掉，改由 $LE2 评（等 $LT2；计划第 7 节）" >> "runs/logs/$LE.log"
echo "=== $(date '+%F %T %Z') arm_r3x_rerun：上面的导出门在等待中停掉；下面是新起的导出门，等 runs/logs/$LE2.log（计划第 7 节）" >> "runs/logs/${LX}_gate.log"

# 先起补跑的训练链，确认它在等，再起评测链与导出门
tmux new-session -d -s v2d-r3x-train2 -c "$PWD" "$CMD_T2"; sleep 3
tmux has-session -t =v2d-r3x-train2 2>/dev/null && [ -e "runs/.chain-active-${LT2}-wait" ] || half "v2d-r3x-train2 没起来或没写等待标记"
tmux new-session -d -s v2d-r3x-eval2 -c "$PWD" "$CMD_E2"; sleep 2
tmux new-session -d -s v2d-r3x-export2 -c "$PWD" "$CMD_X2"; sleep 3
# 以下几行只供人看：起来了就以会话与标记为准
set +e
tmux ls | grep -E '^v2d-r3x-'
for L in "$LT" "$LE2" "${LX}_gate"; do echo "$L：$(tail -1 "runs/logs/$L.log" 2>/dev/null)"; done
ls runs/.chain-active-* 2>/dev/null
n=0; for s in v2d-r3x-train2 v2d-r3x-eval2 v2d-r3x-export2; do tmux has-session -t "=$s" 2>/dev/null && n=$((n + 1)); done
if [ "$n" = 3 ] && [ -e "runs/.chain-active-${LT2}-wait" ] && [ -e "runs/.chain-active-$LE2" ] && [ -e "runs/.chain-active-${LX}-gate" ]; then
  echo ARMED
else
  half "没起全：会话 $n / 3"
fi
