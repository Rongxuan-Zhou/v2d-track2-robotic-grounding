#!/usr/bin/env bash
# 起第 3 步配方试验的四个 tmux（训练机上、仓库根执行）。计划 agent-doc/plan-step3-recipe-trial-2026-10-04.md 第 2 版第 3、7 节。
# 机器定档位：mf4090 跑 Tier 1，ml40 跑 Tier 2。队列与试验清单由 stages/policy/make_trial_queue.py 生成。
# 起之前确认（任何一条不过就停，什么也不起）：
#   - 四段日志、长链标记、同名 tmux 会话都不存在；GPU 上没有别的进程；
#   - 副本 v040_p185_nfg 与 fc_8c778f9 都在，v040_p185_nfg 按 sha256 清单核对通过；
#   - 两段期望表里每条 run 与冒烟的 runs/cfgcheck/<run>.yaml 都在（fc_config_check.sh 先生成）；
#   - 配方断言（审查 M5）：名字带 -r3a- 的，objective.object_position_std 与 termination.object_orientation_threshold（顶层与 env 下各一份）
#     是变体 A 的 0.05、0.6；其余是官方配方的 0.2、1.5；
#   - 试验清单里基线 seed 42 的 8c778f9 重评（fc_export.sh，4096 世界）都在；段间门的判定命令在这台机器的宿主 python3 上能跑
#     （此刻变体还没评，应当报 MIDCHECK ERROR 且原因是变体的评测结果不存在）。
# 起四个会话：
#   v2d-r3-a-train   A 段训练链（环境给 V2D_ALLOW_FC_SRC=1 V2D_FC_SRC=<副本>/v040_p185_nfg）
#   v2d-r3-a-eval    A 段评测链：等 A 段训练链结束，用 fc_8c778f9 只评最终 checkpoint（4096 世界）
#   v2d-r3-bc-gate   段间门：chain_after.sh 等 A 段评完、A 段完成 6 条，再跑 trial_verdict.py --midcheck，CONTINUE 才 exec BC 段训练链
#   v2d-r3-bc-eval   BC 段评测链：等 BC 段训练日志结束（段间门判停时那份日志最后一行是 DAY2_TRAIN_FAILED，评测链什么也不评）
# 用法：stages/policy/arm_trial_chains.sh mf4090|ml40      V2D_ARM_DRYRUN=1：核完只打印四条命令，不起会话
set -euo pipefail
cd "$(dirname "$0")/../.."
M=${1:?用法：arm_trial_chains.sh mf4090|ml40}
case "$M" in
  mf4090) T=1; SRC="$HOME/v2d/fc_src"; PEAK=24564; EXTRA="V2D_PRUNE_SMOKE_STATE=1 V2D_MIN_FREE_MB=1200" ;;
  ml40)   T=2; SRC=/lab_data/rxzhou/v2d/fc_src; PEAK=44000; EXTRA="" ;;
  *) echo "机器只能是 mf4090 或 ml40" >&2; exit 2 ;;
esac
Q=stages/policy/queues; SPEC=stages/evaluate/trials/r3_t${T}_${M}.json
A=$Q/${M}_r3t${T}a; BC=$Q/${M}_r3t${T}bc
LA_T=r3t${T}a_train_$M; LA_E=r3t${T}a_eval_$M; LB_T=r3t${T}bc_train_$M; LB_E=r3t${T}bc_eval_$M
stop() { echo "arm_trial_chains: $*，停" >&2; exit 1; }

for f in "$A.expect.tsv" "${A}_smoke.txt" "$A.txt" "$BC.expect.tsv" "${BC}_smoke.txt" "$BC.txt" "$SPEC"; do [ -f "$f" ] || stop "缺 $f"; done
for L in "$LA_T" "$LA_E" "$LB_T" "$LB_E"; do
  [ ! -e "runs/logs/$L.log" ] || stop "已存在 runs/logs/$L.log"
  [ ! -e "runs/.chain-active-$L" ] || stop "已存在 runs/.chain-active-$L"
done
for s in v2d-r3-a-train v2d-r3-a-eval v2d-r3-bc-gate v2d-r3-bc-eval; do
  if tmux has-session -t "$s" 2>/dev/null; then stop "tmux 会话 $s 已存在"; fi
done
others="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')"
[ -z "$others" ] || stop "GPU 上有别的进程（${others//$'\n'/,}）"
[ -f "$SRC/v040_p185_nfg/SOURCE_V2D.txt" ] && [ -f "$SRC/fc_8c778f9/SOURCE_V2D.txt" ] || stop "$SRC 下缺副本"
python3 stages/policy/fc_src_check.py "$SRC/v040_p185_nfg" third_party/patches/flash_chord-p185-nfg.sha256 | tail -1 | grep -q '^FC_SRC_OK' \
  || stop "$SRC/v040_p185_nfg 按清单核对不过"

runs=()
for E in "$A.expect.tsv" "$BC.expect.tsv"; do
  while IFS=$'\t' read -r run smoke _; do runs+=("$run" "$smoke"); done < <(tail -n +2 "$E" | grep -v '^[[:space:]]*$')
done
[ "${#runs[@]}" = 32 ] || stop "两段期望表里应有 16 条 run（连冒烟 32 份配置），实际 ${#runs[@]}"
VAR="objective.object_position_std=0.05 env.objective.object_position_std=0.05 termination.object_orientation_threshold=0.6 env.termination.object_orientation_threshold=0.6"
BASE="objective.object_position_std=0.2 env.objective.object_position_std=0.2 termination.object_orientation_threshold=1.5 env.termination.object_orientation_threshold=1.5"
cmds=""
for r in "${runs[@]}"; do
  [ -f "runs/cfgcheck/$r.yaml" ] || stop "缺 runs/cfgcheck/$r.yaml（先跑 fc_config_check.sh）"
  case "$r" in *-r3a-*) a=$VAR ;; *) a=$BASE ;; esac
  cmds+="python /repo/stages/policy/assert_cfg.py /repo/runs/cfgcheck/$r.yaml $a || bad=1; "
done
scripts/dev/in-flash-chord "bad=0; ${cmds}exit \$bad" || stop "配方断言不过（见上）"
echo "配方断言：${#runs[@]} 份配置全过"

python3 - "$SPEC" <<'PY' || stop "基线 seed 42 的 8c778f9 重评不齐"
import json, sys
from pathlib import Path
s = json.load(open(sys.argv[1]))
need = [(e["base42"], e["ep"]) for e in s["episodes"]] + [(s["canary"]["base42"], s["canary"]["ep"])]
miss = [r for r, ep in need if not (Path("runs/flash_chord") / r / "export_8c778f9" / f"s{s['steps']}_w{s['worlds']}" / f"episode_{ep:06d}.json").is_file()]
print(f"基线 seed 42 的重评：{len(need) - len(miss)} / {len(need)} 在" + (f"，缺 {miss}" if miss else ""))
sys.exit(1 if miss else 0)
PY
set +e; gate_out="$(python3 stages/evaluate/trial_verdict.py "$SPEC" --midcheck 2>&1)"; grc=$?; set -e
echo "$gate_out" | tail -2
[ "$grc" = 2 ] && grep -q "MIDCHECK ERROR .*-r3a-.*没有评测结果" <<< "$gate_out" || stop "段间门的判定命令在这里跑不对（退出码 $grc）"

CMD_AT="V2D_ALLOW_FC_SRC=1 V2D_FC_SRC=$SRC/v040_p185_nfg $EXTRA stages/policy/day2_train_chain.sh $A.expect.tsv ${A}_smoke.txt $A.txt $PEAK $LA_T"
CMD_AE="V2D_FC_SRC=$SRC/fc_8c778f9 V2D_EVAL_STEPS=349999104 V2D_WAIT_FOR_LOG=runs/logs/$LA_T.log stages/evaluate/day2_eval_chain.sh $A.expect.tsv - $LA_E"
CMD_BG="V2D_ALLOW_FC_SRC=1 V2D_FC_SRC=$SRC/v040_p185_nfg $EXTRA V2D_GATE_CMD='python3 stages/evaluate/trial_verdict.py $SPEC --midcheck' stages/policy/chain_after.sh runs/logs/$LA_T.log runs/logs/$LA_E.log 6 $BC.expect.tsv ${BC}_smoke.txt $BC.txt $PEAK $LB_T"
CMD_BE="V2D_FC_SRC=$SRC/fc_8c778f9 V2D_EVAL_STEPS=349999104 V2D_WAIT_FOR_LOG=runs/logs/$LB_T.log stages/evaluate/day2_eval_chain.sh $BC.expect.tsv - $LB_E"
if [ "${V2D_ARM_DRYRUN:-0}" = 1 ]; then
  printf '预演，不起会话：\n  a-train: %s\n  a-eval:  %s\n  bc-gate: %s\n  bc-eval: %s\n' "$CMD_AT" "$CMD_AE" "$CMD_BG" "$CMD_BE"
  echo DRYRUN_OK; exit 0
fi
tmux new-session -d -s v2d-r3-a-train -c "$PWD" "$CMD_AT"; sleep 2
tmux new-session -d -s v2d-r3-a-eval -c "$PWD" "$CMD_AE"; sleep 2
tmux new-session -d -s v2d-r3-bc-gate -c "$PWD" "$CMD_BG"; sleep 2
tmux new-session -d -s v2d-r3-bc-eval -c "$PWD" "$CMD_BE"; sleep 3
tmux ls | grep -E '^v2d-r3-(a-train|a-eval|bc-gate|bc-eval):'
for L in "$LA_T" "$LA_E" "$LB_E"; do echo "$L：$(tail -1 "runs/logs/$L.log" 2>/dev/null)"; done
ls runs/.chain-active-* 2>/dev/null
echo ARMED
