#!/usr/bin/env bash
# 起 Tier 1 变体 A 补跑（ep0、ep15、ep5，seed 42 与 43）的三个 tmux（mf4090 上、仓库根执行）。计划 agent-doc/plan-t1-r3x-2026-10-08.md。
# 队列与导出清单由 stages/policy/make_r3x_queue.py 生成。起之前确认（任何一条不过就停，什么也不起）：
#   - 这台是 mf4090（GPU 是 RTX 4090）；来历干净；三段日志、导出结果表、长链标记、同名 tmux 会话都不存在；GPU 上没有进程；
#     仓库所在分区可用 ≥ 4000 MB（6 条约 2 GB，训练链不到 1200 MB 就停）
#   - 副本 v040_p185_nfg 与 fc_8c778f9 都在，v040_p185_nfg 按 sha256 清单核对通过
#   - 导出要读的 runs/motion_src/tier1/<序列>/v2d_meta.json 都在（fc_export.sh 从这里取 episode 号；审查 M1）
#   - 期望表 6 条 run 与冒烟都没有训练产物（目录不存在，或里面只有 fc_config_check.sh 经 fc_run.sh 打印模式建的空 .lock；不续训、不覆盖）；
#     导出清单的 6 行与期望表的 run 一一对应、顺序相同
#   - 12 份配置的 runs/cfgcheck/<run>.yaml 都在（fc_config_check.sh 先生成），precheck_expect.sh 全过；
#     配方断言：objective.object_position_std 与 termination.object_orientation_threshold（顶层与 env 下各一份）都是变体 A 的 0.05、0.6
# 起三个会话：
#   v2d-r3x-train   训练链（环境给 V2D_ALLOW_FC_SRC=1 V2D_FC_SRC=<副本>/v040_p185_nfg，冒烟判过后删它的 state，可用空间不到 1200 MB 就停）
#   v2d-r3x-eval    评测链：等训练链结束，用 fc_8c778f9 只评最终 checkpoint（4096 世界）
#   v2d-r3x-export  导出门（stages/evaluate/export_after_eval.sh）：等评测链结束，最后一行是“成功 N、失败 0”才跑导出链，
#                   用 fc_8c778f9 导出 1 个世界的轨迹，V2D_FALLBACK_MAX=0（只用最终 checkpoint，不回退）；评测链没有干净结束就一条都不导
#                   （没有 4096 世界评测的导出进不了候选，又不许重导；审查 M2）
# 用法：stages/policy/arm_r3x_chain.sh      V2D_ARM_DRYRUN=1：核完只打印三条命令，不起会话
set -euo pipefail
cd "$(dirname "$0")/../.."
SRC="$HOME/v2d/fc_src"; PEAK=24564
Q=stages/policy/queues/mf4090_r3x; LIST=stages/evaluate/exports/cand_t1_r3x_mf4090.tsv
LT=r3x_train_mf4090; LE=r3x_eval_mf4090; LX=cand_t1_r3x_mf4090
stop() { echo "arm_r3x_chain: $*，停" >&2; exit 1; }

[ "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)" = "NVIDIA GeForce RTX 4090" ] || stop "这台不是 mf4090（GPU 不是 RTX 4090）"
src0="$(cat .v2d-source-commit 2>/dev/null || echo unknown)"
case "$src0" in *-dirty*|*-syncing*|unknown) stop "来历 $src0 不干净" ;; esac
for f in "$Q.expect.tsv" "${Q}_smoke.txt" "$Q.txt" "$LIST"; do [ -f "$f" ] || stop "缺 $f"; done
for L in "$LT" "$LE" "$LX"; do
  [ ! -e "runs/logs/$L.log" ] || stop "已存在 runs/logs/$L.log"
  [ ! -e "runs/.chain-active-$L" ] || stop "已存在 runs/.chain-active-$L"
done
[ ! -e "runs/exports/$LX.tsv" ] || stop "已存在 runs/exports/$LX.tsv"
[ ! -e "runs/logs/${LX}_gate.log" ] && [ ! -e "runs/.chain-active-${LX}-gate" ] || stop "已存在导出门的日志或标记"
for s in v2d-r3x-train v2d-r3x-eval v2d-r3x-export; do
  if tmux has-session -t "$s" 2>/dev/null; then stop "tmux 会话 $s 已存在"; fi
done
others="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | tr -d ' ')"
[ -z "$others" ] || stop "GPU 上有别的进程（${others//$'\n'/,}）"
free_mb="$(df -Pk . | awk 'NR == 2 {print int($4 / 1024)}')"
[ "$free_mb" -ge 4000 ] || stop "可用空间只有 ${free_mb} MB"
[ -f "$SRC/v040_p185_nfg/SOURCE_V2D.txt" ] && [ -f "$SRC/fc_8c778f9/SOURCE_V2D.txt" ] || stop "$SRC 下缺副本"
python3 stages/policy/fc_src_check.py "$SRC/v040_p185_nfg" third_party/patches/flash_chord-p185-nfg.sha256 | tail -1 | grep -q '^FC_SRC_OK' \
  || stop "$SRC/v040_p185_nfg 按清单核对不过"

runs=(); cfgs=(); seqs=()
while IFS=$'\t' read -r run smoke seq _; do runs+=("$run"); cfgs+=("$run" "$smoke"); seqs+=("$seq"); done < <(tail -n +2 "$Q.expect.tsv" | grep -v '^[[:space:]]*$')
for seq in "${seqs[@]}"; do [ -f "runs/motion_src/tier1/$seq/v2d_meta.json" ] || stop "缺 runs/motion_src/tier1/$seq/v2d_meta.json（导出要用）"; done
[ "${#runs[@]}" = 6 ] || stop "期望表里应有 6 条 run，实际 ${#runs[@]}"
mapfile -t listed < <(grep -vE '^(#|[[:space:]]*$)' "$LIST" | cut -f3)
[ "${listed[*]}" = "${runs[*]}" ] || stop "导出清单的 run 与期望表不一致：${listed[*]}"
for r in "${cfgs[@]}"; do
  # 目录还不存在时 find 退出码非零，set -e 下会不打印就退出（10/16 凌晨 t1ob、t2ob 就停在这里）；不存在就当没有产物，往下核 cfgcheck
  left="$(find "runs/flash_chord/$r" -mindepth 1 ! -name .lock -print -quit 2>/dev/null || true)"
  [ -z "$left" ] || stop "runs/flash_chord/$r 里已有训练产物（$left）"
  [ -f "runs/cfgcheck/$r.yaml" ] || stop "缺 runs/cfgcheck/$r.yaml（先跑 fc_config_check.sh）"
done
stages/policy/precheck_expect.sh "$Q.expect.tsv" "${Q}_smoke.txt" "$Q.txt" | tail -1 | grep -q '^PRECHECK_OK' || stop "precheck_expect.sh 不过"
VAR="objective.object_position_std=0.05 env.objective.object_position_std=0.05 termination.object_orientation_threshold=0.6 env.termination.object_orientation_threshold=0.6"
cmds=""
for r in "${cfgs[@]}"; do cmds+="python /repo/stages/policy/assert_cfg.py /repo/runs/cfgcheck/$r.yaml $VAR || bad=1; "; done
scripts/dev/in-flash-chord "bad=0; ${cmds}exit \$bad" || stop "配方断言不过（见上）"
echo "核对：来历 $src0，可用 ${free_mb} MB，${#cfgs[@]} 份配置的 precheck 与配方断言全过"

CMD_T="V2D_ALLOW_FC_SRC=1 V2D_FC_SRC=$SRC/v040_p185_nfg V2D_PRUNE_SMOKE_STATE=1 V2D_MIN_FREE_MB=1200 stages/policy/day2_train_chain.sh $Q.expect.tsv ${Q}_smoke.txt $Q.txt $PEAK $LT"
CMD_E="V2D_FC_SRC=$SRC/fc_8c778f9 V2D_EVAL_STEPS=349999104 V2D_WAIT_FOR_LOG=runs/logs/$LT.log stages/evaluate/day2_eval_chain.sh $Q.expect.tsv - $LE"
CMD_X="V2D_FC_SRC=$SRC/fc_8c778f9 V2D_FALLBACK_MAX=0 stages/evaluate/export_after_eval.sh runs/logs/$LE.log $LIST $LX"
if [ "${V2D_ARM_DRYRUN:-0}" = 1 ]; then
  printf '预演，不起会话：\n  train:  %s\n  eval:   %s\n  export: %s\n' "$CMD_T" "$CMD_E" "$CMD_X"
  echo DRYRUN_OK; exit 0
fi
tmux new-session -d -s v2d-r3x-train -c "$PWD" "$CMD_T"; sleep 2
tmux new-session -d -s v2d-r3x-eval -c "$PWD" "$CMD_E"; sleep 2
tmux new-session -d -s v2d-r3x-export -c "$PWD" "$CMD_X"; sleep 3
# 以下几行只供人看：起来了就以会话与标记为准
set +e
tmux ls | grep -E '^v2d-r3x-(train|eval|export):'
for L in "$LT" "$LE" "$LX"; do echo "$L：$(tail -1 "runs/logs/$L.log" 2>/dev/null)"; done
ls runs/.chain-active-* 2>/dev/null
n=0; for s in v2d-r3x-train v2d-r3x-eval v2d-r3x-export; do tmux has-session -t "$s" 2>/dev/null && n=$((n + 1)); done
if [ "$n" = 3 ] && [ -e "runs/.chain-active-$LT" ]; then echo ARMED; else echo "没起全：会话 $n / 3" >&2; exit 1; fi
