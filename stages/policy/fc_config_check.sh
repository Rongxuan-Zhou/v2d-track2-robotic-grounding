#!/usr/bin/env bash
# 开训前核对：对队列里的每一条，用 fc_run.sh 的打印模式拿到真正会执行的命令，在容器里（不占 GPU）
# 展开最终配置（--cfg job --resolve），与只指定 parquet 的官方 G1 配方逐键比较。
# 允许的差异：logging.*（及其派生的 training.experiment_name/run_name/wandb_entity/wandb_project）、output_dir、task.parquet、scene.world_count、training.total_environment_steps、
# 以及队列行里显式写出的覆盖键（例如 training.replay.capacity）。其余任何差异都算失败。
# 用法：stages/policy/fc_config_check.sh <队列文件>...；结果写到 runs/cfgcheck/。一次给多份队列（例如同一段的冒烟与正式）时，
#   同一个 parquet 的官方配方只展开一次。
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"; cd "$ROOT"
[ $# -ge 1 ] || { echo "用法：fc_config_check.sh <队列文件>..." >&2; exit 2; }
mkdir -p runs/cfgcheck; fail=0
declare -A RECIPE   # parquet → 这次已展开过的官方配方文件：冒烟与正式同一个 parquet，官方配方只展开一次（每条省一次容器启动）
for Q in "$@"; do
echo "=== 队列 $Q"
while IFS= read -r line || [ -n "$line" ]; do
  line="${line%%#*}"; [[ -z "${line//[[:space:]]/}" ]] && continue
  read -r -a tok <<< "$line"; envs=(); i=0
  while (( i < ${#tok[@]} )) && [[ ${tok[$i]} =~ ^[A-Z_][A-Z0-9_]*= ]]; do envs+=("${tok[$i]}"); i=$((i + 1)); done
  args=("${tok[@]:$i}"); run=${args[0]}; pq=${args[1]}
  echo "== $run"
  [ -f "runs/flash_chord/$run/DONE" ] && { echo "  已完成（DONE），跳过核对"; continue; }
  cmd="$(env "${envs[@]}" V2D_PRINT_CMD=1 bash stages/policy/fc_run.sh "${args[@]}" < /dev/null)" || { echo "  fc_run 打印模式失败"; fail=1; continue; }
  echo "  命令：${cmd:0:160}…"
  # training.{experiment_name,run_name,wandb_entity,wandb_project} 在配方里是 ${logging.*} 的引用，随 logging 覆盖一起变
  allowed="logging,output_dir,task.parquet,scene.world_count,training.total_environment_steps,training.experiment_name,training.run_name,training.wandb_entity,training.wandb_project"
  for o in "${args[@]:4}"; do k=${o%%=*}; k=${k#+}; k=${k#+}; allowed+=",$k"   # +键 / ++键 是 Hydra 的追加写法，按键名放行
    case "$k" in reset.*|objective.*|termination.*) allowed+=",env.$k" ;; esac
    case "$k" in curriculum.*) allowed+=",training.$k" ;; esac
    [ "$k" = reset.seed ] && allowed+=",training.seed"   # training.seed 是 ${reset.seed} 的引用（计划 agent-doc/plan-t2ft2-2026-10-26.md）   # training.curriculum 是 ${curriculum} 的引用（计划 agent-doc/plan-t2ft-2026-10-26.md 复核 2）   # 展开的配置在 env 下面也放了一份 reset、objective、termination
    #   （configs/env/rl.yaml 里是对顶层的引用），随顶层覆盖一起变；第 3 步配方试验的变体改 objective / termination 的取值
  done
  scripts/dev/in-flash-chord "$cmd --cfg job --resolve" > "runs/cfgcheck/$run.yaml" 2> "runs/cfgcheck/$run.err" \
    || { echo "  展开本次配置失败，见 runs/cfgcheck/$run.err"; fail=1; continue; }
  if [ -n "${RECIPE[$pq]:-}" ]; then
    cp "${RECIPE[$pq]}" "runs/cfgcheck/$run.recipe.yaml"; echo "  官方配方沿用 ${RECIPE[$pq]}（同一 parquet）"
  else
    scripts/dev/in-flash-chord "cd /workspace && python scripts/train_flash_sac.py experiment=g1_recon_body_flash_sac \"task.parquet='$pq'\" --cfg job --resolve" \
      > "runs/cfgcheck/$run.recipe.yaml" 2> "runs/cfgcheck/$run.recipe.err" \
      || { echo "  展开官方配方失败，见 runs/cfgcheck/$run.recipe.err"; fail=1; continue; }
    RECIPE[$pq]="runs/cfgcheck/$run.recipe.yaml"
  fi
  scripts/dev/in-flash-chord "python /repo/stages/policy/cfg_diff.py /repo/runs/cfgcheck/$run.recipe.yaml /repo/runs/cfgcheck/$run.yaml $allowed" || fail=1
done < "$Q"
done
[ "$fail" = 0 ] && echo "== 配置核对通过" || { echo "== 配置核对有问题"; exit 1; }
