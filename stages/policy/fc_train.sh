#!/usr/bin/env bash
# FlashCHORD G1 训练（在 FlashCHORD 容器内执行，经 scripts/dev/in-flash-chord 调用）。
# 用法：stages/policy/fc_train.sh <parquet> <world_count> <total_env_steps> <输出目录> [额外 hydra 覆盖...]
# parquet 路径含 '='（sequence_id=/robot_name= 分区），Hydra 要求整体加单引号，写在脚本里免得多层转义吃掉。
# 默认值只在调用方没给时补上：logging.mode=disabled（W&B 由 fc_run.sh 打开），
# hydra.run.dir 放进输出目录（否则 Hydra 往上游 flash_chord/logs/ 里写 .hydra）。
set -euo pipefail
PARQ=$1; W=$2; STEPS=$3; OUT=$4; shift 4
has() { local k=$1; shift; for a in "$@"; do [[ $a == "$k="* ]] && return 0; done; return 1; }
DEFAULTS=()
has logging.mode "$@" || DEFAULTS+=(logging.mode=disabled)
has hydra.run.dir "$@" || DEFAULTS+=("hydra.run.dir=${OUT}/hydra/$(date +%Y%m%d-%H%M%S)")
has reset.seed "$@" || DEFAULTS+=(reset.seed=42)   # 续训换种子用（计划 agent-doc/plan-t2ft2-2026-10-26.md）；不给时与原来相同
cd /workspace
exec python scripts/train_flash_sac.py experiment=g1_recon_body_flash_sac \
  "task.parquet='${PARQ}'" "scene.world_count=${W}" "training.total_environment_steps=${STEPS}" \
  "output_dir=${OUT}" "${DEFAULTS[@]}" "$@"
