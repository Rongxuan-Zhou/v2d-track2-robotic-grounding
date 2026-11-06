#!/usr/bin/env bash
# ReconBody 训练冒烟：200 迭代，确认管线能跑通，不追求效果。训练机上执行。
# 超参取官方 create_soma_task.py:98-133 的配方，只把 max_iterations 与 num_envs 调小。
# 用法: stages/policy/smoke_recon_body.sh <tier> <sequence_id> [num_envs]
set -euo pipefail
TIER=$1; SEQ=$2; NENV=${3:-1024}
cd "$(dirname "$0")/../.."
RUN=smoke_t${TIER}_${SEQ:0:40}
mkdir -p runs/train runs/logs
V2D_GPU=1 V2D_WORKDIR=/repo/runs/train scripts/dev/incontainer \
  /repo/third_party/video_to_data/robotic_grounding/scripts/rsl_rl/train.py --headless \
  --task SonicG1-ReconBody-v0 \
  --motion_file "/repo/runs/motion/tier${TIER}/whole_body/soma/sequence_id=${SEQ}/robot_name=g1" \
  --num_envs "$NENV" --max_iterations 200 --zero-actor --run_name "$RUN" \
  agent.experiment_name="recon_body_t${TIER}" \
  env.commands.motion.reset_freeze_steps=50 \
  env.commands.motion.voc_decay_steps=10 \
  env.commands.motion.voc_reset_scale=1.0 \
  env.commands.motion.initial_virtual_object_control_curriculum_scale=1.0 \
  env.commands.motion.reset_shoulder_spread=0.2 \
  env.episode_length_s=5.0 \
  > "runs/logs/${RUN}.log" 2>&1
