#!/usr/bin/env bash
# ep7 方盒正式训练（#185 几何）发散相关的上游代码摘录（Mac 上执行，只读，经 gh api 读训练所用的 v0.4.0）。每行带文件与行号，供 jev 核对。
# 训练挂的是 v0.4.0 + #185 补丁副本；#185 只改 scene/support.py 与它的测试，这里摘的文件与 v0.4.0 相同。
# divergence_code_facts.sh（9/30）查的是 BatchNorm 统计量、loss scale 与回放精度；这里补 10/2 分析要用的几处：
#   奖励归一化器怎么更新、有没有非有限过滤；采集一步时奖励与回放缓冲的写入顺序；ReconBody 终止判定里的比较与训练、评测各用的阈值；
#   训练端与评测端对非有限世界的处理；每步更新次数、批大小、n-step；课程各段的 VOC；几个 W&B 指标的定义。
# 用法（仓库根）：scripts/dev/record ep7/code_facts stages/policy/ep7_code_facts.sh
set -uo pipefail
REF=v0.4.0
B=robotic_grounding/flash_chord/src/flash_chord
get() { gh api "repos/nvidia-isaac/video_to_data/contents/$B/$1?ref=$REF" --jq .content | base64 -d; }
g() {  # g <文件> <正则>：带行号列出匹配行
  echo "== $REF $1"
  get "$1" | grep -nE "$2" | sed "s|^|$1:|"
}
count() {  # count <文件> <正则>：只报匹配行数
  echo "$1：匹配 /$2/ 的行数 $(get "$1" | grep -cE "$2")"
}
ctx() {  # ctx <文件> <sed 行号范围>：带行号的原文段落
  echo "== $REF $1（原文段落 $2）"
  get "$1" | awk -v f="$1" '{printf "%s:%d:%s\n", f, NR, $0}' | sed -n "$2"
}

echo "## 1. 奖励归一化器：每个世界的折扣回报、全局最大值与滑动矩，更新处有无非有限过滤"
ctx training/flash_sac/reward_normalization.py '50,75p;97,112p;114,127p'
count training/flash_sac/reward_normalization.py 'jnp.isfinite|nan_to_num|jnp.where'

echo "## 2. 采集一步：先用这一步的奖励更新归一化器，再把这一步写进回放缓冲"
ctx training/flash_sac/learner.py '1060,1086p'
count training/flash_sac/learner.py 'jnp.isfinite|nan_to_num'
g training/flash_sac/update.py 'isfinite'

echo "## 3. ReconBody 终止判定：误差与阈值直接比较，状态本身不查是否有限"
ctx lifecycle/termination.py '601,605p;637,652p'
count lifecycle/termination.py 'isfinite\(|isnan\('
g lifecycle/termination.py 'isfinite\(|isnan\('

echo "## 3b. 训练用的终止阈值（G1 配方选 termination: recon_body）与评测时改成的阈值"
g configs/experiment/g1_recon_body_flash_sac.yaml 'override /termination'
echo "== $REF configs/termination/recon_body.yaml"
get configs/termination/recon_body.yaml | grep -nE 'threshold|reset_freeze_steps|truncate' | sed 's|^|configs/termination/recon_body.yaml:|'
g evaluation/harness.py '1\.0e6|1e6|_threshold'

echo "## 4. 训练端的环境代码里没有非有限世界的检测；评测端有"
for f in envs/base.py envs/recon_body.py envs/observation.py lifecycle/reset.py objectives/recon_body.py; do count "$f" 'non_finite|isnan\(|jnp.isfinite|wp.isfinite|np.isfinite\(.*body_q|nan_to_num'; done
g evaluation/rollout.py 'isfinite|non_finite'

echo "## 5. 更新与回放的配置"
g configs/training/reference_flash_sac.yaml 'updates_per_collection|batch_size|n_step|capacity|save_replay|sampling:'
g configs/experiment/g1_recon_body_flash_sac.yaml 'updates_per_collection|capacity|n_step|minimum_size|observation_storage_dtype|log_interval|save_interval_environment_steps|override /curriculum'

echo "## 6. 课程：各段的起点与 VOC"
echo "== $REF configs/curriculum/recon_body_flash_sac.yaml"
get configs/curriculum/recon_body_flash_sac.yaml | grep -nE 'start|voc_scale|environment_steps|fraction|first_frame|name:' | sed 's|^|configs/curriculum/recon_body_flash_sac.yaml:|'

echo "## 7. W&B 指标的定义"
g training/flash_sac/runner.py 'Mean across logging-window world-steps|action_abs_mean_per_env_action|critic_skipped_fraction|scale_denominator|assigned exclusively'
