#!/usr/bin/env bash
# ep1、ep7 训练发散相关的上游代码摘录（Mac 上执行，只读，经 gh api 读训练镜像所用的 v0.4.0）。每行带文件与行号，供 jev 核对。
# 计划 agent-doc/plan-ep1-ep7-divergence-2026-09-30.md 的 C4（计划里暂名 support_code_facts.sh，内容比支撑面宽，改名）。
# 查：支撑面的方盒与圆柱怎么建；BatchNorm 滑动统计怎么更新、有没有非有限过滤；跳过更新时统计量是否照写；loss scale 的回退；
#     回放缓冲的观测存储精度；观测、终止、重置里有没有截断或非有限处理；G1 配方的精度设置；观测各块的排列。
# 用法（仓库根）：scripts/dev/record diverge/code_facts stages/policy/divergence_code_facts.sh
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
ctx() {  # ctx <文件> <sed 行号范围，如 226,245p;625,647p>：带行号的原文段落
  echo "== $REF $1（原文段落 $2）"
  get "$1" | awk -v f="$1" '{printf "%s:%d:%s\n", f, NR, $0}' | sed -n "$2"
}

echo "## 1. 支撑面：USDA 里的 Cylinder / Cube 各建成什么碰撞体"
g scene/support.py 'type_name not in|"Cylinder"|Cube: USD|add_shape_cylinder|add_shape_box|half_height|hx=half'

echo "## 2. BatchNorm 滑动统计：更新式，有无非有限过滤"
g training/flash_sac/network.py 'class UnitBatchNorm|class EnsembleUnitBatchNorm|update_rate: float|running_mean.value = |running_variance.value = |rsqrt\(variance'
count training/flash_sac/network.py 'isfinite|nan_to_num|where\(jnp.isfinite'
g training/flash_sac/network.py 'isfinite|nan_to_num'

echo "## 3. 梯度非有限时：参数跳过更新，统计量照写"
g training/flash_sac/learner.py 'def _apply_adam_if_finite|= gradients_are_finite\(|_apply_adam_if_finite\(|batch_stats=(actor|critic)_batch_stats|batch_stats=target_mutable'
ctx training/flash_sac/learner.py '226,245p;625,647p;759,787p'

echo "## 4. loss scale：有非有限梯度就按系数回退"
g training/flash_sac/update.py 'jnp.all\(jnp.isfinite\(gradient\)\)|def update_loss_scale|finite_scale =|scale = jnp.where\(finite'

echo "## 5. 回放缓冲：观测按存储精度写入，采样时转回 float32"
g training/flash_sac/replay.py 'observation_storage_dtype: str =|astype\(pending.observation.dtype\)|astype\(pending.next_observation.dtype\)|observation=storage.observation\[indices\]|next_observation=storage.next_observation\[indices\]'

echo "## 6. 观测、终止、重置里的截断与非有限处理（isfinite 只出现在配置校验里）"
for f in envs/observation.py envs/recon_body.py lifecycle/termination.py lifecycle/reset.py; do count "$f" 'clip\(|nan_to_num|isnan'; done
g lifecycle/termination.py 'isfinite'
g lifecycle/reset.py 'isfinite'

echo "## 7. 配方的精度设置"
g configs/experiment/g1_recon_body_flash_sac.yaml 'observation_storage_dtype|override /training|override /observation'
g configs/training/reference_flash_sac.yaml 'mixed_precision|compute_dtype|batch_norm_update_rate|loss_scale_|observation_storage_dtype'

echo "## 8. ReconBody 观测的排列（486 维）"
echo "== $REF envs/recon_body.py"
get envs/recon_body.py | sed -n '22,42p' | nl -ba -v22 | sed 's|^|envs/recon_body.py:|'
