#!/usr/bin/env bash
# 回放速度对照（Day 1b 诊断触发条件 T2；与 Tier 2 无关）用到的上游事实：按匹配行摘出 FlashCHORD 源码与配置，带行号。
# 在训练机的仓库根执行（宿主机上读 third_party，不进容器、不占 GPU）；输出经 scripts/dev/record 落到 results/playback/。
# 摘录位置按内容匹配而不是写死行号；匹配不到就报错退出，免得上游变了还拿旧结论当证据。
# 用法（Mac）：scripts/dev/remote scripts/dev/record playback/upstream_playback_facts stages/policy/playback_facts.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
C=third_party/video_to_data/robotic_grounding/flash_chord/src/flash_chord
echo "upstream $(git -C third_party/video_to_data rev-parse --short=7 HEAD)"

show() {  # show <文件> <首行> <末行>
  echo "--- $1:$2-$3"
  awk -v a="$2" -v b="$3" 'NR >= a && NR <= b {printf "%5d  %s\n", NR, $0}' "$1"
}
blk() {  # blk <文件> <固定字符串> <往下多印几行>：从第一处匹配开始印
  local s
  s="$(grep -n -m1 -F -- "$2" "$1" | cut -d: -f1)"
  [ -n "$s" ] || { echo "--- $1：找不到“$2”" >&2; exit 1; }
  show "$1" "$s" $((s + $3))
}
lines() {  # lines <文件> <正则>：印出所有匹配行
  echo "--- $1（匹配 $2）"
  grep -n -E -- "$2" "$1" | sed -E 's/^([0-9]+):/\1  /' || { echo "  找不到" >&2; exit 1; }
}

echo "== 1. 各配方的回放方式"
P='motion_speed|source_frame_playback'
lines "$C/configs/experiment/g1_recon_body_flash_sac.yaml" "$P|total_environment_steps|capacity|save_interval"
lines "$C/configs/experiment/g1_recon_body_ppo.yaml" "$P"
lines "$C/configs/task/default.yaml" "$P"
for e in sharpa_flash_sac sharpa_ppo dexmate_sharpa_flash_sac; do lines "$C/configs/experiment/$e.yaml" "$P"; done

echo "== 2. 加载器：逐帧回放与重采样"
blk "$C/data/motion_v1.py" "if source_frame_playback and control_fps is None:" 5
lines "$C/data/motion_v1.py" 'source_fps = _positive_float'
blk "$C/data/motion_v1.py" "source_times = target_times = None" 20
lines "$C/data/motion_v1.py" '_fps=control_fps'
blk "$C/data/resampling.py" "def resample_times" 7

echo "== 3. 评测器按 checkpoint 自带配置重建参考"
blk "$C/evaluation/harness.py" "def evaluate_checkpoint(" 31
blk "$C/evaluation/harness.py" "def apply_evaluation_composition(" 35   # 评测组合只改 scene / env / termination，不碰 task 的回放设置
blk "$C/evaluation/harness.py" "setup = setup_scene(" 16
lines "$C/evaluation/harness.py" 'step_count = setup.reference.num_frames'

echo "== 4. curriculum 阈值是绝对步数；reset 与种子"
blk "$C/configs/curriculum/recon_body_flash_sac.yaml" "# Environment transitions" 4
lines "$C/configs/curriculum/recon_body_flash_sac.yaml" 'voc_scale|immediate_first_frame_probability'
lines "$C/configs/reset/recon_body.yaml" 'reset_freeze_steps|voc_decay_steps|seed'
lines "$C/configs/training/reference_flash_sac.yaml" 'seed'
lines stages/policy/fc_train.sh 'reset.seed'

echo "== 5. 观测里的参考预览按控制步计"
lines "$C/envs/recon_body.py" 'RECON_BODY_FUTURE_OFFSETS =|future_frame = wp.min'
lines "$C/envs/base.py" 'fps'

echo "== 6. 上游里有没有比赛评测脚本"
echo "git ls-files 里含 eval_robotic_grounding 的文件数：$(git -C third_party/video_to_data ls-files | grep -c eval_robotic_grounding || true)"

echo "== 7. 训练入口按配置传回放方式"
lines third_party/video_to_data/robotic_grounding/flash_chord/scripts/train_flash_sac.py 'source_frame_playback=|motion_speed='

echo "== 8. G1 绑定参考时的 palm 校验：逐帧、不区分插值帧（Sharpa 对插值帧另有宽容差）"
lines "$C/embodiments/g1_dex3.py" 'frame_position_tolerance|frame_orientation_tolerance'
blk "$C/embodiments/g1_dex3.py" "def _validate_palm_frames" 28
lines "$C/embodiments/vega_sharpa.py" 'interpolated_frame_(position|orientation)_tolerance: float'
blk "$C/embodiments/vega_sharpa.py" "if reference.metadata.is_resampled:" 3
