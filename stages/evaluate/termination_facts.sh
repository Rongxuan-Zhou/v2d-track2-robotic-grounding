#!/usr/bin/env bash
# FlashCHORD 评测器遇到非有限世界、终止世界时怎么处理（Mac 上执行，只读，经 gh api 读上游源码）。
# 对比训练镜像用的 v0.4.0 与 kit 要求的 aee95f2：rollout.py 剔除非有限位姿时 completion 是否一起剔除，
# harness.py 在什么条件下报 "evaluation must disable task failure termination"。每行带版本、文件与行号，供 jev 核对。
# 用法（仓库根）：scripts/dev/record batch1/termination_facts stages/evaluate/termination_facts.sh
set -uo pipefail
P=robotic_grounding/flash_chord/src/flash_chord/evaluation
for ref in v0.4.0 aee95f25ae0943f335d8bb42fa95b6a9acfa266f; do
  for f in rollout.py harness.py; do
    echo "== $ref $P/$f"
    gh api "repos/nvidia-isaac/video_to_data/contents/$P/$f?ref=$ref" --jq .content | base64 -d \
      | grep -nE 'non_finite = tuple|achieved\[:, ~finite\]|recorded non-finite poses|terminated=completed_terminated|def score_reference_tracking|completion.terminated.any|disable task failure termination' \
      | sed "s|^|$ref $f:|"
  done
done
