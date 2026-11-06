#!/usr/bin/env bash
# 评测器成功率的定义与门槛原文（Mac 上执行，只读，经 gh api 读训练镜像所用的 flash_chord v0.4.0）。每行带文件与行号。
# SP-SR = spider_success，MP-SR = maniptrans_success：都是每个世界先对时间取平均误差，再与门槛比。
# 用法（仓库根）：scripts/dev/record batch1/metric_threshold_facts stages/evaluate/metric_threshold_facts.sh
set -uo pipefail
F=robotic_grounding/flash_chord/src/flash_chord/evaluation/metrics.py
gh api "repos/nvidia-isaac/video_to_data/contents/$F?ref=v0.4.0" --jq .content | base64 -d \
  | awk '{printf "evaluation/metrics.py:%d:%s\n", NR, $0}' | sed -n '21,24p;296,317p;327,349p;385,403p'
