#!/usr/bin/env bash
# 比较上游三个版本的 scene/support.py：训练镜像用的 v0.4.0、官方 kit 指定的评测运行时 aee95f2、修了方盒缩放的 8c778f905c（#185）。
# 每个版本打 md5，并列出 Cube 尺寸相关的行（读 size、读 xformOp:scale、算半尺寸），供 jev 与 SOP 页引用（Mac 上执行，只读，经 gh api）。
# 计划 agent-doc/plan-ep1-ep7-divergence-2026-09-30.md 附录 A.0（“aee95f2 里 support.py 与 v0.4.0 相同”原先只在复审里口头核对，这里落盘）。
# 用法（仓库根）：scripts/dev/record diverge/support_py_versions stages/policy/support_py_versions.sh
set -uo pipefail
F=robotic_grounding/flash_chord/src/flash_chord/scene/support.py
get() { gh api "repos/nvidia-isaac/video_to_data/contents/$F?ref=$1" --jq .content | base64 -d; }
md5of() { if command -v md5sum >/dev/null; then md5sum | cut -d' ' -f1; else md5 -q; fi; }
for ref in v0.4.0 aee95f2 8c778f905c; do
  sum="$(get "$ref" | md5of)" || { echo "取不到 $ref"; exit 1; }   # 直接对原始字节算，不经 shell 变量（会吃掉结尾换行）
  echo "== $ref $F  md5 $sum"
  get "$ref" | grep -nE 'Cube|size|[Ss]cale|half' | sed "s|^|  support.py:|"
done
