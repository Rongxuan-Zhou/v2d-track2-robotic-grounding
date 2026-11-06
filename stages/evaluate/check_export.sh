#!/usr/bin/env bash
# 跑 stages/evaluate/check_export.py（训练机上、仓库根执行）：宿主机 python3 能 import numpy 与 pyarrow.parquet 就在宿主机跑，
# 不能就在一次性 FlashCHORD 容器里跑（镜像里有 numpy、pyarrow；仓库挂在 /repo，脚本按自己的位置找仓库根，路径照样对）。
# 10/1 实测：mf4090 宿主机有这两个库，ml40 宿主机没有 numpy（results/box/tests_ml40.txt）。计划 agent-doc/plan-box6-and-eval-runtime-2026-10-01.md 第 3.2 节。
# 用法：stages/evaluate/check_export.sh <run> <步数> <世界数> [--no-traj]   退出码与输出同 check_export.py（最后一行 EXPORT_OK / EXPORT_FAIL n）
#       stages/evaluate/check_export.sh --probe   能跑（宿主机有库，或有 flash_chord:latest 镜像）就退出码 0，打印走哪条路
# V2D_COMPAT_PY_IMPORT：探测宿主机库时 import 的语句，只给测试换。
set -uo pipefail
cd "$(dirname "$0")/../.."
host_ok() { python3 -c "${V2D_COMPAT_PY_IMPORT:-import numpy, pyarrow.parquet}" >/dev/null 2>&1; }
if [ "${1:-}" = --probe ]; then
  if host_ok; then echo "check_export：宿主机 python3"; exit 0; fi
  docker image inspect flash_chord:latest >/dev/null 2>&1 && { echo "check_export：宿主机缺库，走 flash_chord:latest 容器"; exit 0; }
  echo "check_export：宿主机缺 numpy / pyarrow，也没有 flash_chord:latest 镜像"; exit 1
fi
if host_ok; then exec python3 stages/evaluate/check_export.py "$@"; fi
exec env V2D_GPU=0 scripts/dev/in-flash-chord "python /repo/stages/evaluate/check_export.py $(printf '%q ' "$@")"
