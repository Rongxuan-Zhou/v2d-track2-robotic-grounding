#!/usr/bin/env bash
# t3static：对给定的集做 stand_still.py 的 dry-run（只读，不写文件），报每集第 0 帧机器人离物体的最近距离（计划 agent-doc/plan-t3static-2026-10-21.md 第 2 节 S2 的试点选集）。
# 用法（仓库根，训练机）：bash stages/retarget/stand_still_dryrun.sh ep=/repo/runs/motion/<档位目录>/.../data.parquet …   最后一行 STAND_DRYRUN_DONE
set -uo pipefail
for a in "$@"; do
  ep=${a%%=*}; p=${a#*=}
  src=$(echo "$p" | sed -E 's#/repo/runs/motion/([^/]+)/.*#/repo/runs/motion_src/\1#')
  echo "== ep$ep"
  PYTHONDONTWRITEBYTECODE=1 scripts/dev/in-flash-chord python /repo/stages/retarget/stand_still.py "$p" /tmp/stand_dry --meta-src "$src" --meta-dst /tmp/stand_dry_meta --dry-run 2>&1 | grep -E "自检|最近|FAIL|没有|拒绝|Error"
done
echo STAND_DRYRUN_DONE
