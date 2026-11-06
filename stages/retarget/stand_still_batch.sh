#!/usr/bin/env bash
# t3static：按给定的集写站立参考到 runs/motion/tier3_st（计划 agent-doc/plan-t3static-2026-10-21.md 第 1 节第 1 步）。
# 每集从它在 t3m1022a 里那条 run 用的参考版本生成（参数直接给那份 parquet），v2d_meta.json 写到 runs/motion_src/tier3_st；已有的集拒绝（stand_still.py 自己查）。
# 写完对每集打出源 parquet 与新 parquet 的 sha256，供核对。
# 用法（仓库根，训练机）：bash stages/retarget/stand_still_batch.sh ep=/repo/runs/motion/<档位目录>/.../data.parquet …   最后一行 STAND_BATCH_DONE 或 STAND_BATCH_FAILED <集>
set -uo pipefail
bad=()
for a in "$@"; do
  ep=${a%%=*}; p=${a#*=}
  src=$(echo "$p" | sed -E 's#/repo/runs/motion/([^/]+)/.*#/repo/runs/motion_src/\1#')
  echo "== ep$ep 源 $p"
  out=$(PYTHONDONTWRITEBYTECODE=1 scripts/dev/in-flash-chord python /repo/stages/retarget/stand_still.py "$p" /repo/runs/motion/tier3_st \
        --meta-src "$src" --meta-dst /repo/runs/motion_src/tier3_st 2>&1)
  echo "$out" | grep -vE "^\s*$" | tail -n 6
  if [ "$(echo "$out" | tail -n 1)" != STAND_STILL_DONE ]; then bad+=("$ep"); continue; fi
  rel=${p#/repo/}; new=runs/motion/tier3_st/${rel#runs/motion/*/}
  echo "  sha256 源 $(sha256sum "$rel" | cut -c1-16)  新 $(sha256sum "$new" | cut -c1-16)  $new"
done
[ ${#bad[@]} -eq 0 ] && echo STAND_BATCH_DONE || echo "STAND_BATCH_FAILED ${bad[*]}"
