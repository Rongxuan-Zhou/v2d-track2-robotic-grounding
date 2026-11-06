#!/usr/bin/env bash
# 在训练机上逐个核对计分提交用到的 checkpoint（tools/counted_inventory.py 写的 runs/inventory/ckpt_<机器>.tsv：sha256、容器内路径）。
# 容器内路径 /repo/ 换成仓库根；每个文件打一行 OK / MISMATCH / MISSING，最后一行 CKPT_CHECK_DONE 加三类计数。只读。
# 用法（Mac 上）：ssh <机器> 'cd <仓库根> && bash -s runs/inventory/ckpt_<机器>.tsv' < tools/ckpt_check_remote.sh
set -u
list=$1
ok=0; bad=0; miss=0
while IFS=$'\t' read -r sha path; do
  [ -n "$sha" ] || continue
  local_path=${path#/repo/}
  if [ ! -f "$local_path" ]; then
    echo "MISSING	$local_path"; miss=$((miss + 1)); continue
  fi
  got=$(sha256sum "$local_path" | cut -c1-64)
  if [ "$got" = "$sha" ]; then
    echo "OK	$local_path"; ok=$((ok + 1))
  else
    echo "MISMATCH	$local_path	记录 $sha	实际 $got"; bad=$((bad + 1))
  fi
done < "$list"
echo "CKPT_CHECK_DONE OK $ok MISMATCH $bad MISSING $miss"
