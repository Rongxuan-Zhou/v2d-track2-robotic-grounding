#!/usr/bin/env bash
# 在一台机器上按 sha256 找丢了路径的 checkpoint（只读）。stdin 不用；参数是一个 TSV（sha256、run 名）和若干搜索根目录。
# 对每个 run 名，在搜索根下找路径里含这个 run 名的 *.safetensors、*.pt、*.tar*，逐个算 sha256；
# 打印 MATCH（sha 相同）、OTHER（找到同名 run 的文件但 sha 不同）；每个 run 最后一行 RUN_DONE 加找到的文件数。
# 用法（Mac 上）：ssh <机器> 'bash -s <TSV> <根1> [根2 …]' < tools/find_ckpt_remote.sh
set -u
list=$1; shift
while IFS=$'\t' read -r sha run; do
  [ -n "$sha" ] || continue
  n=0
  while IFS= read -r f; do
    n=$((n + 1))
    case "$f" in
      *.tar*) echo "ARCHIVE	$run	$f	$(stat -c %s "$f") B";;
      *) got=$(sha256sum "$f" | cut -c1-64)
         if [ "$got" = "$sha" ]; then echo "MATCH	$run	$f"; else echo "OTHER	$run	$f	$got"; fi;;
    esac
  done < <(find "$@" -xdev \( -name '*.safetensors' -o -name '*.pt' -o -name '*.tar' -o -name '*.tar.*' \) -path "*${run}*" 2>/dev/null)
  echo "RUN_DONE	$run	找到 $n 个"
done < "$list"
echo FIND_CKPT_DONE
