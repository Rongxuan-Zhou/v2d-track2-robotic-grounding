#!/usr/bin/env bash
# t3takeover 第 1 级的复制（训练机上执行；必须经 stages/recovery/t3r_run.sh 在“她的目录只读”的命名空间里跑）。计划 agent-doc/plan-t3takeover-v3-2026-10-21.md 第 1 节第 3 条、判据 K0。
# 读清单（stages/recovery/t3r_level1_files.tsv：类别、源、目标），源相对于 T3R_SRC（另一位成员的 v2d 目录，运行时由调用方给，仓库里不写），目标相对于 T3R_ROOT。
# 1. 先查：所有源都是普通文件（软链接取它指向的文件），目标都还不存在；缺一个或已有一个就停（退出码 2），一个都不复制。
# 2. 逐个 cp --dereference --no-preserve=all 到 <目标>.partial、再改名；源与目标各算一次 sha256，不同就停（退出码 3）。
# 3. 每行打 “类别 sha256 字节数 目标”，最后一行 T3R_COPY_DONE <文件数> <总字节数>。
# 用法：T3R_HER=<她的目录> T3R_ROOT=<我们的根> stages/recovery/t3r_run.sh /bin/bash <仓库>/stages/recovery/t3r_copy.sh <仓库>/stages/recovery/t3r_level1_files.tsv
#       （T3R_SRC 由调用方放进环境，不出现在命令行里）
set -euo pipefail
LIST=${1:?用法见文件头}
SRC=${T3R_SRC:?要设 T3R_SRC}; ROOT=${T3R_ROOT:?要设 T3R_ROOT}
[ ! -w "$SRC" ] || { echo "T3R_COPY_REFUSED 源目录可写：不在只读命名空间里"; exit 2; }
mapfile -t ROWS < <(grep -v '^#' "$LIST" | grep -v '^[[:space:]]*$')
missing=0
for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r kind rel dst <<< "$row"
  case "$rel$dst" in *..*) echo "路径含 ..：${rel} ${dst}"; missing=$((missing + 1)); continue ;; esac
  case "$dst" in /*) echo "目标是绝对路径：${dst}"; missing=$((missing + 1)); continue ;; esac
  [ -f "$SRC/$rel" ] || { echo "缺源：${kind} ${rel}"; missing=$((missing + 1)); }
  [ ! -e "$ROOT/$dst" ] || { echo "目标已存在：${dst}"; missing=$((missing + 1)); }
done
[ "$missing" = 0 ] || { echo "T3R_COPY_FAILED 先查不过（${missing} 处）"; exit 2; }
n=0; bytes=0
for row in "${ROWS[@]}"; do
  IFS=$'\t' read -r kind rel dst <<< "$row"
  mkdir -p "$(dirname "$ROOT/$dst")"
  cp --dereference --no-preserve=all "$SRC/$rel" "$ROOT/$dst.partial"
  mv "$ROOT/$dst.partial" "$ROOT/$dst"
  a="$(sha256sum "$SRC/$rel" | cut -c1-64)"; b="$(sha256sum "$ROOT/$dst" | cut -c1-64)"
  [ "$a" = "$b" ] || { echo "T3R_COPY_FAILED sha256 不同：${dst}"; exit 3; }
  s="$(stat -c %s "$ROOT/$dst")"
  echo "${kind} ${b} ${s} ${dst}"
  n=$((n + 1)); bytes=$((bytes + s))
done
echo "T3R_COPY_DONE ${n} ${bytes}"
