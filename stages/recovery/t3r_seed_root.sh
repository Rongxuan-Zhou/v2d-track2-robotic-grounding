#!/usr/bin/env bash
# t3takeover 第 1 级：从第一个根 t3 种一个新根（训练机 ml40 上执行；只在我们的目录里读写，不碰另一位成员的目录）。计划 agent-doc/plan-t3takeover-v3-2026-10-21.md 判据 K1、K2 登记第 3 条。
# 两种：
#   inputs        只复制输入（stages/recovery/t3r_level1_files.tsv 里类别 in、目标在 t3/ 下的 30 行），给正对照用：在新根里从 decode 起整条重跑；
#   world-from-ref 输入之外，再把参照里三集转换目录的 decode、focal、world 产物复制进新根（ref/runs/<转换目录>/<集名>/ 下的全部文件），
#                  给 K2 登记第 3 条用：T3R_FROM=assemble stages/recovery/t3r_level1.sh <新根> 从 assemble 往后跑，看 ep27 的下游段能否逐字节复现。
# 每个文件 cp 后核 sha256，不同就停（退出码 3）；新根已存在就拒绝（退出码 2）。最后一行 T3R_SEED_DONE <文件数>。
# 用法：stages/recovery/t3r_seed_root.sh <新根名> <inputs|world-from-ref>
set -euo pipefail
cd "$(dirname "$0")/../.."
NAME=${1:?新根名}; KIND=${2:?inputs 或 world-from-ref}
R=/lab_data/rxzhou/v2d/t3recon
[[ $NAME =~ ^t3_[a-z0-9]+$ ]] || { echo "T3R_SEED_FAILED 新根名要像 t3_rep"; exit 2; }
[ -e "$R/$NAME" ] && { echo "T3R_SEED_FAILED $R/$NAME 已存在"; exit 2; }
case $KIND in inputs|world-from-ref) ;; *) echo "T3R_SEED_FAILED 不认识 $KIND"; exit 2 ;; esac
n=0
put() {   # put <源（绝对路径）> <目标（相对于新根）>
  local src=$1 dst="$R/$NAME/$2"
  [ -f "$src" ] && [ ! -L "$src" ] || { echo "T3R_SEED_FAILED 源不是普通文件：$src"; exit 2; }
  mkdir -p "$(dirname "$dst")"
  cp --no-preserve=all "$src" "$dst"
  [ "$(sha256sum < "$src")" = "$(sha256sum < "$dst")" ] || { echo "T3R_SEED_FAILED sha256 不同：$2"; exit 3; }
  n=$((n + 1))
}
while IFS=$'\t' read -r kind _ dst; do
  if [ "$kind" = in ] && [[ $dst == t3/* ]]; then put "$R/$dst" "${dst#t3/}"; fi
done < stages/recovery/t3r_level1_files.tsv
if [ "$KIND" = world-from-ref ]; then
  for d in convert_gb5/ep14_round_table convert_gb5/ep20_rice_cooker convert_fp_gb5/ep27_woven_basket; do
    for f in "$R/ref/runs/$d"/*; do
      put "$f" "runs/$d/$(basename "$f")"
    done
  done
fi
echo "T3R_SEED_DONE $n"
