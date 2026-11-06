#!/usr/bin/env bash
# 把 ml40 上同组成员 Tier 3 重建产物里的几集转到 mf4090，拼进档目录 runs/tier3/root（Mac 上、仓库根执行，Mac 只转发）。
# 计划 agent-doc/plan-t3-pilot-2026-10-09.md 第 3 节。
#   0. 哈希表 stages/recovery/expect/<标签>.tsv 要已提交并同步（stages/recovery/tier3_expect.py 从两份只读核对记录生成）
#   1. mf4090 上建 runs/tier3/incoming/<标签>.partial/（它或 <标签>/ 已存在就拒绝：同一标签只转一次）
#   2. ssh ml40 经 stdin 跑 stages/recovery/tier3_pack_remote.sh（自己找目录，不出现别人的名字），tar 流经 Mac 管道在 mf4090 上解开；
#      失败就删掉 .partial（审查 m3），成功才改名为 <标签>/
#   3. mf4090 上列出解出文件的 sha256 与部署版本（只打 40 位 commit，审查 M2），再跑 stages/recovery/tier3_root.py --expect <哈希表>
#      （逐个比对，不符就什么也不写，审查 M1）
# 用法：stages/recovery/tier3_stage.sh <标签> <集>...      例：stages/recovery/tier3_stage.sh t3p_1009 8 24
#   落盘：scripts/dev/record t3p/stage_mf4090 stages/recovery/tier3_stage.sh t3p_1009 8 24
# 校正版（计划 agent-doc/plan-t3-pilot-v2-2026-10-10.md）：V2D_T3_SUBDIR=<同组成员那边的目录> V2D_T3_SET=<名> tier3_stage.sh <标签> <集>...
#   V2D_T3_SUBDIR 传给 ml40 上的打包（白名单在 tier3_pack_remote.sh），V2D_T3_SET 传给 mf4090 上的 tier3_root.py，档目录写到
#   runs/tier3/root_<名>（libs/v2d_infra/paths.py）。两者都不设时与 10/9 相同。
# V2D_T3_SUPPORT=delivery（只许空或 delivery，计划 agent-doc/plan-t3-hm-p2-2026-10-12.md）：打包带上交付目录里每集的支撑面，
#   tier3_root.py 加 --support-from-incoming（支撑面从转来的目录复制，按哈希表的 sha256 核）；哈希表要用 tier3_expect.py --support-from-remote 生成。
#   不设时与以前相同。
# V2D_T3_SUPPORT_DIR=runs/tier3/root_<名>（计划 agent-doc/plan-t3dyn-pilot-2026-10-14.md）：转来的目录里不带支撑面（V2D_T3_SUPPORT 不设），
#   tier3_root.py 加 --support-dir <目录>，支撑面从 mf4090 上这个已拼好的档目录复制（例如 root_hm），按哈希表的 sha256 核；
#   哈希表要用 tier3_expect.py --support-record 生成。与 V2D_T3_SUPPORT=delivery 只能设一个。
set -euo pipefail
cd "$(dirname "$0")/../.."
TAG=${1:?用法：tier3_stage.sh <标签> <集>...}; shift
[ $# -ge 1 ] || { echo "tier3_stage: 要至少一集" >&2; exit 2; }
[[ $TAG =~ ^[A-Za-z0-9_-]+$ ]] || { echo "tier3_stage: 标签非法：$TAG" >&2; exit 2; }
for ep in "$@"; do [[ $ep =~ ^[0-9]+$ ]] || { echo "tier3_stage: 集号非法：$ep" >&2; exit 2; }; done
SUB=${V2D_T3_SUBDIR:-tier3_cari4d}; SET=${V2D_T3_SET:-}
[[ $SUB =~ ^tier3_cari4d[a-z0-9_]*$ ]] || { echo "tier3_stage: V2D_T3_SUBDIR 非法：$SUB" >&2; exit 2; }
[[ $SET =~ ^[a-z0-9]*$ ]] || { echo "tier3_stage: V2D_T3_SET 只许小写字母与数字：$SET" >&2; exit 2; }
SUP=${V2D_T3_SUPPORT:-}
[[ $SUP =~ ^(delivery)?$ ]] || { echo "tier3_stage: V2D_T3_SUPPORT 只许空或 delivery：$SUP" >&2; exit 2; }
SDIR=${V2D_T3_SUPPORT_DIR:-}
[[ $SDIR =~ ^(runs/tier3/root_[a-z0-9]+)?$ ]] || { echo "tier3_stage: V2D_T3_SUPPORT_DIR 只许空或 runs/tier3/root_<名>：$SDIR" >&2; exit 2; }
[ -z "$SDIR" ] || [ "$SUP" != delivery ] || { echo "tier3_stage: V2D_T3_SUPPORT=delivery 与 V2D_T3_SUPPORT_DIR 只能设一个" >&2; exit 2; }
SFLAG=""; SNOTE=""
if [ "$SUP" = delivery ]; then SFLAG=" --support-from-incoming"; SNOTE="；V2D_T3_SUPPORT=delivery"; fi
if [ -n "$SDIR" ]; then SFLAG=" --support-dir $SDIR"; SNOTE="；V2D_T3_SUPPORT_DIR=$SDIR"; fi
R='~/v2d/maniphys_lab_v2d_challenge'; IN="runs/tier3/incoming/$TAG"; EXP="stages/recovery/expect/$TAG.tsv"
[ -f "$EXP" ] || { echo "tier3_stage: 缺哈希表 ${EXP}（先用 stages/recovery/tier3_expect.py 生成、提交并同步）" >&2; exit 1; }
ssh -o BatchMode=yes mf4090 "cd $R && [ -f $EXP ] && [ ! -e $IN ] && [ ! -e $IN.partial ] && mkdir -p $IN.partial" \
  || { echo "tier3_stage: mf4090 上缺 ${EXP}（没同步？），或 $IN 已存在，或建不了目录" >&2; exit 1; }
if ! ssh -o BatchMode=yes ml40 "V2D_T3_SUBDIR=$SUB V2D_T3_SUPPORT=$SUP bash -s -- $*" < stages/recovery/tier3_pack_remote.sh | ssh -o BatchMode=yes mf4090 "cd $R && tar -xf - -C $IN.partial"; then
  ssh -o BatchMode=yes mf4090 "cd $R && rm -rf $IN.partial"
  echo "tier3_stage: 转发失败，已删 mf4090 上的 $IN.partial" >&2; exit 1
fi
ssh -o BatchMode=yes mf4090 "cd $R && mv $IN.partial $IN && echo \"== \$(hostname) \$(date '+%F %T %Z') 来历 \$(cat .v2d-source-commit)\" && (cd $IN && find . -type f | sort | xargs sha256sum) && echo \"部署版本 \$(grep -oE '^[0-9a-f]{40}' $IN/DEPLOYED)；同组成员那边的目录 runs/${SUB}；V2D_T3_SET=${SET:-（不设）}${SNOTE}\" && V2D_T3_SET=$SET python3 stages/recovery/tier3_root.py $IN --expect $EXP$SFLAG"
