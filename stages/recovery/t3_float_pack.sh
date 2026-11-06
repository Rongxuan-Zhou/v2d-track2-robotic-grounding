#!/usr/bin/env bash
# 把同组成员某一版 Tier 3 重建里的几集原样转到 mf4090 的 runs/tier3/incoming/<标签>/ok，只给 t3_float_facts.py 做只读分析
# （Mac 上、仓库根执行，Mac 只转发）。不进 Tier 3 的档目录，不参与训练。
# 计划 agent-doc/plan-t3-pilot-v2-2026-10-10.md 第 2 节 A 段；做法同 10/9 的 results/t3float/pack_ml40_to_mf4090.txt。
#   1. mf4090 上建 <标签>/ok.partial（它或 <标签>/ok 已存在就拒绝：同一标签只转一次）
#   2. ssh ml40 经 stdin 跑 stages/recovery/tier3_pack_remote.sh（V2D_T3_SUBDIR=<版本目录>，白名单在那边），tar 流经 Mac 管道在 mf4090 上解开；
#      失败就删掉 .partial，成功才改名为 ok
#   3. 打印 parquet 与元数据的个数、部署版本（只打 40 位 commit）、每个 parquet 的修改时间（UTC）与 sha256 前 12 位
# V2D_T3_SUPPORT=delivery（只许空或 delivery，计划 agent-doc/plan-t3-hm-p2-2026-10-12.md）：原样传给 ml40 上的打包，交付目录里每集的支撑面
#   一起转来（support_surface/），第 3 步另打支撑面的个数与每个文件的 sha256 前 12 位（t3_sha_match.py --pack 拿它与 B.1 比）。不设时与以前相同。
# 用法：scripts/dev/record t3float/pack_1010_<版> stages/recovery/t3_float_pack.sh <标签> <版本目录> <集>...
#   例：scripts/dev/record t3float/pack_1010_floor stages/recovery/t3_float_pack.sh float_1010_floor tier3_cari4d_floor 0 3 5 7 8
set -euo pipefail
cd "$(dirname "$0")/../.."
TAG=${1:?用法：t3_float_pack.sh <标签> <版本目录> <集>...}; SUB=${2:?用法：t3_float_pack.sh <标签> <版本目录> <集>...}; shift 2
[ $# -ge 1 ] || { echo "t3_float_pack: 要至少一集" >&2; exit 2; }
[[ $TAG =~ ^[A-Za-z0-9_-]+$ ]] || { echo "t3_float_pack: 标签非法：$TAG" >&2; exit 2; }
[[ $SUB =~ ^tier3_cari4d[a-z0-9_]*$ ]] || { echo "t3_float_pack: 版本目录非法：$SUB" >&2; exit 2; }
for ep in "$@"; do [[ $ep =~ ^[0-9]+$ ]] || { echo "t3_float_pack: 集号非法：$ep" >&2; exit 2; }; done
SUP=${V2D_T3_SUPPORT:-}
[[ $SUP =~ ^(delivery)?$ ]] || { echo "t3_float_pack: V2D_T3_SUPPORT 只许空或 delivery：$SUP" >&2; exit 2; }
R='~/v2d/maniphys_lab_v2d_challenge'; IN="runs/tier3/incoming/$TAG/ok"
ssh -o BatchMode=yes mf4090 "cd $R && [ ! -e $IN ] && [ ! -e $IN.partial ] && mkdir -p $IN.partial" \
  || { echo "t3_float_pack: mf4090 上 $IN 已存在，或建不了目录" >&2; exit 1; }
if ! ssh -o BatchMode=yes ml40 "V2D_T3_SUBDIR=$SUB V2D_T3_SUPPORT=$SUP bash -s -- $*" < stages/recovery/tier3_pack_remote.sh | ssh -o BatchMode=yes mf4090 "cd $R && tar -xf - -C $IN.partial"; then
  ssh -o BatchMode=yes mf4090 "cd $R && rm -rf $IN.partial"
  echo "t3_float_pack: 转发失败，已删 mf4090 上的 $IN.partial" >&2; exit 1
fi
ssh -o BatchMode=yes mf4090 "cd $R && mv $IN.partial $IN && echo \"== \$(hostname) \$(date '+%F %T %Z') 来历 \$(cat .v2d-source-commit)；同组成员那边的目录 runs/$SUB → $IN\" \
  && echo \"parquet \$(ls $IN/data/chunk-000 | wc -l) 个，元数据 \$(ls $IN/meta | wc -l) 个；部署版本 \$(grep -oE '^[0-9a-f]{40}' $IN/DEPLOYED)\" \
  && echo '## parquet、修改时间（UTC）、sha256 前 12 位' && (cd $IN/data/chunk-000 && for f in *.parquet; do echo \"\$f \$(date -u -r \$f '+%F_%T') \$(sha256sum \$f | cut -c1-12)\"; done)"
if [ "$SUP" = delivery ]; then
  ssh -o BatchMode=yes mf4090 "cd $R && echo \"## 支撑面 \$(ls $IN/support_surface | wc -l) 个：文件名、sha256 前 12 位\" && cd $IN/support_surface && for f in *; do echo \"support_surface/\$f \$(sha256sum \$f | cut -c1-12)\"; done"
fi
echo T3_FLOAT_PACK_DONE
