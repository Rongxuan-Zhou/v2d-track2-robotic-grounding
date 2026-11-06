#!/usr/bin/env bash
# 把 mf4090 上 Tier 3 某一版的档目录、转换结果与重定向结果原样转到 ml40 的仓库副本（Mac 上、仓库根执行，Mac 只发命令、比清单）。
# 计划 agent-doc/plan-d-t3hm-train-2026-10-12.md 第 3 节第 8 步、第 4 节“ml40 的数据转发”与第 8 节（转发的补充，第 3 版：经 HF 私有仓库中转）。
# 转三个目录：runs/tier3/root_<名>、runs/motion_src/tier3_<名>、runs/motion/tier3_<名>
#   （parquet 里的 raw_motion_file 指向 motion_src；root 留作对照，三个一起转）。
#   1. 来源三个目录都在；目标三个目录与暂存目录 runs/.relay_tier3_<名>.partial 都不存在，才建暂存目录
#   2. 经 HF 私有仓库中转（用户 10/13：经 Mac 的管道太慢，“需要传数据还是得HF”）：mf4090 上 runs/tools/hf_transfer.py push 三个目录到
#      <仓库>:<标签>/，要 HF_PUSH_OK <文件数>；ml40 上 runs/tools/hf_transfer.py pull 到暂存目录，要 HF_PULL_OK <文件数>。两边都经
#      tools/hf_remote.sh 由 stdin 注入凭据，两次输出里 _load-env 打的“→ 主机:路径”要与 SH:SR、DH:DR 相同。失败就删暂存目录；
#      仓库里的副本留着：同一个标签可以重跑（push 见到与清单一致的副本就不重传，传了一半就接着传）
#   3. 两端各列一份 sha256 清单（相对仓库根的路径，按字节序排序），Mac 上比对：文件数等于 <文件数>、逐行相同；两端都只有普通文件与目录。
#      不一致就删暂存目录、停
#   4. 在暂存目录里查：data.parquet、v2d_meta.json、reconstructed_stage 的 <序列>_support.usda 各 <集数> 个，没有 <序列>_g1_support.usda；
#      v2d_meta.json 的档位都是 3、集号各不相同、sequence_id 与所在目录同名；每份 reconstructed_stage 支撑面与 root 的
#      support_surface 里同名文件的 sha256 相同。不对就删暂存目录、停
#   5. 都过才挪到位：挪之前再查一次目标都不存在、暂存目录与三个目标的上级目录在同一个文件系统（mv 只是改名）；挪完在目标位置再列一次
#      sha256 清单，要与来源的逐行相同（防止挪进了别人刚建的同名目录里）
# 用法：stages/recovery/tier3_relay.sh <名> <集数> <文件数> <标签>
#   例：caffeinate -i scripts/dev/record t3hm/relay_ml40_hf stages/recovery/tier3_relay.sh hm 29 523 t3hm_1013a
#   V2D_RELAY_REPO（默认 Rongxuan-Zhou/v2d-track2-transfer，私有）。V2D_RELAY_SRC_HOST、V2D_RELAY_SRC_ROOT、V2D_RELAY_DST_HOST、V2D_RELAY_DST_ROOT、
#   V2D_RELAY_HF_REMOTE 只给测试用（默认 mf4090 与 ml40 的仓库副本，同 .env 与 .env.ml40；tools/hf_remote.sh）。
# 最后一行 RELAY_OK，或 RELAY_FAILED <哪一步>。第 5 步中途失败时不删任何东西（见计划第 8 节的手动收尾）。测试：stages/recovery/test_tier3_relay.py。
set -uo pipefail
NAME=${1:?用法：tier3_relay.sh <名> <集数> <文件数> <标签>}; N=${2:?}; NF=${3:?}; TAG=${4:?}
[[ $NAME =~ ^[a-z0-9]+$ ]] || { echo "tier3_relay: 名只许小写字母与数字：$NAME" >&2; exit 2; }
[[ $N =~ ^[1-9][0-9]*$ && $NF =~ ^[1-9][0-9]*$ ]] || { echo "tier3_relay: 集数与文件数必须是正整数：$N $NF" >&2; exit 2; }
[[ $TAG =~ ^[A-Za-z0-9_-]+$ ]] || { echo "tier3_relay: 标签只许字母、数字、下划线、连字符：$TAG" >&2; exit 2; }
REPO=${V2D_RELAY_REPO:-Rongxuan-Zhou/v2d-track2-transfer}; HFR=${V2D_RELAY_HF_REMOTE:-tools/hf_remote.sh}
SH=${V2D_RELAY_SRC_HOST:-mf4090}; SR=${V2D_RELAY_SRC_ROOT:-/home/rxzhou/v2d/maniphys_lab_v2d_challenge}
DH=${V2D_RELAY_DST_HOST:-ml40}; DR=${V2D_RELAY_DST_ROOT:-/lab_data/rxzhou/v2d/maniphys_lab_v2d_challenge}
D1=runs/tier3/root_$NAME; D2=runs/motion_src/tier3_$NAME; D3=runs/motion/tier3_$NAME; P=runs/.relay_tier3_$NAME.partial
ST=$D3/whole_body/reconstructed_stage
rs() { ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 "$1" "$2"; }
ts() { date '+%F %T %Z'; }
fail() { echo "=== $(ts) RELAY_FAILED $1"; exit 1; }
clean() { rs "$DH" "cd $DR && rm -rf $P" && echo "=== $(ts) 已删 ${DH} 上的暂存目录 $P"; }
LIST="find $D1 $D2 $D3 -type f -print0 | LC_ALL=C sort -z | xargs -0r sha256sum"
ODD="find $D1 $D2 $D3 ! -type f ! -type d | wc -l | tr -d ' '"

echo "=== $(ts) ${SH}:${SR} → ${DH}:${DR}：$D1 $D2 ${D3}（${N} 集、${NF} 个文件）"
rs "$SH" "cd $SR && [ -d $D1 ] && [ -d $D2 ] && [ -d $D3 ] && echo 来历 \$(cat .v2d-source-commit 2>/dev/null) && du -sm $D1 $D2 $D3" || fail "1（来源缺目录）"
rs "$DH" "cd $DR && [ ! -e $D1 ] && [ ! -e $D2 ] && [ ! -e $D3 ] && [ ! -e $P ] && mkdir -p $P && echo 来历 \$(cat .v2d-source-commit 2>/dev/null) && df -Pm . | tail -1" \
  || fail "1（目标已有这些目录或暂存目录，或建不了暂存目录）"

t0=$(date +%s)
out="$(V2D_ENV= "$HFR" "python3 runs/tools/hf_transfer.py push --repo $REPO --tag $TAG --dirs $D1 $D2 $D3" 2>&1)"; rc=$?
printf '%s\n' "$out"
[ "$rc" = 0 ] && [ "$(tail -1 <<< "$out")" = "HF_PUSH_OK $NF" ] || { clean; fail "2（上传到 ${REPO}:${TAG}，退出码 ${rc}）"; }
grep -qxF "→ ${SH}:${SR}" <<< "$out" || { clean; fail "2（上传连的不是 ${SH}:${SR}）"; }
echo "=== $(ts) 上传用时 $(( $(date +%s) - t0 )) 秒"; t0=$(date +%s)
out="$(V2D_ENV=ml40 "$HFR" "python3 runs/tools/hf_transfer.py pull --repo $REPO --tag $TAG --dest $P" 2>&1)"; rc=$?
printf '%s\n' "$out"
[ "$rc" = 0 ] && [ "$(tail -1 <<< "$out")" = "HF_PULL_OK $NF" ] || { clean; fail "2（从 ${REPO}:${TAG} 下载，退出码 ${rc}）"; }
grep -qxF "→ ${DH}:${DR}  (V2D_ENV=ml40)" <<< "$out" || { clean; fail "2（下载连的不是 ${DH}:${DR}）"; }
echo "=== $(ts) 下载用时 $(( $(date +%s) - t0 )) 秒"

ms="$(rs "$SH" "cd $SR && $LIST")" || { clean; fail "3（来源清单）"; }
md="$(rs "$DH" "cd $DR/$P && $LIST")" || { clean; fail "3（目标清单）"; }
os="$(rs "$SH" "cd $SR && $ODD")" || { clean; fail "3（来源的非普通文件数）"; }
od="$(rs "$DH" "cd $DR/$P && $ODD")" || { clean; fail "3（目标的非普通文件数）"; }
ns="$(grep -c . <<< "$ms")"; nd="$(grep -c . <<< "$md")"
echo "=== $(ts) sha256 清单：来源 $ns 个文件（清单摘要 $(shasum -a 256 <<< "$ms" | cut -c1-16)），目标 $nd 个（$(shasum -a 256 <<< "$md" | cut -c1-16)）；非普通文件：来源 ${os}，目标 ${od}"
[ "$ns" = "$NF" ] && [ "$ms" = "$md" ] && [ "$os" = 0 ] && [ "$od" = 0 ] || { diff <(printf '%s\n' "$ms") <(printf '%s\n' "$md") | head -20; clean; fail "3（两端清单不一致或文件数不是 ${NF}）"; }

sup="$(python3 -c 'import sys
h = {}
for line in sys.stdin:
    d, p = line.rstrip("\n").split("  ", 1)
    h[p] = d
st, rt = sys.argv[1] + "/", sys.argv[2] + "/"
pairs = [(p, rt + p[len(st):]) for p in h if p.startswith(st)]
bad = [p for p, q in pairs if h.get(q) != h[p]]
print(len(pairs), len(bad), " ".join(bad[:3]))' "$ST" "$D1/support_surface" <<< "$md")"
read -r npair nbad badl <<< "$sup"
echo "=== $(ts) reconstructed_stage 与 root 的 support_surface 同名比对：${npair:-?} 份，不同 ${nbad:-?} 份 ${badl:-}"
[ "${npair:-}" = "$N" ] && [ "${nbad:-}" = 0 ] || { clean; fail "4（支撑面与 root 的不一致或份数不对）"; }

cnt="$(rs "$DH" "cd $DR/$P && echo \$(find $D3/whole_body/soma -name data.parquet | wc -l) \$(find $D2 -name v2d_meta.json | wc -l) \$(find $ST -name '*_support.usda' ! -name '*_g1_support.usda' | wc -l) \$(find $ST -name '*_g1_support.usda' | wc -l) && python3 -c 'import glob, json, os, sys
ms = [(p, json.load(open(p))) for p in sorted(glob.glob(sys.argv[1] + \"/*/v2d_meta.json\"))]
bad = [p for p, m in ms if m.get(\"sequence_id\") != os.path.basename(os.path.dirname(p))]
print(\",\".join(map(str, sorted({m[\"tier\"] for _, m in ms}))), len({m[\"episode\"] for _, m in ms}), len(bad), \",\".join(str(m[\"episode\"]) for _, m in sorted(ms, key=lambda x: x[1][\"episode\"])))' $D2")" \
  || { clean; fail "4（数不了）"; }
read -r np nm nu ng tiers neps nbadseq eps <<< "$(tr '\n' ' ' <<< "$cnt")"
echo "=== $(ts) 暂存目录里：parquet ${np}、v2d_meta.json ${nm}、支撑面 ${nu}（_g1_ ${ng}）；档位 ${tiers}；集号 ${neps} 个（${eps}）；sequence_id 与目录不符 ${nbadseq}"
[ "$np" = "$N" ] && [ "$nm" = "$N" ] && [ "$nu" = "$N" ] && [ "$ng" = 0 ] && [ "$tiers" = 3 ] && [ "$neps" = "$N" ] && [ "$nbadseq" = 0 ] \
  || { clean; fail "4（数目、档位、集号或 sequence_id 不对）"; }

rs "$DH" "cd $DR && mkdir -p runs/tier3 runs/motion_src runs/motion && [ ! -e $D1 ] && [ ! -e $D2 ] && [ ! -e $D3 ] \
  && [ \"\$(python3 -c 'import os, sys; print(len({os.stat(p).st_dev for p in sys.argv[1:]}))' $P runs/tier3 runs/motion_src runs/motion)\" = 1 ] \
  && mv $P/$D1 $D1 && mv $P/$D2 $D2 && mv $P/$D3 $D3 && find $P -depth -type d -empty -delete && [ ! -e $P ]" \
  || fail "5（挪到位：目标在挪之前出现、不在同一个文件系统，或挪的途中出错；没删任何东西）"
mf="$(rs "$DH" "cd $DR && $LIST")" || fail "5（挪完后的清单）"
[ "$mf" = "$ms" ] || { diff <(printf '%s\n' "$ms") <(printf '%s\n' "$mf") | head -20; fail "5（挪完后的清单与来源不一致）"; }
rs "$DH" "cd $DR && du -sm $D1 $D2 $D3"
echo "=== $(ts) 挪到位后清单与来源逐行相同（$(grep -c . <<< "$mf") 个文件）"
echo "=== $(ts) RELAY_OK"
