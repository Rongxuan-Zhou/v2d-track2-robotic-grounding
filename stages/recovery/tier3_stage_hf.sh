#!/usr/bin/env bash
# 把 ml40 上一个 Tier 3 版本目录里的几集经私有 HF 中转仓转到 mf4090，拼进 runs/tier3/root_<名>（Mac 上、仓库根执行；数据不经 Mac，Mac 只发命令）。
# 计划 agent-doc/plan-t3-kfix-2026-10-16.md “方案 4 的执行细则”第 4 步（10/16 定：新参考走 HF，不走 Mac 管道）。与 stages/recovery/tier3_stage.sh
# 只差第 2 步（那边是 ml40 打包 → Mac 管道 → mf4090 解包），其余各步相同：
#   0. 哈希表 stages/recovery/expect/<标签>.tsv 要已提交并在 mf4090 上（tier3_expect.py 生成）
#   1. mf4090 上 runs/tier3/incoming/<标签>、<标签>.partial、<标签>.pull 都不能已存在；ml40 上 runs/hf_src/<标签> 不能已存在
#   2a. ml40：tier3_pack_remote.sh 经 stdin 执行，tar 写到 ml40 本机 runs/hf_src/<标签>/pack.tar，解到 runs/hf_src/<标签>/pack/，删掉 tar
#   2b. ml40：v2d_cari4d 容器里跑 tools/hf_transfer.py push（--dirs pack，标签就是 <标签>）。凭据经 tools/hf_remote.sh 的 stdin 进 ml40 上的环境变量，
#       docker 只按变量名传（-e HF_TOKEN 不带值），命令行与日志里没有凭据；整个仓库副本挂成 /repo，pack/ 与暂存的硬链接在同一个挂载里
#   2c. mf4090：hf_transfer.py pull 到空目录 runs/tier3/incoming/<标签>.pull（逐个文件按清单的 sha256 核），把其中的 pack/ 搬成 <标签>.partial，删 .pull。
#       mf4090 用 .env（不设 V2D_ENV；没有 .env.mf4090，同 pull_submission.sh 的 host_env），ml40 用 .env.ml40
#   3. 同 tier3_stage.sh：改名为 <标签>/、列出 sha256 与部署版本、V2D_T3_SET=<名> tier3_root.py <incoming> --expect <哈希表>（加 --support-dir 时
#      支撑面从 mf4090 上已拼好的档目录复制，按哈希表的 sha256 核；加 --support-from-incoming 时支撑面从转来的目录复制，同样按哈希表核）
# 环境变量：V2D_T3_SUBDIR（必填，白名单在 tier3_pack_remote.sh）、V2D_T3_OWN（本用户目录模式，见 tier3_pack_remote.sh）、V2D_T3_SET（必填）、
#   V2D_T3_SUPPORT_DIR（runs/tier3/root_<名>，可空）、V2D_HF_XFER_REPO（默认 Rongxuan-Zhou/v2d-track2-transfer）。
# V2D_T3_SUPPORT=delivery（只许空或 delivery；计划 agent-doc/plan-t3-gb3-switch-2026-10-17.md 做法 1b，用法同 tier3_stage.sh）：打包带上交付目录里
#   每集元数据 support_surface 指向的文件，tier3_root.py 加 --support-from-incoming；哈希表要用 tier3_expect.py --support-from-remote 生成。
#   与 V2D_T3_SUPPORT_DIR 只能设一个。
# 任何一步失败就停，已经建的 .partial / .pull 留着给人看（不自动删 HF 上的文件）。
# V2D_STAGEHF_R、V2D_STAGEHF_R40、V2D_STAGEHF_HF_REMOTE、V2D_STAGEHF_PACK 只给测试用：两台的仓库副本路径、hf_remote 与打包脚本。
# 用法：scripts/dev/record t3kfix/stage_hf_mf4090 stages/recovery/tier3_stage_hf.sh <标签> <集>...
set -euo pipefail
cd "$(dirname "$0")/../.."
TAG=${1:?用法：tier3_stage_hf.sh <标签> <集>...}; shift
[ $# -ge 1 ] || { echo "tier3_stage_hf: 要至少一集" >&2; exit 2; }
[[ $TAG =~ ^[A-Za-z0-9_-]+$ ]] || { echo "tier3_stage_hf: 标签非法：${TAG}" >&2; exit 2; }
for ep in "$@"; do [[ $ep =~ ^[0-9]+$ ]] || { echo "tier3_stage_hf: 集号非法：${ep}" >&2; exit 2; }; done
SUB=${V2D_T3_SUBDIR:?要设 V2D_T3_SUBDIR}; SET=${V2D_T3_SET:?要设 V2D_T3_SET}; OWN=${V2D_T3_OWN:-}
[[ $SUB =~ ^tier3_[a-zA-Z0-9_]+$ ]] || { echo "tier3_stage_hf: V2D_T3_SUBDIR 非法：${SUB}" >&2; exit 2; }
[[ $SET =~ ^[a-z0-9]+$ ]] || { echo "tier3_stage_hf: V2D_T3_SET 只许小写字母与数字：${SET}" >&2; exit 2; }
[[ -z $OWN || $OWN =~ ^/lab_data/[a-z0-9]+/v2d/[A-Za-z0-9_/-]+$ ]] || { echo "tier3_stage_hf: V2D_T3_OWN 非法：${OWN}" >&2; exit 2; }
SDIR=${V2D_T3_SUPPORT_DIR:-}
[[ $SDIR =~ ^(runs/tier3/root_[a-z0-9]+)?$ ]] || { echo "tier3_stage_hf: V2D_T3_SUPPORT_DIR 只许空或 runs/tier3/root_<名>：${SDIR}" >&2; exit 2; }
SUP=${V2D_T3_SUPPORT:-}
[[ $SUP =~ ^(delivery)?$ ]] || { echo "tier3_stage_hf: V2D_T3_SUPPORT 只许空或 delivery：${SUP}" >&2; exit 2; }
[ -z "$SDIR" ] || [ "$SUP" != delivery ] || { echo "tier3_stage_hf: V2D_T3_SUPPORT=delivery 与 V2D_T3_SUPPORT_DIR 只能设一个" >&2; exit 2; }
REPO=${V2D_HF_XFER_REPO:-Rongxuan-Zhou/v2d-track2-transfer}
[[ $REPO =~ ^[A-Za-z0-9_-]+/[A-Za-z0-9._-]+$ ]] || { echo "tier3_stage_hf: V2D_HF_XFER_REPO 非法：${REPO}" >&2; exit 2; }
SFLAG=""; SNOTE=""
if [ -n "$SDIR" ]; then SFLAG=" --support-dir $SDIR"; SNOTE="；V2D_T3_SUPPORT_DIR=${SDIR}"; fi
if [ "$SUP" = delivery ]; then SFLAG=" --support-from-incoming"; SNOTE="；V2D_T3_SUPPORT=delivery"; fi
R=${V2D_STAGEHF_R:-'~/v2d/maniphys_lab_v2d_challenge'}; IN="runs/tier3/incoming/$TAG"; EXP="stages/recovery/expect/$TAG.tsv"; HS="runs/hf_src/$TAG"
HFR=${V2D_STAGEHF_HF_REMOTE:-tools/hf_remote.sh}; PACK=${V2D_STAGEHF_PACK:-stages/recovery/tier3_pack_remote.sh}   # V2D_STAGEHF_* 只给测试用（stages/recovery/test_tier3_stage_hf.py）
[ -f "$EXP" ] || { echo "tier3_stage_hf: 缺哈希表 ${EXP}（先用 stages/recovery/tier3_expect.py 生成、提交并放到 mf4090）" >&2; exit 1; }

echo "== 1. 前置检查"
ssh -o BatchMode=yes mf4090 "cd $R && [ -f $EXP ] && [ ! -e $IN ] && [ ! -e $IN.partial ] && [ ! -e $IN.pull ] && cmp -s $EXP <(cat) && echo mf4090_OK" < "$EXP" \
  || { echo "tier3_stage_hf: mf4090 上缺 ${EXP} 或与本地不同，或 ${IN}（.partial、.pull）已存在" >&2; exit 1; }
R40=${V2D_STAGEHF_R40:-$(ROOT=$PWD V2D_ENV=ml40 bash -c '. "$ROOT/scripts/dev/_load-env" >/dev/null 2>&1; printf %s "$V2D_REMOTE"')}
[[ -n ${V2D_STAGEHF_R40:-} || $R40 =~ ^/lab_data/[a-z0-9]+/v2d/[A-Za-z0-9_/-]+$ ]] || { echo "tier3_stage_hf: ml40 的仓库副本路径不对：${R40}" >&2; exit 1; }
ssh -o BatchMode=yes ml40 "cd $R40 && [ ! -e $HS ] && mkdir -p $HS/pack && echo ml40_OK" || { echo "tier3_stage_hf: ml40 上 ${HS} 已存在或建不了" >&2; exit 1; }

echo "== 2a. ml40 本机打包并解开（数据不经 Mac）"
ssh -o BatchMode=yes ml40 "cd $R40 && V2D_T3_SUBDIR=$SUB V2D_T3_OWN=$OWN V2D_T3_SUPPORT=$SUP bash -s -- $* > $HS/pack.tar" < "$PACK" \
  || { echo "tier3_stage_hf: ml40 上打包失败（${HS}/pack.tar 留着）" >&2; exit 1; }
ssh -o BatchMode=yes ml40 "cd $R40 && tar -xf $HS/pack.tar -C $HS/pack && rm $HS/pack.tar && (cd $HS/pack && find . -type f | sort | xargs sha256sum)" \
  || { echo "tier3_stage_hf: ml40 上解包失败" >&2; exit 1; }

echo "== 2b. ml40：v2d_cari4d 容器里推到 ${REPO} 的 ${TAG}/"
out=$(V2D_ENV=ml40 "$HFR" "docker run --rm --user \"\$(id -u):\$(id -g)\" --entrypoint '' --network host -e HOME=/tmp -e HF_TOKEN -e https_proxy -e HTTPS_PROXY -e HF_ENDPOINT -e HF_HUB_DISABLE_PROGRESS_BARS -v \"\$PWD\":/repo -w /repo/$HS v2d_cari4d:latest python /repo/tools/hf_transfer.py push --repo $REPO --tag $TAG --dirs pack" 2>&1 || true)
printf '%s\n' "$out" | tail -n 15
printf '%s\n' "$out" | tail -n 1 | grep -q '^HF_PUSH_OK ' || { echo "tier3_stage_hf: 推送没有以 HF_PUSH_OK 结束" >&2; exit 1; }

echo "== 2c. mf4090：拉到 ${IN}.pull 再搬成 ${IN}.partial"
out=$(env -u V2D_ENV "$HFR" "mkdir -p runs/tier3/incoming && mkdir $IN.pull && python3 tools/hf_transfer.py pull --repo $REPO --tag $TAG --dest $IN.pull" 2>&1 || true)
printf '%s\n' "$out" | tail -n 5
printf '%s\n' "$out" | tail -n 1 | grep -q '^HF_PULL_OK ' || { echo "tier3_stage_hf: 拉取没有以 HF_PULL_OK 结束（${IN}.pull 留着）" >&2; exit 1; }
ssh -o BatchMode=yes mf4090 "cd $R && [ -d $IN.pull/pack ] && [ \"\$(ls -A $IN.pull)\" = pack ] && mv $IN.pull/pack $IN.partial && rmdir $IN.pull && echo MOVED" \
  || { echo "tier3_stage_hf: ${IN}.pull 里不只有 pack/，没动" >&2; exit 1; }

echo "== 3. 核对并拼档目录"
ssh -o BatchMode=yes mf4090 "cd $R && mv $IN.partial $IN && echo \"== \$(hostname) \$(date '+%F %T %Z') 来历 \$(cat .v2d-source-commit)\" && (cd $IN && find . -type f | sort | xargs sha256sum) && echo \"部署版本 \$(grep -oE '^[0-9a-f]{40}' $IN/DEPLOYED)；版本目录 runs/${SUB}；V2D_T3_SET=${SET}；经 HF ${REPO} ${TAG}/${SNOTE}\" && V2D_T3_SET=$SET python3 stages/recovery/tier3_root.py $IN --expect $EXP$SFLAG"
