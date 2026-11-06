#!/usr/bin/env bash
# Tier 1 逐条选的打分、选择与打包（ml40、仓库根执行）。计划 agent-doc/plan-t1-per-episode-select-2026-10-08.md 第 2 版。
# 这台机器上的仓库是 rsync 副本，pack_tier.sh 的守卫不在这里跑，所以这里自己核（审查第 1 条）：
#   .v2d-source-commit 是 --commit-url 那个 commit、没有 -dirty / -syncing；kit zip 的 sha256 是 0f2f9f4b…；CSV 已存在就拒绝。
# 用法：[V2D_SELECT_TAG=<标签>] [V2D_SELECT_DIR=<目录>] stages/evaluate/select_t1.sh <https://github.com/…/commit/<sha>> [--no-pack]
#   标签默认 1008、目录默认 runs/select_t1（10/8 那一次）；Tier 1 变体 A 补跑（计划 agent-doc/plan-t1-r3x-2026-10-08.md）用
#   V2D_SELECT_TAG=r3x V2D_SELECT_DIR=runs/select_t1_r3x。同一标签的分数文件或 CSV 已存在就拒绝。
#   10/12 起（计划 agent-doc/plan-next-training-2026-10-12.md 第 0 步第 4 条）：V2D_SELECT_CURRENT=<现在交的清单…>（空格分隔，都在
#   stages/evaluate/exports/ 下；默认 mf4090.tsv 与 ml40.tsv，即 10/3 那份）、V2D_SELECT_GATE=<AUC,SP,MP,MPPE>（打分门 B 对照的合计，
#   默认 10/3 的值）、V2D_SELECT_GATE_LABEL=<名>（只用于打印）。t1xa 那一次：V2D_SELECT_CURRENT=stages/evaluate/exports/t1_select_r3x.tsv
#   V2D_SELECT_GATE=0.459651,0.600000,0.500000,24.623127 V2D_SELECT_GATE_LABEL=10/9。标签不是 1008、r3x 时这两个变量必须给（代码审查 m1）。
# 输入：<目录>/cand/<run>/（stage_t1_candidates.sh 集中来的候选）、<目录>/tables/*.tsv（导出链结果表）、
#       现在交的 30 条（默认 stages/evaluate/exports/{mf4090,ml40}.tsv）、runs/kit/update_0f2f9f4b/（官方 kit）、runs/kitdeps/（kit 依赖）。
# 输出：results/select_t1/select_<标签>.txt（record）、results/select_t1/scores_<标签>.json、results/select_t1/t1_select_<标签>.tsv、
#       runs/submission/csv/track2_tier1_select_<标签>.csv（不进 git）。
set -uo pipefail
cd "$(dirname "$0")/../.."
die() { echo "select_t1: $*" >&2; exit "${2:-2}"; }
URL=${1:?commit URL}; NOPACK=${2:-}
[[ $URL =~ ^https://github\.com/Manifold-ai/maniphys_lab_v2d_challenge/commit/([0-9a-f]{40})$ ]] || die "commit URL 格式不对：$URL"
SHA=${BASH_REMATCH[1]}
SRC="$(cat .v2d-source-commit 2>/dev/null)"
[[ $SRC == "${SHA:0:12}" ]] || die "这里的来历是 '$SRC'，不是 ${SHA:0:12}（或带 -dirty / -syncing）：先提交、推送、再同步"
KIT=runs/kit/update_0f2f9f4b
[ "$(sha256sum $KIT/v2d_submission_kit.zip | cut -d' ' -f1)" = 0f2f9f4bb01082a48ab684750cbcbddd985c2de7031847928ae124e53fa605d7 ] || die "kit zip 不是 0f2f9f4b"
TAG=${V2D_SELECT_TAG:-1008}; DIR=${V2D_SELECT_DIR:-runs/select_t1}
[[ $TAG =~ ^[A-Za-z0-9_]+$ ]] || die "标签只能是字母、数字、下划线：$TAG"
[[ $DIR =~ ^runs/[A-Za-z0-9_./-]+$ && $DIR != *..* ]] || die "目录要在 runs/ 下：$DIR"
[ -d "$DIR/cand" ] && compgen -G "$DIR/tables/*.tsv" > /dev/null || die "$DIR 下缺 cand/ 或 tables/*.tsv"
[ ! -e "results/select_t1/scores_$TAG.json" ] || die "results/select_t1/scores_$TAG.json 已存在"
CSV=runs/submission/csv/track2_tier1_select_$TAG.csv
PACK=(--pack /repo/$CSV)
[ "$NOPACK" = --no-pack ] && PACK=()
[ -n "${PACK[*]:-}" ] && [ -e "$CSV" ] && die "$CSV 已存在"
case "$TAG" in
  1008|r3x) ;;   # 这两次的现在交的清单与打分门就是默认值
  *) [ -n "${V2D_SELECT_CURRENT:-}" ] && [ -n "${V2D_SELECT_GATE:-}" ] \
       || die "标签 $TAG 不是 1008、r3x：要显式给 V2D_SELECT_CURRENT 与 V2D_SELECT_GATE（不默认套用 10/3 的清单与打分门）" ;;
esac
CUR=${V2D_SELECT_CURRENT:-stages/evaluate/exports/mf4090.tsv stages/evaluate/exports/ml40.tsv}
for f in $CUR; do
  [[ $f =~ ^stages/evaluate/exports/[A-Za-z0-9_.-]+\.tsv$ ]] && [ -f "$f" ] || die "现在交的清单要是 stages/evaluate/exports/ 下存在的 .tsv：$f"
done
GATEARG=()
if [ -n "${V2D_SELECT_GATE:-}" ]; then
  [[ $V2D_SELECT_GATE =~ ^[0-9.]+,[0-9.]+,[0-9.]+,[0-9.]+$ ]] || die "V2D_SELECT_GATE 要四个数、逗号分隔：$V2D_SELECT_GATE"
  LABEL=${V2D_SELECT_GATE_LABEL:-custom}
  [[ $LABEL =~ ^[A-Za-z0-9][A-Za-z0-9_/.-]*$ ]] || die "V2D_SELECT_GATE_LABEL 只能是字母、数字与 _/.-，且以字母或数字开头：$LABEL"
  GATEARG=(--gate "$V2D_SELECT_GATE" --gate-label "$LABEL")
fi
mkdir -p results/select_t1 runs/submission/csv
scripts/dev/record "select_t1/select_$TAG" scripts/dev/in-flash-chord "cd /repo && PYTHONPATH=/repo/runs/kitdeps python stages/evaluate/select_t1.py \
  --cand $DIR/cand --kit $KIT/v2d_submission_kit --dataset-root data/challenge --commit-url $URL \
  --current $CUR --sha-tables $DIR/tables/*.tsv ${GATEARG[*]:-} \
  --out-json results/select_t1/scores_$TAG.json --out-tsv results/select_t1/t1_select_$TAG.tsv ${PACK[*]:-}"
