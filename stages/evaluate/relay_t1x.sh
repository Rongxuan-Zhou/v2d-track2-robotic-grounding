#!/usr/bin/env bash
# Tier 1 补候选：把一段变体 A 导出 ok 的新候选集中到 ml40，并在 ml40 上拼出这一次逐条选的输入目录
# （Mac 上、仓库根执行；Mac 只转发，不计算）。计划 agent-doc/plan-t1-r3x-2026-10-08.md 第 3 节第 5 步；
# 10/12 起按段参数化（计划 agent-doc/plan-next-training-2026-10-12.md 第 2 版第 0 步第 4 条）。
#   1. 段的训练机（t1y* 是 ml40，其余是 mf4090）：导出日志 runs/logs/cand_t1_<段>_<机器>.log 的最后一行是 EXPORT_CHAIN_DONE，
#      结果表 runs/exports/cand_t1_<段>_<机器>.tsv 在；stage_t1_candidates.sh 把状态 ok 的候选（1 世界导出的 json、parquet 与 4096 世界评测）
#      抄到暂存目录的 cand/，结果表抄到 tables/
#   2. 训练机是 mf4090 时：暂存目录是 mf4090 的 runs/select_t1_<段>_stage，经 Mac 管道 tar 到 ml40 的 runs/select_t1_<段>/；
#      训练机是 ml40 时：直接抄到 ml40 的 runs/select_t1_<段>/。目标目录已存在就拒绝
#   3. ml40：基础目录（上一次逐条选用的那份）的 cand/* 硬链接进 runs/select_t1_<段>/cand/，tables/*.tsv 抄进 tables/；
#      新旧候选重名就停；打印候选数、结果表数与这一段的新候选（run 与导出状态）
# 用法（经 record 落盘）：scripts/dev/record ops/relay_t1x_<段> stages/evaluate/relay_t1x.sh [<段> <基础目录>]
#   不给参数等于 r3x 那一次：段 r3x、基础目录 runs/select_t1（10/8）。10/12 起的 t1xa：段 t1xa、基础目录 runs/select_t1_r3x（10/9）。
# 中途失败后重跑：先删暂存目录与 ml40 的 runs/select_t1_<段>（只放抄来的副本与硬链接，删了不影响原件）。
set -euo pipefail
cd "$(dirname "$0")/../.."
MF=/home/rxzhou/v2d/maniphys_lab_v2d_challenge; ML=/lab_data/rxzhou/v2d/maniphys_lab_v2d_challenge
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=20)
stop() { echo "relay_t1x: $*，停" >&2; exit 1; }
SEG=${1:-r3x}; BASE=${2:-runs/select_t1}
[[ $SEG =~ ^[a-z0-9_]+$ ]] || stop "段名非法：$SEG"
[[ $BASE =~ ^runs/[A-Za-z0-9_./-]+$ && $BASE != *..* ]] || stop "基础目录要在 runs/ 下：$BASE"
case "$SEG" in t1y*) HOST=ml40; HDIR=$ML ;; *) HOST=mf4090; HDIR=$MF ;; esac
L=cand_t1_${SEG}_$HOST; DST=runs/select_t1_$SEG; STAGE=runs/select_t1_${SEG}_stage
CLEAN="重跑前删 ${HOST} 的 ${STAGE}（若有）与 ml40 的 ${DST}"

"${SSH[@]}" "$HOST" "cd $HDIR && tail -1 runs/logs/$L.log | grep -q EXPORT_CHAIN_DONE && test -f runs/exports/$L.tsv" \
  || stop "${HOST} 的导出链 $L 还没结束或没有结果表"
"${SSH[@]}" ml40 "cd $ML && test ! -e $DST && test -d $BASE/cand && ls $BASE/tables/*.tsv > /dev/null" \
  || stop "ml40 上 $DST 已存在，或 $BASE 缺 cand/ 或 tables/*.tsv（上次没跑完的话：$CLEAN）"
if [ "$HOST" = mf4090 ]; then
  "${SSH[@]}" mf4090 "cd $MF && test ! -e $STAGE && stages/evaluate/stage_t1_candidates.sh $STAGE/cand runs/exports/$L.tsv" \
    || stop "mf4090 上抄候选失败或 $STAGE 已存在（$CLEAN）"
  "${SSH[@]}" mf4090 "cd $MF && tar -C $STAGE -cf - cand tables" \
    | "${SSH[@]}" ml40 "cd $ML && mkdir $DST && tar -C $DST -xf -" || stop "转发失败（$CLEAN）"
else
  "${SSH[@]}" ml40 "cd $ML && stages/evaluate/stage_t1_candidates.sh $DST/cand runs/exports/$L.tsv" || stop "ml40 上抄候选失败（$CLEAN）"
fi
"${SSH[@]}" ml40 "set -e; cd $ML
for d in $BASE/cand/*/; do n=\$(basename \$d); if [ -e $DST/cand/\$n ]; then echo 重名 \$n; exit 1; fi; done
cp -al $BASE/cand/. $DST/cand/
cp -n $BASE/tables/*.tsv $DST/tables/
echo 候选 \$(ls $DST/cand | wc -l) 个，结果表 \$(ls $DST/tables | wc -l) 张
echo 这一段的新候选（run、导出状态）：; cut -f3,5 $DST/tables/$L.tsv" || stop "ml40 上拼目录失败（$CLEAN）"
echo RELAY_DONE
