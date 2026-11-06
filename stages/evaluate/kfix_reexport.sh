#!/usr/bin/env bash
# k-fix 方案 4 第 6 步：用现有 checkpoint 对新参考评测与导出（mf4090 上、仓库根执行，放 tmux 经 scripts/dev/record 落盘）。
# 计划 agent-doc/plan-t3-kfix-2026-10-16.md “方案 4 的执行细则”第 6、7 步与“评测器版本与两道核对”。清单（默认 stages/evaluate/exports/kfix_t3_mf4090.tsv）每行：
#   run、步数、参考 parquet（容器内 /repo/... 路径；plain 行写 -）、后缀、模式。
#   plain：不换参考的普通路径（fc_export.sh 不设 V2D_EXPORT_PARQUET），评 4096 世界，用作同一评测器（fc_8c778f9）下的旧值；
#   idchk：这条 run 原来训练用的 parquet 经 eval_playback_override.py 的包装层评 4096 世界，与 plain 比，看包装层改不改结果；
#   kfix：先评 4096 世界（只写指标），再导出 1 个世界（带轨迹，给打包用）。
# 为什么要 plain：10/11 的旧评测挂的是 third_party 的 v0.4.0，现在走 fc_8c778f9（改了 11 个文件），旧值要用同一评测器重评才能比。
# 开跑之前（不占 GPU）：打印每行参考 parquet 的 sha256；对清单每一行把要跑的每一步都预演一遍（V2D_EXPORT_DRYRUN=1：4096 世界评测、kfix 行的
# 1 世界导出、评测器不一致时要补的 plain），有一步不过就停（最后一行 KFIX_REEXPORT_STOPPED preflight）；删掉上一次的结果表。
# plain 行失败就停（恒等核对缺它没法做）；idchk 行评测失败、或评完调 kfix_reexport_table.py --check 判包装层不恒等，就停
# （KFIX_REEXPORT_STOPPED idchk，后面不占 GPU）；评测器不一致、或没有 10/11 的旧评测可比（--check 报“缺”）时，之后每条 kfix 行先补一条它自己的
# plain（判据 5 的旧值都用同一评测器的），补的 plain 失败就跳过这一集。
# 每一步都经 stages/evaluate/fc_export.sh：它核 checkpoint、NaN 守卫、GPU 占用，换参考时还核评测 JSON 里记的 parquet；失败时删掉这一次的输出。
# kfix 行失败不停，记下退出码；最后跑 kfix_reexport_table.py 汇总并写结果表 runs/exports/kfix_t3_mf4090.tsv（判“换”、导出齐全的集，格式同
# export_chain.sh），汇总脚本退出码非 0 时 KFIX_REEXPORT_STOPPED table。正常结束的最后一行 KFIX_REEXPORT_DONE <失败步数>。
# 用法：V2D_FC_SRC=<8c778f9 副本> stages/evaluate/kfix_reexport.sh [清单]
set -uo pipefail
cd "$(dirname "$0")/../.."
L=${1:-stages/evaluate/exports/kfix_t3_mf4090.tsv}
T=runs/exports/kfix_t3_mf4090.tsv
[ -n "${V2D_FC_SRC:-}" ] || { echo "kfix_reexport: 要设 V2D_FC_SRC" >&2; exit 2; }
[ -f "$L" ] || { echo "kfix_reexport: 没有清单 ${L}" >&2; exit 2; }
echo "== $(hostname) $(date '+%F %T %Z') 来历 $(cat .v2d-source-commit 2>/dev/null)；清单 ${L}（md5 $(md5sum < "$L" | cut -c1-32)）"
ep6() { printf %06d "$((10#$(echo "$1" | sed -E 's/^t3-ep([0-9]+)-.*/\1/')))"; }
# fx <后缀> <世界数> <轨迹> <参考或 -> <run> <步数> [预演]：调 fc_export.sh 一次
fx() {
  local pqv=$4; [ "$pqv" = - ] && pqv=
  V2D_EXPORT_PARQUET=$pqv V2D_EXPORT_SUFFIX=$1 V2D_EXPORT_WORLDS=$2 V2D_EXPORT_TRAJ=$3 V2D_EXPORT_DRYRUN=${7:-0} \
    stages/evaluate/fc_export.sh "$5" "$6" < /dev/null
}
pre=0
while IFS=$'\t' read -r run step pq suf mode <&3; do
  [ "$run" = run ] && continue
  if [ "$pq" != - ]; then
    h=${pq#/repo/}; [ -f "$h" ] && echo "参考 ${pq}：sha256 $(sha256sum < "$h" | cut -c1-64)" || { echo "参考不在：${h}"; pre=1; }
  fi
  case "$mode" in
    plain) fx _plain 4096 0 - "$run" "$step" 1 > /dev/null || { echo "预演不过：$run plain"; pre=1; } ;;
    idchk) fx "$suf" 4096 0 "$pq" "$run" "$step" 1 > /dev/null || { echo "预演不过：$run idchk"; pre=1; } ;;
    kfix) fx "$suf" 4096 0 "$pq" "$run" "$step" 1 > /dev/null || { echo "预演不过：$run 4096 世界"; pre=1; }
          fx "$suf" 1 1 "$pq" "$run" "$step" 1 > /dev/null || { echo "预演不过：$run 1 世界导出"; pre=1; }
          fx _plain 4096 0 - "$run" "$step" 1 > /dev/null || { echo "预演不过：$run 补的 plain"; pre=1; } ;;
    *) echo "清单里有不认识的模式：${mode}（${run}）"; pre=1 ;;
  esac
done 3< "$L"
[ "$pre" = 0 ] || { echo "预演没过，什么也不跑"; echo "KFIX_REEXPORT_STOPPED preflight"; exit 1; }
echo "预演全过（每行的每一步）"
rm -f "$T"
fail=0; drift=0
while IFS=$'\t' read -r run step pq suf mode <&3; do
  [ "$run" = run ] && continue
  echo "=== $(date '+%F %T %Z') ${run} ${step} ${mode} ${suf}"
  if [ "$mode" = plain ]; then
    fx _plain 4096 0 - "$run" "$step"; rc=$?
    echo "plain 4096 世界评测 退出码 ${rc}"
    [ "$rc" = 0 ] || { echo "plain 失败，恒等核对没法做，停"; echo "KFIX_REEXPORT_STOPPED plain"; exit 1; }
    continue
  fi
  if [ "$mode" = kfix ] && [ "$drift" = 1 ] && [ ! -f "runs/flash_chord/$run/export_8c778f9/s${step}_w4096_plain/episode_$(ep6 "$run").json" ]; then
    fx _plain 4096 0 - "$run" "$step"; rc=$?
    echo "补的 plain 4096 世界评测 退出码 ${rc}"
    [ "$rc" = 0 ] || { fail=$((fail + 1)); echo "补的 plain 失败，这一集没有同一评测器的旧值，跳过"; continue; }
  fi
  fx "$suf" 4096 0 "$pq" "$run" "$step"; rc=$?
  echo "4096 世界评测 退出码 ${rc}"; [ "$rc" = 0 ] || fail=$((fail + 1))
  if [ "$mode" = idchk ]; then
    [ "$rc" = 0 ] || { echo "idchk 评测失败，停（后面各行不跑）"; echo "KFIX_REEXPORT_STOPPED idchk"; exit 1; }
    chk=$(python3 stages/evaluate/kfix_reexport_table.py "$L" --check 2>&1)
    printf '%s\n' "$chk" | grep -a -E "恒等核对|评测器核对|缺|旁证" || true
    last=$(printf '%s\n' "$chk" | tail -1)
    case "$last" in
      "KFIX_CHECK 恒等 一致") ;;
      "KFIX_CHECK 恒等 不一致") drift=1; echo "评测器不一致：之后每条 kfix 行先补 plain" ;;
      "KFIX_CHECK 恒等 缺") drift=1; echo "没有 10/11 的旧评测可比：按评测器不一致处理，之后每条 kfix 行先补 plain" ;;
      *) echo "恒等核对没过（${last}），停（后面各行不跑）"; echo "KFIX_REEXPORT_STOPPED idchk"; exit 1 ;;
    esac
  fi
  if [ "$mode" = kfix ] && [ "$rc" = 0 ]; then
    fx "$suf" 1 1 "$pq" "$run" "$step"; rc=$?
    echo "1 世界导出 退出码 ${rc}"; [ "$rc" = 0 ] || fail=$((fail + 1))
  fi
done 3< "$L"
python3 stages/evaluate/kfix_reexport_table.py "$L" --write-table "$T"; trc=$?
[ "$trc" = 0 ] || { echo "汇总脚本退出码 ${trc}，结果表不可信"; rm -f "$T"; echo "KFIX_REEXPORT_STOPPED table"; exit 1; }
echo "KFIX_REEXPORT_DONE ${fail}"
