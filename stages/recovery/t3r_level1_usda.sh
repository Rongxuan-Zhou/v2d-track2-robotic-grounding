#!/usr/bin/env bash
# t3takeover 第 1 级 B 部分：在我们的 gate 仓库里照她 10/20 gb5dyn7 那一轮重出三集的支撑面 usda，再与参照比（训练机 ml40 上执行，放 tmux；只用 CPU）。
# 计划 agent-doc/plan-t3takeover-v3-2026-10-21.md 第 1 节第 3 条（usda 步）、复核第 1 条、判据 K1（usda 逐字节，support_usda.json 按集比、去掉路径字段）。
# 照她的 gate/chain_check.sh 13–20 行与 run_parallel.sh 23 行：
#   mkroot（T3_ROOT=我们的根；地面用 Tier 3 输入的 plane，mkroot 的默认）→ eps.sh 3（stages/recovery/t3r_gate_eps.sh：OMP/MKL 3、PATH 先她的 gate venv、
#   V2D_T3_SET=gb5dyn7；one.sh 自己给重定向 4 线程、对齐检查 2 线程）→ t3_support_usda.py（OMP 4；SUP 是 runs/support/tier3_cari4d_gb5_best，参照的 support_usda.json 记的就是它）。
# gate 脚本用我们部署的 42f01f0 副本（她跑时用 $T/code 下的同名脚本，d7aae27 之后没改；t3_inputs.py、t3_support.py 10/17 之后没改）。
# t3_support_usda.py 从 <根>/code/stages/recovery/tier3 导入 t3_inputs、t3_support，所以在根里放相对链接 code → ../code（链到我们自己的副本，不出 t3recon）。
# 每步经 stages/recovery/t3r_run.sh（她的目录只读、能力全丢、docker socket 盖住）。前提：stages/recovery/t3r_gate_setup.sh 搭好的 gate；根里有 A 部分的 dyn7 与 support。
# 重定向的结果行（帧数、接触、IK 中位、间隙、支撑面、状态）另与她的 gate/logs/rec_gb5dyn7.txt 里同集的行逐字段比（只读）。
# 用法：stages/recovery/t3r_level1_usda.sh <根名，t3 或 t3_refw> [<ep>...]   日志 runs/logs/t3r_level1_usda_<根名>.log，最后一行 T3R_USDA_DONE 或 T3R_USDA_FAILED <步>。
# 根名不是 t3 时集名是 gb5dyn7<后缀>（paths.py 只许小写字母与数字；同一个 gate 仓库里另起 root，不碰 gb5dyn7），support_usda.json 的比较另把 set 字段算进忽略的键；不给 ep 就跑 14 20 27。
set -uo pipefail
cd "$(dirname "$0")/../.."
REPO="$(pwd)"
NAME=${1:?根名}
R=/lab_data/rxzhou/v2d/t3recon; T3="$R/$NAME"; C="$R/code/stages/recovery/tier3/gate"; SET=gb5dyn7
IGN=(); [ "$NAME" = t3 ] || { SET="gb5dyn7${NAME#t3_}"; IGN=(--ignore-key "(path|dir|file|root|time|date|stamp|argv|host|cwd|user|from|created|deployed|code_files|^set$)"); }
shift; EPS=("$@"); [ ${#EPS[@]} -gt 0 ] || EPS=(14 20 27)
LOG="$REPO/runs/logs/t3r_level1_usda_${NAME}.log"
mkdir -p "$REPO/runs/logs"
exec >> "$LOG" 2>&1
say() { echo "=== $(date '+%F %T %Z') $*"; }
fail() { say "T3R_USDA_FAILED $*"; exit 1; }
. "$REPO/stages/recovery/t3r_env.sh"   # T3R_ENV=hers：gate 目录 t3recon/gate；ours：t3recon/gate_e、我们的环境、她的解释器被盖住（1b）
t3r_env "$REPO" || fail "另一位成员的目录、解释器或我们的环境没过核对（T3R_ENV=${T3R_ENV:-ours}）"
G="$GDIR"; GR="$G/repo"
export T3R_ROOT="$R" T3R_ALLOW="$GPY" T3R_GATE_VENV_BIN="$GATE_VENV_BIN"
GUARD="$REPO/stages/recovery/t3r_run.sh"
TG="${T3R_TAG:+_$T3R_TAG}"
rec() { local n=$1; shift; "$REPO/scripts/dev/record" "t3recon/${NAME}_usda${TG}_$n" "$@" > /dev/null 2>&1 || fail "$n（results/t3recon/${NAME}_usda${TG}_$n.txt.failed）"; say "$n 过"; }

[ -d "$GR" ] || fail "没有 $GR（先跑 t3r_gate_setup.sh）"
[ -e "$GR/runs/tier3/root_$SET" ] && fail "$GR/runs/tier3/root_$SET 已存在"
DYN="$T3/runs/tier3_cari4d_gb5_best_dr2s_dyn7"; SUP="$T3/runs/support/tier3_cari4d_gb5_best"
for d in "$DYN" "$SUP"; do [ -d "$d" ] || fail "没有 $d"; done
[ -e "$T3/code" ] || ln -s ../code "$T3/code"
[ "$(realpath -e "$T3/code")" = "$R/code" ] || fail "$T3/code 没有指向 $R/code"
say "开始，来历 $(cat .v2d-source-commit 2>/dev/null || echo unknown)，gate $(cat "$GR/.v2d-source-commit")，根 $T3，集 $SET"

rec mkroot bash "$GUARD" env "T3_ROOT=$T3" "$GPY" "$C/mkroot.py" "$GR" "$SET" - "$DYN" "${EPS[@]}"
rec eps bash "$GUARD" /bin/bash "$REPO/stages/recovery/t3r_gate_eps.sh" "$GR" "$SET" "${EPS[@]}"
rec support_usda bash "$GUARD" env "PYTHONPATH=$GR/third_party/video_to_data/robotic_grounding/source/robotic_grounding" OMP_NUM_THREADS=4 \
  "$GPY" "$C/t3_support_usda.py" "$GR" "$SET" "$SUP" "$T3" "$G/support_usda_$SET"

# 比较：usda 逐字节、support_usda.json 按集比（t3r_compare.py --subset），比较器用她的宿主机解释器（只读）
REF="$R/ref/gate/support_usda_gb5dyn7"
p=(); for f in "$G/support_usda_$SET"/*.usda; do p+=("$f" "$REF/$(basename "$f")"); done
"$REPO/scripts/dev/record" "t3recon/${NAME}_vs_ref_cmp${TG}_usda" env T3R_ALLOW="$HPY" bash "$GUARD" "$HPY" "$REPO/stages/recovery/t3r_compare.py" "${p[@]}" > /dev/null 2>&1
say "usda 比较退出码 $?"
"$REPO/scripts/dev/record" "t3recon/${NAME}_vs_ref_cmp${TG}_usda_json" env T3R_ALLOW="$HPY" bash "$GUARD" "$HPY" "$REPO/stages/recovery/t3r_compare.py" --subset "${IGN[@]}" \
  "$G/support_usda_$SET/support_usda.json" "$REF/support_usda.json" > /dev/null 2>&1
say "support_usda.json 比较退出码 $?"
# 重定向结果行逐字段比：参照是第 1 级在 t3_refw 上重出的 t3_gb5dyn7refw_ep*.tsv（已与她 rec_gb5dyn7.txt 的同集行逐字段相同；1b 复核改动第 7 条，不再读她仍在用的文件）
for e in "${EPS[@]}"; do
  ours="$(cat "$GR/runs/logs/batch/t3_${SET}_ep$(printf %02d "$e").tsv" 2>/dev/null)"
  want="$(cat "$R/gate/repo/runs/logs/batch/t3_gb5dyn7refw_ep$(printf %02d "$e").tsv" 2>/dev/null)"
  [ -n "$want" ] && [ "$ours" = "$want" ] && say "ep$e 重定向结果行相同：$ours" || say "ep$e 重定向结果行不同：本次 [$ours] 参照 [$want]"
done
say "T3R_USDA_DONE"
