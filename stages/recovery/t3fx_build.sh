#!/usr/bin/env bash
# t3hand P1 的下游链：在我们自己的根 t3recon/t3fx 上，把 t3_held_fix.py 出的 tier3_cari4d_gb5_best_dr2s_hr 层接到训练用的参考
# （训练机 ml40 上执行，放 setsid；只用 CPU）。计划 agent-doc/plan-t3hand-2026-10-27.md 第 3 版 P1 第 6 条。
# 步骤照 stages/recovery/t3r_level1.sh（physics、dyn7）与 t3r_level1_usda.sh（mkroot、重定向 eps.sh、support_usda），版本名换成 _hr：
#   1. 物理检查 tier3_cari4d_gb5_best_dr2s_hr（宿主机提取、镜像里检查）；
#   2. dyn7：tier3_cari4d_gb5_best_dr2s_hr → tier3_cari4d_gb5_best_dr2s_hr_dyn7（只出给定的集；dyn7 读 _hr 的 depth_post.json 锚点，t3_held_fix 已照抄）；
#   3. 物理检查 tier3_cari4d_gb5_best_dr2s_hr_dyn7；
#   4. gate 仓库里 mkroot（集名 gb5hrdyn7）→ eps.sh 重定向 → t3_support_usda.py，只对给定的集。
# 每步经 stages/recovery/t3r_run.sh（她的目录只读、能力全丢、docker socket 盖住）；代码用部署在 t3recon/code 的副本（dyn7、物理检查、gate 脚本都在那里）。
# 用法：[T3FX_HR=<_hr 层> T3FX_SET=<集名>] [T3FX_SUP=<support 下的目录名>] [T3FX_DYN7_PHYS=runs/physics/<版本>] stages/recovery/t3fx_build.sh <ep>...   日志 runs/logs/t3fx_build.log，最后一行 T3FX_BUILD_DONE 或 T3FX_BUILD_FAILED <步>。
set -uo pipefail
cd "$(dirname "$0")/../.."
REPO="$(pwd)"
EPS=("$@"); [ ${#EPS[@]} -gt 0 ] || { echo "用法：t3fx_build.sh <ep>..." >&2; exit 2; }
R=/lab_data/rxzhou/v2d/t3recon; T3="$R/t3fx"; CODE="$R/code"; S="$CODE/stages/recovery/tier3"; C="$S/gate"
SOMA="$REPO/third_party/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/body_models/soma"
IMG=ca1774d94cd9
HR=${T3FX_HR:-tier3_cari4d_gb5_best_dr2s_hr}; DYN=${HR}_dyn7; SET=${T3FX_SET:-gb5hrdyn7}   # 10/28 plan-t3fix2：ep27 用 T3FX_HR=…_hr2 T3FX_SET=gb5hr2dyn7
[ "$([ -n "${T3FX_HR:-}" ] && echo 1)" = "$([ -n "${T3FX_SET:-}" ] && echo 1)" ] || { echo "T3FX_HR 与 T3FX_SET 要同时设或同时不设" >&2; exit 2; }
RP=""; [ "$SET" = gb5hrdyn7 ] || RP="${SET}_"   # 记录名前缀：默认集沿用旧名，其它集不覆盖旧记录
LOG="$REPO/runs/logs/t3fx_build.log"
mkdir -p "$REPO/runs/logs"
exec >> "$LOG" 2>&1
say() { echo "=== $(date '+%F %T %Z') $*"; }
fail() { say "T3FX_BUILD_FAILED $*"; exit 1; }
. "$REPO/stages/recovery/t3r_env.sh"
t3r_env "$REPO" 2>> "$LOG" || fail "环境核对不过（T3R_ENV=${T3R_ENV:-ours}）"
export T3R_ROOT="$T3" T3R_ALLOW="$GPY:$HPY" T3R_GATE_VENV_BIN="$GATE_VENV_BIN"
GUARD="$REPO/stages/recovery/t3r_run.sh"
G="$GDIR"; GR="$G/repo"
DK=(docker run --rm --user 1003:1005 --entrypoint "" --network none -e HOME=/tmp -w /workspace/v2d_cari4d/lib/cari4d)
rec() { local n=$1; shift; "$REPO/scripts/dev/record" "ml40/t3hand/build_$RP$n" "$@" > /dev/null 2>&1 || fail "$n（results/ml40/t3hand/build_$RP$n.txt.failed）"; say "$n 过"; }

[ -d "$T3/runs/$HR" ] || fail "没有 $T3/runs/$HR"
[ -f "$T3/runs/$HR.depth_post.json" ] || fail "没有 $T3/runs/$HR.depth_post.json"
[ -e "$T3/runs/$DYN" ] && fail "$T3/runs/$DYN 已存在"
[ -d "$GR" ] || fail "没有 $GR（先跑 t3r_gate_setup.sh）"
[ -e "$GR/runs/tier3/root_$SET" ] && fail "$GR/runs/tier3/root_$SET 已存在"
docker image inspect "$IMG" > /dev/null 2>&1 || fail "没有镜像 $IMG"
say "开始，来历 $(cat .v2d-source-commit 2>/dev/null || echo unknown)，根 $T3，代码 $(cat "$CODE/DEPLOYED" 2>/dev/null)，集 ${EPS[*]}"

physics() {   # physics <版本>：宿主机提取、镜像里检查
  local V=$1
  rec "physics_extract_$V" bash "$GUARD" "$HPY" "$S/t3_physics_check.py" extract "$T3" "$T3/runs/$V" "$T3/runs/physics/$V/npz"
  rec "physics_check_$V" "${DK[@]}" --cpus 8 -v "$S:/s:ro" -v "$T3:/t:ro" -v "$SOMA:/a:ro" -v "$T3/runs/physics/$V:/o" "$IMG" python /s/t3_physics_check.py check /t /o/npz /a /o/out
}
physics "$HR"
EPL="$(IFS=,; echo "${EPS[*]}")"
DPH=${T3FX_DYN7_PHYS:-runs/physics/$HR}   # 10/28 plan-t3b2n12 第 1.2 版：B2n 改了几何，"被拿着"在阈值边上会翻一帧；可指定 dyn7 沿用转动前那一层的物理检查
[ -d "$T3/$DPH/out" ] || fail "没有 $T3/$DPH/out"
rec dyn7 bash "$GUARD" "$GPY" "$S/t3_dynamics_post.py" "$T3" "$HR" "$DPH" "$DYN" "$EPL"   # 只出给定的集（_suspect 里的集没有物理检查，dyn7 全跑会找不到 npz）
physics "$DYN"
[ -e "$T3/code" ] || ln -s ../code "$T3/code"
[ "$(realpath -e "$T3/code")" = "$R/code" ] || fail "$T3/code 没有指向 $R/code"
SUP="$T3/runs/support/${T3FX_SUP:-tier3_cari4d_gb5_best}"   # 11/04 plan-t3gb6：gb6 的支撑面与 gb5 不同，T3FX_SUP=tier3_cari4d_gb6_best
[ -d "$SUP" ] || fail "没有 $SUP"
rec mkroot bash "$GUARD" env "T3_ROOT=$T3" "$GPY" "$C/mkroot.py" "$GR" "$SET" - "$T3/runs/$DYN" "${EPS[@]}"
rec eps bash "$GUARD" /bin/bash "$REPO/stages/recovery/t3r_gate_eps.sh" "$GR" "$SET" "${EPS[@]}"
rec support_usda bash "$GUARD" env "PYTHONPATH=$GR/third_party/video_to_data/robotic_grounding/source/robotic_grounding" OMP_NUM_THREADS=4 \
  "$GPY" "$C/t3_support_usda.py" "$GR" "$SET" "$SUP" "$T3" "$G/support_usda_$SET"
for e in "${EPS[@]}"; do say "ep$e 重定向结果行：$(cat "$GR/runs/logs/batch/t3_${SET}_ep$(printf %02d "$e").tsv" 2>/dev/null)"; done
say "T3FX_BUILD_DONE"
