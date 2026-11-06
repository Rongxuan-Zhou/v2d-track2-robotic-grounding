#!/usr/bin/env bash
# t3takeover 第 1 级 A 部分：在我们自己的根里按她的步骤重出 gb5 的 decode → focal → world → assemble → 物理检查 → support → dr2s → 物理检查 → dyn7 → 物理检查
# （训练机 ml40 上执行，放 tmux；只用 CPU）。计划 agent-doc/plan-t3takeover-v3-2026-10-21.md 第 1 节第 4 条与复核。支撑面 usda 是 B 部分，另跑。
# 宿主机的步骤一律经 stages/recovery/t3r_run.sh（她的目录只读、能力全丢）调她的两个解释器；docker 的步骤只挂我们的目录、--user 1003:1005、--network none、镜像按 ID。
# 线程照她当时的：decode 固定 8 线程（ep27 另给 --cpus 6、OMP 6）；focal、world 在 ep14、ep20 是 --cpus 4、OMP 4，ep27 是 --cpus 6、OMP 6；物理检查 --cpus 8；
# support 设 T3_FLOOR_FIT、T3_REGISTER 与 OMP 4；dr2s、dyn7 不设。ep14、ep20 的 world 用她当时的文件名（floor_fit_gb3.json、camera_register_gb3.json，内容是 gb5 那两份）。
# 用法：stages/recovery/t3r_level1.sh <根名，t3 或 t3_rep>   日志 runs/logs/t3r_level1_<根名>.log，最后一行 T3R_LEVEL1_DONE 或 T3R_LEVEL1_FAILED <步>。
# T3R_FROM=assemble：跳过 decode、focal、world，转换目录里要已有 world 的三个产物（stages/recovery/t3r_seed_root.sh world-from-ref 种的根；K2 登记第 3 条）。
set -uo pipefail
cd "$(dirname "$0")/../.."
REPO="$(pwd)"
NAME=${1:?根名}
FROM=${T3R_FROM:-decode}
R=/lab_data/rxzhou/v2d/t3recon
T3="$R/$NAME"; CODE="$R/code"; S="$CODE/stages/recovery/tier3"; W="$R/w"
SOMA="$REPO/third_party/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/body_models/soma"
IMG=ca1774d94cd9
LOG="$REPO/runs/logs/t3r_level1_${NAME}.log"
mkdir -p "$REPO/runs/logs"
exec >> "$LOG" 2>&1
say() { echo "=== $(date '+%F %T %Z') $*"; }
fail() { say "T3R_LEVEL1_FAILED $*"; exit 1; }
. "$REPO/stages/recovery/t3r_env.sh"   # T3R_ENV=hers（第 1 级）或 ours（第 1b 级）：定 HPY、GPY，ours 时盖住她的解释器
t3r_env "$REPO" 2>> "$LOG" || fail "另一位成员的目录、解释器或我们的环境没过核对（T3R_ENV=${T3R_ENV:-ours}，见日志）"
export T3R_ROOT="$T3" T3R_ALLOW="$GPY:$HPY"
GUARD="$REPO/stages/recovery/t3r_run.sh"
DK=(docker run --rm --user 1003:1005 --entrypoint "" --network none -e HOME=/tmp -w /workspace/v2d_cari4d/lib/cari4d)
rec() { local n=$1; shift; "$REPO/scripts/dev/record" "t3recon/${NAME}_$n" "$@" > /dev/null 2>&1 || fail "$n（results/t3recon/${NAME}_$n.txt.failed）"; say "$n 过"; }

[ -d "$T3" ] || fail "没有 $T3"
case $FROM in decode|assemble) ;; *) fail "T3R_FROM 只认 decode、assemble" ;; esac
docker image inspect "$IMG" > /dev/null 2>&1 || fail "没有镜像 $IMG"
say "开始，来历 $(cat .v2d-source-commit 2>/dev/null || echo unknown)，根 $T3，镜像 $IMG，从 $FROM 起，环境 ${T3R_ENV:-ours}"
# ep14、ep20 的 world 当时用的文件名
mkdir -p "$T3/runs/floor_gbfix"
cp -n "$T3/runs/floor/floor_fit_gb5.json" "$T3/runs/floor_gbfix/floor_fit_gb3.json"
cp -n "$T3/runs/floor/camera_register_gb5.json" "$T3/runs/floor_gbfix/camera_register_gb3.json"
cp -n "$T3/runs/floor/intrinsics_gb3.json" "$T3/runs/floor_gbfix/intrinsics_gb3.json"

for spec in "14 ep14_round_table work convert_gb5 round_table 4" "20 ep20_rice_cooker work convert_gb5 rice_cooker 4" "27 ep27_woven_basket work_b050 convert_fp_gb5 woven_basket 6"; do
  read -r ep name work conv obj thr <<< "$spec"
  e6="episode_$(printf %06d "$ep")"
  E="$T3/runs/$work/$name/cari4d/$e6"; O="$T3/runs/$conv/$name"
  if [ "$FROM" = assemble ]; then
    for f in object_world.npz soma_params_world.npz world_calib.json; do [ -f "$O/$f" ] || fail "从 assemble 起要已有 $O/$f"; done
  elif [ "$ep" = 27 ]; then   # 她在 t3_sel_full.sh 里的写法：输出挂成 /o/<集名>，另挂地面目录与输入
    M=(-v "$S:/s:ro" -v "$E:/e:ro" -v "$W:/w:ro" -v "$SOMA:/a:ro" -v "$T3/runs/floor:/f:ro" -v "$T3/input:/in:ro" -v "$O:/o/$name")
    rec "decode_$name" "${DK[@]}" --cpus 6 -e OMP_NUM_THREADS=6 "${M[@]}" "$IMG" python /s/t3_convert_ep.py decode /e /w "/o/$name"
    rec "focal_$name" "${DK[@]}" --cpus 6 -e OMP_NUM_THREADS=6 "${M[@]}" "$IMG" python /s/t3_convert_ep.py focal "/o/$name" /f/intrinsics_gb3.json /a "/in/mesh/$obj/$obj.glb"
    rec "world_$name" "${DK[@]}" --cpus 6 -e OMP_NUM_THREADS=6 "${M[@]}" "$IMG" python /s/t3_convert_ep.py world "/o/$name" /f/floor_fit_gb5.json /a /f/camera_register_gb5.json
  else                      # ep14、ep20：decode 照 t3_queue_cari4d.sh，focal、world 照 t3_focal_all.sh 与 t3_gb3_register.sh
    rec "decode_$name" "${DK[@]}" -v "$S:/s:ro" -v "$E:/e:ro" -v "$W:/w:ro" -v "$SOMA:/a:ro" -v "$O:/o" "$IMG" python /s/t3_convert_ep.py decode /e /w /o
    M=(-v "$S:/s:ro" -v "$T3/runs/floor_gbfix:/f:ro" -v "$SOMA:/a:ro" -v "$T3/input:/i:ro" -v "$O:/o/$name")
    rec "focal_$name" "${DK[@]}" --cpus "$thr" -e "OMP_NUM_THREADS=$thr" "${M[@]}" "$IMG" python /s/t3_convert_ep.py focal "/o/$name" /f/intrinsics_gb3.json /a "/i/mesh/$obj/$obj.glb"
    rec "world_$name" "${DK[@]}" --cpus "$thr" -e "OMP_NUM_THREADS=$thr" "${M[@]}" "$IMG" python /s/t3_convert_ep.py world "/o/$name" /f/floor_fit_gb3.json /a /f/camera_register_gb3.json
  fi
  rec "assemble_$name" bash "$GUARD" "$HPY" "$S/t3_convert_ep.py" assemble "runs/$conv/$name" input "$ep" runs/tier3_cari4d_gb5_best
done

physics() {   # physics <版本>：宿主机提取、镜像里检查
  local V=$1
  rec "physics_extract_$V" bash "$GUARD" "$HPY" "$S/t3_physics_check.py" extract "$T3" "$T3/runs/$V" "$T3/runs/physics/$V/npz"
  rec "physics_check_$V" "${DK[@]}" --cpus 8 -v "$S:/s:ro" -v "$T3:/t:ro" -v "$SOMA:/a:ro" -v "$T3/runs/physics/$V:/o" "$IMG" python /s/t3_physics_check.py check /t /o/npz /a /o/out
}
physics tier3_cari4d_gb5_best
rec support bash "$GUARD" env T3_FLOOR_FIT=floor_fit_gb5.json T3_REGISTER=camera_register_gb5.json OMP_NUM_THREADS=4 "$GPY" "$S/t3_support.py" \
  "$T3" tier3_cari4d_gb5_best runs/physics/tier3_cari4d_gb5_best dev/walls runs/support/tier3_cari4d_gb5_best
rec dr2s bash "$GUARD" "$GPY" "$S/t3_depth_post.py" "$T3" tier3_cari4d_gb5_best runs/tier3_cari4d_gb5_best.selection.json \
  runs/support/tier3_cari4d_gb5_best runs/physics/tier3_cari4d_gb5_best tier3_cari4d_gb5_best_dr2s --move-human=3d
physics tier3_cari4d_gb5_best_dr2s
rec dyn7 bash "$GUARD" "$GPY" "$S/t3_dynamics_post.py" "$T3" tier3_cari4d_gb5_best_dr2s runs/physics/tier3_cari4d_gb5_best_dr2s tier3_cari4d_gb5_best_dr2s_dyn7 14,20,27
physics tier3_cari4d_gb5_best_dr2s_dyn7
say "T3R_LEVEL1_DONE"
