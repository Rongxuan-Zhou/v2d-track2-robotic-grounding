#!/usr/bin/env bash
# t3takeover 第 1 级 A 部分的比较（训练机 ml40 上执行；比较器经 stages/recovery/t3r_run.sh 用她的宿主机解释器只读地跑）。计划 agent-doc/plan-t3takeover-v3-2026-10-21.md 判据 K1。
# 先跑三项对照与门槛自检，任何一项不对就整轮不判：
#   自检：t3r_compare.py --self-test 用 ref 里 ep27 的 dyn7 parquet（改一个字节判不同、挪 2e-6 判不同、取反四元数判相同）；
#   负对照：ctl/gb4 里 ep27 的三版 parquet 对 ref 的 gb5 同版，必须 DIFFERENT；
#   已知相同对：ctl/eq 里 ep14、ep20 的 gb4 物理 out npz 对 ref 的 gb5 同版，必须 BYTES_EQUAL。
# 再按段落比我们的产物与参照（decode、focal、world、assemble、物理、support、dr2s、dyn7；多集的 json 用 --subset），每段一份记录。
# 用法：stages/recovery/t3r_level1_compare.sh <根名，t3 或 t3_rep> [<对照根名>]   不给对照根名时与 ref 比；给了就两个根互比（正对照：同步骤跑两遍）。
# 记录 results/t3recon/<根名>_vs_<对照根名>_cmp[_<T3R_TAG>]_<段>.txt，最后一行 T3R_L1_COMPARE_DONE（各段的结论计数）。
# T3R_ENV=ours：比较器用我们的宿主机环境、她的解释器被盖住（第 1b 级）；T3R_TAG 给记录名加后缀，回归与 1b 的比较不覆盖第 1 级的记录（1b 复核改动第 2 条）。
set -uo pipefail
cd "$(dirname "$0")/../.."
REPO="$(pwd)"
NAME=${1:?根名}; OTHER=${2:-ref}
R=/lab_data/rxzhou/v2d/t3recon; T3="$R/$NAME"; REF="$R/$OTHER"; CTL="$R/ctl"
. "$REPO/stages/recovery/t3r_env.sh"
t3r_env "$REPO" || { echo "T3R_L1_COMPARE_FAILED 另一位成员的目录、解释器或我们的环境没过核对（T3R_ENV=${T3R_ENV:-ours}）"; exit 2; }
export T3R_ROOT="$R" T3R_ALLOW="$HPY"
GUARD="$REPO/stages/recovery/t3r_run.sh"
CMP="$REPO/stages/recovery/t3r_compare.py"
run() { local n=$1; shift; "$REPO/scripts/dev/record" "t3recon/${NAME}_vs_${OTHER}_cmp${T3R_TAG:+_$T3R_TAG}_$n" bash "$GUARD" "$HPY" "$CMP" "$@" > /dev/null 2>&1; echo "$n 退出码 $?"; }
V=(tier3_cari4d_gb5_best tier3_cari4d_gb5_best_dr2s tier3_cari4d_gb5_best_dr2s_dyn7)
EPS=("14 ep14_round_table convert_gb5" "20 ep20_rice_cooker convert_gb5" "27 ep27_woven_basket convert_fp_gb5")
if [ "$OTHER" = ref ]; then
  p=(); for v in "${V[@]}"; do p+=("$CTL/gb4/${v/gb5/gb4}/episode_000027.parquet" "$REF/runs/$v/data/chunk-000/episode_000027.parquet"); done
  run controls_negative "${p[@]}"
  p=(); for v in tier3_cari4d_gb4_best tier3_cari4d_gb4_best_dr2s; do for n in ep14_round_table ep20_rice_cooker; do
    p+=("$CTL/eq/$v/$n.npz" "$REF/runs/physics/${v/gb4/gb5}/out/$n.npz"); done; done
  run controls_known_equal "${p[@]}"
  run selftest --self-test "$REF/runs/${V[2]}/data/chunk-000/episode_000027.parquet" "$REF/runs/${V[2]}/data/chunk-000/episode_000027.parquet"
fi
for stage in decode focal world; do
  case $stage in
    decode) F=(object_cam.npz decode_report.json) ;;
    focal) F=(object_cam_focal.npz soma_params_focal.npz focal_calib.json) ;;
    world) F=(object_world.npz soma_params_world.npz world_calib.json) ;;
  esac
  p=(); for e in "${EPS[@]}"; do read -r ep name conv <<< "$e"; for f in "${F[@]}"; do p+=("$T3/runs/$conv/$name/$f" "$REF/runs/$conv/$name/$f"); done; done
  run "$stage" "${p[@]}"
done
for v in "${V[@]}"; do
  p=(); for e in "${EPS[@]}"; do read -r ep name conv <<< "$e"; e6="episode_$(printf %06d "$ep")"
    p+=("$T3/runs/$v/data/chunk-000/$e6.parquet" "$REF/runs/$v/data/chunk-000/$e6.parquet"
        "$T3/runs/$v/meta/$e6.metadata.json" "$REF/runs/$v/meta/$e6.metadata.json"
        "$T3/runs/physics/$v/npz/$e6.npz" "$REF/runs/physics/$v/npz/$e6.npz"
        "$T3/runs/physics/$v/out/$name.npz" "$REF/runs/physics/$v/out/$name.npz"
        "$T3/runs/physics/$v/out/$name.json" "$REF/runs/physics/$v/out/$name.json"); done
  run "data_$v" "${p[@]}"
  run "summary_$v" --subset "$T3/runs/physics/$v/out/summary.json" "$REF/runs/physics/$v/out/summary.json"
done
p=(); for e in "${EPS[@]}"; do read -r ep name conv <<< "$e"; p+=("$T3/runs/support/tier3_cari4d_gb5_best/$name.json" "$REF/runs/support/tier3_cari4d_gb5_best/$name.json"); done
run support "${p[@]}"
run post_json --subset "$T3/runs/tier3_cari4d_gb5_best_dr2s.depth_post.json" "$REF/runs/tier3_cari4d_gb5_best_dr2s.depth_post.json" \
  "$T3/runs/tier3_cari4d_gb5_best_dr2s_dyn7.dynamics_post.json" "$REF/runs/tier3_cari4d_gb5_best_dr2s_dyn7.dynamics_post.json"
echo "T3R_L1_COMPARE_DONE（各段结论见 results/t3recon/${NAME}_vs_${OTHER}_cmp${T3R_TAG:+_$T3R_TAG}_*.txt）"
