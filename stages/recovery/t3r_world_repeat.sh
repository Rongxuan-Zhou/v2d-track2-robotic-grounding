#!/usr/bin/env bash
# t3takeover 判据 K2 的排查：把一集的 world 步在我们的目录里再跑一遍，测这一步是否逐位确定（训练机 ml40 上执行；只用 CPU）。
# 计划 agent-doc/plan-t3takeover-v3-2026-10-21.md 判据 K2（先查环境与线程）与复核第 2 条（重复跑不逐位相同时，以两次之差作噪声底）。
# 输入：第 1 级根 t3 里这一集 focal 之后的 4 个文件（object_cam_focal.npz、soma_params_focal.npz、focal_calib.json、residual_report.json，
# 前三个与参照逐字节相同，第四个是复制来的参照）；镜像、挂载与文件名照 stages/recovery/t3r_level1.sh（ep14、ep20 用 floor_gbfix 下的 *_gb3 名字，
# ep27 用 floor 下的 gb5 名字）。docker 只挂我们的目录、--user 1003:1005、--network none。
# 输出 /lab_data/rxzhou/v2d/t3recon/rep_world/<标签>/<集名>/，已存在就拒绝；跑完用 t3r_compare.py --all 把 world 的三个产物与 t3 根、参照各比一次
# （比较器经 stages/recovery/t3r_run.sh 跑，解释器由 t3r_env.sh 定：默认我们的宿主机环境）。比较有不同不算失败，最后一行 T3R_WORLD_REPEAT_DONE。
# 用法：stages/recovery/t3r_world_repeat.sh <14|20|27> <线程数> <标签> [额外的 docker 参数…]
#   例：… 27 6 a；… 27 6 b；… 14 4 a；… 27 6 c -e MKL_NUM_THREADS=1
set -uo pipefail
cd "$(dirname "$0")/../.."
REPO="$(pwd)"
EP=${1:?集号}; THR=${2:?线程数}; TAG=${3:?标签}; shift 3
R=/lab_data/rxzhou/v2d/t3recon; T3="$R/t3"; S="$R/code/stages/recovery/tier3"
SOMA="$REPO/third_party/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/body_models/soma"
IMG=ca1774d94cd9
case $EP in
  14) name=ep14_round_table; conv=convert_gb5; F="$T3/runs/floor_gbfix"; ff=floor_fit_gb3.json; rg=camera_register_gb3.json ;;
  20) name=ep20_rice_cooker; conv=convert_gb5; F="$T3/runs/floor_gbfix"; ff=floor_fit_gb3.json; rg=camera_register_gb3.json ;;
  27) name=ep27_woven_basket; conv=convert_fp_gb5; F="$T3/runs/floor"; ff=floor_fit_gb5.json; rg=camera_register_gb5.json ;;
  *) echo "T3R_WORLD_REPEAT_FAILED 只认 14、20、27"; exit 2 ;;
esac
O="$R/rep_world/$TAG/$name"
[ -e "$O" ] && { echo "T3R_WORLD_REPEAT_FAILED $O 已存在"; exit 2; }
. "$REPO/stages/recovery/t3r_env.sh"   # 比较器的解释器：默认我们的环境（1b 之后），T3R_ENV=hers 时是她的
t3r_env "$REPO" || { echo "T3R_WORLD_REPEAT_FAILED 另一位成员的目录、解释器或我们的环境没过核对"; exit 2; }
mkdir -p "$O"
for f in object_cam_focal.npz soma_params_focal.npz focal_calib.json residual_report.json; do
  cp "$T3/runs/$conv/$name/$f" "$O/" || { echo "T3R_WORLD_REPEAT_FAILED 缺 $f"; exit 2; }
done
echo "=== $(date '+%F %T %Z') world $name，--cpus $THR、OMP_NUM_THREADS=$THR，标签 $TAG，额外参数：${*:-无}"
docker run --rm --user 1003:1005 --entrypoint "" --network none -e HOME=/tmp -w /workspace/v2d_cari4d/lib/cari4d \
  --cpus "$THR" -e "OMP_NUM_THREADS=$THR" "$@" \
  -v "$S:/s:ro" -v "$SOMA:/a:ro" -v "$F:/f:ro" -v "$O:/o/$name" "$IMG" \
  python /s/t3_convert_ep.py world "/o/$name" "/f/$ff" /a "/f/$rg" || { echo "T3R_WORLD_REPEAT_FAILED world 步"; exit 1; }
echo "=== $(date '+%F %T %Z') world 跑完"
export T3R_ROOT="$R" T3R_ALLOW="$HPY"
for ref in t3 ref; do
  p=(); for f in object_world.npz soma_params_world.npz world_calib.json; do p+=("$O/$f" "$R/$ref/runs/$conv/$name/$f"); done
  echo "--- 对 $ref"
  bash "$REPO/stages/recovery/t3r_run.sh" "$HPY" "$REPO/stages/recovery/t3r_compare.py" --all "${p[@]}"
done
echo "T3R_WORLD_REPEAT_DONE"
