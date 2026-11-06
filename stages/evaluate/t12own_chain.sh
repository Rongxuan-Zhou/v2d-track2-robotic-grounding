#!/usr/bin/env bash
# t12own 链（ml40 上、仓库根执行，放 tmux；计划 agent-doc/plan-t12ctl-2026-10-20.md 执行记录里的 C3）。
# 用法：stages/evaluate/t12own_chain.sh <tag> <序列名> <第一次接触帧>
# 1. 用前等卡上没有进程（每分钟看一次，最多 2 小时；不停任何进程）；fc_run.sh 训一条冒烟 smoke-t2-<tag>-w2048-cap4m-r3a-sd-ml40（照 ep4-sd 的冒烟行：
#    2048 世界、4096000 步、buffer 400 万、scene.support=false、变体 A、每 409600 步存一份；挂 v040_p185_nfg；不记 W&B、不重试）。
# 2. residual_head_norm.py 量第 409600 步：state.0015、state.0018、state.0019 的 max|.| 都要是 0，否则停（不是零残差）。
# 3. diag_rollout.py --worlds 1024 --palms --bodies（不换参考，V2D_GPU=1，挂 fc_8c778f9）→ diag2_sync.py → fall_mechanism.py --before <接触帧>。
# 4. 镜像对照：t12ctl_chain.sh 用这份 checkpoint 配 ep4 原参考（记录在 results/t12ctl/）。
# 记录在 results/t12own/，日志 runs/logs/t12own_ml40.log，最后一行 T12OWN_DONE 或 T12OWN_FAILED <原因>。
set -uo pipefail
cd "$(dirname "$0")/../.."
TAG=${1:?tag}; SEQ=${2:?序列名}; TC=${3:?接触帧}
LOG=runs/logs/t12own_ml40.log
exec >> "$LOG" 2>&1
say() { echo "=== $(date '+%F %T %Z') $*"; }
fail() { say "T12OWN_FAILED $*"; exit 1; }
TRAIN_SRC=/lab_data/rxzhou/v2d/fc_src/v040_p185_nfg
EVAL_SRC=/lab_data/rxzhou/v2d/fc_src/fc_8c778f9
RUN="smoke-t2-${TAG}-w2048-cap4m-r3a-sd-ml40"
Q="runs/motion/tier2/whole_body/soma/sequence_id=$SEQ/robot_name=g1/data.parquet"
P="runs/flash_chord/$RUN/policy_409600.safetensors"
touch runs/.chain-active-t12own_ml40; trap 'rm -f runs/.chain-active-t12own_ml40' EXIT
wait_gpu() { local t0=$SECONDS; while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')" ]; do
  [ $((SECONDS - t0)) -lt 7200 ] || return 1; sleep 60; done; }
SRCC="$(cat .v2d-source-commit 2>/dev/null || echo unknown)"
case "$SRCC" in *-syncing) fail "来历标记是 ${SRCC}，同步还没完" ;; esac   # 复核 MINOR 3
say "开始，来历 ${SRCC}；${RUN}，接触帧 ${TC}"
[ -f "$Q" ] || fail "没有 $Q"
[ ! -e "runs/flash_chord/$RUN" ] || fail "runs/flash_chord/$RUN 已存在"
wait_gpu || fail "GPU 等了 2 小时还被占"
V2D_WANDB=0 V2D_MAX_RETRY=0 V2D_WANDB_UPLOAD_CKPT=0 V2D_GPU_APPS=1 V2D_FC_SRC=$TRAIN_SRC \
  stages/policy/fc_run.sh "$RUN" "/repo/$Q" 2048 4096000 training.replay.capacity=4000000 scene.support=false \
  objective.object_position_std=0.05 termination.object_orientation_threshold=0.6 training.checkpoint.save_interval_environment_steps=409600 \
  > "runs/logs/${RUN}.host.log" 2>&1 || fail "冒烟没成（runs/logs/${RUN}.host.log）"
[ -f "$P" ] || fail "没有 $P"
scripts/dev/record "t12own/head_$TAG" scripts/dev/in-flash-chord "python /repo/stages/evaluate/residual_head_norm.py /repo/$P" > /dev/null 2>&1 || fail "residual_head_norm 没成"
H="results/t12own/head_$TAG.txt"   # 按形状数，不按匿名编号（复核 MINOR 2）：全为 0 的 (256, 43) 至少 1 个、(43,) 至少 2 个
z256=$(grep -cE "\(256, 43\)[[:space:]]+max\|\.\| 0\.000e\+00" "$H"); z43=$(grep -cE "\(43,\)[[:space:]]+max\|\.\| 0\.000e\+00" "$H")
[ "$z256" -ge 1 ] && [ "$z43" -ge 2 ] || fail "全为 0 的输出头张量 (256,43) ${z256} 个、(43,) ${z43} 个，不是零残差"
say "残差均值头为 0"
wait_gpu || fail "GPU 等了 2 小时还被占"
out="runs/t12own/${RUN}_s409600"; mkdir -p runs/t12own
V2D_GPU=1 V2D_FC_SRC=$EVAL_SRC scripts/dev/record "t12own/rollout_$TAG" scripts/dev/in-flash-chord \
  "cd /workspace && python /repo/stages/evaluate/diag_rollout.py /repo/$P /repo/$out --worlds 1024 --palms --bodies" > /dev/null 2>&1 || fail "rollout 没成"
rm -rf "${out}_hydra"
V2D_GPU=0 V2D_FC_SRC=$EVAL_SRC scripts/dev/record "t12own/sync_$TAG" scripts/dev/in-flash-chord "python /repo/stages/evaluate/diag2_sync.py /repo/$out.npz" > /dev/null 2>&1 || say "同步闸不过"
V2D_GPU=0 scripts/dev/record "t12own/mech_$TAG" scripts/dev/in-flash-chord \
  "python /repo/stages/evaluate/fall_mechanism.py --npz /repo/$out.npz --motion /repo/$Q --label $TAG --before $TC" > /dev/null 2>&1 || fail "fall_mechanism 没成"
say "${TAG}：$(grep -h '^FALL_BEFORE' "results/t12own/mech_$TAG.txt" | tr '\n' ' ')；$(grep -h '摔倒帧分位' "results/t12own/mech_$TAG.txt" | head -1 | cut -c1-90)"
# 镜像对照（复核 MAJOR 1）：这份新 checkpoint 配 ep4 原参考，要复现 ep4-sd 的摔倒帧 88/88/89（±2）
stages/evaluate/t12ctl_chain.sh "$RUN:409600" "ep4own:2026-04-24_11-40-52_baseball_bat_rotate_05:133" \
  && say "镜像对照：$(grep -a 'ep4own：' runs/logs/t12ctl_ml40.log | tail -1 | cut -c1-200)" || say "镜像对照没跑成"
say "T12OWN_DONE"
