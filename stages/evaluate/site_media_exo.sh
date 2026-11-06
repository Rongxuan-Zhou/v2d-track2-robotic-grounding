#!/usr/bin/env bash
# SOP“任务一览”的人类示范片段：exo 相机视频（三档逐字节相同，9/30 review 时 tier_1 与 tier_2 逐条 cmp 0 条不同；读 tier_2 目录，新代码里不出现 Tier 1 路径）转成 2 倍速、640 宽、H.264 CRF 28、无音轨、faststart，
# 另从原视频中点截一帧作封面。计划 agent-doc/plan-site-media-2026-09-30.md。
# 在 FlashCHORD 容器里跑，只用 CPU，ffmpeg 用镜像里 imageio_ffmpeg 自带的那个（宿主机没有 ffmpeg）：
#   V2D_GPU=0 scripts/dev/in-flash-chord "bash /repo/stages/evaluate/site_media_exo.sh"
# 输出 /repo/runs/site_media/exo_epNN.{mp4,jpg}（runs/ 不进 git）；已存在的跳过，先写临时文件再改名。
set -euo pipefail
FF="$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"
SRC=/repo/data/challenge/track_2/tier_2_synthetic_noise/videos/chunk-000/observation.images.exo_camera
OUT=/repo/runs/site_media
mkdir -p "$OUT"
n_ok=0
for f in "$SRC"/episode_*.mp4; do
  idx=$((10#$(basename "$f" .mp4 | sed 's/^episode_//')))
  ep=$(printf 'ep%02d' "$idx")
  if [ ! -s "$OUT/exo_$ep.mp4" ]; then
    "$FF" -nostdin -loglevel error -y -i "$f" -an -map_metadata -1 -vf "setpts=0.5*PTS,scale=640:-2:flags=lanczos" -r 30 \
      -c:v libx264 -preset slow -crf 28 -pix_fmt yuv420p -movflags +faststart "$OUT/.exo_$ep.tmp.mp4"
    mv "$OUT/.exo_$ep.tmp.mp4" "$OUT/exo_$ep.mp4"
  fi
  if [ ! -s "$OUT/exo_$ep.jpg" ]; then
    mid="$(python -c "import imageio_ffmpeg as f; print(f.count_frames_and_secs('$f')[1] / 2)")"
    "$FF" -nostdin -loglevel error -y -ss "$mid" -i "$f" -frames:v 1 -map_metadata -1 -vf "scale=640:-2:flags=lanczos" -q:v 6 "$OUT/.exo_$ep.tmp.jpg"
    mv "$OUT/.exo_$ep.tmp.jpg" "$OUT/exo_$ep.jpg"
  fi
  n_ok=$((n_ok + 1))
  echo "exo $ep $(stat -c %s "$OUT/exo_$ep.mp4") B"
done
echo "SITE_MEDIA_EXO_DONE $n_ok"
