#!/usr/bin/env bash
# SOP“任务一览”的 G1 参考动画：stages/policy/site_ref_replay.py 把参考第 f 帧直接放进场景（不跑物理），mp4 viewer 无头渲染；
# 再转成 2 倍速 H.264 CRF 32（去掉元数据），截中点作封面。计划 agent-doc/plan-site-media-2026-09-30.md（第 2 版）。
# 宿主机上、仓库根跑（放 tmux）：stages/policy/site_media_ref.sh <ep 序号>...
#   V2D_FC_SRC=<#185 副本> 时挂补丁导入器（scripts/dev/in-flash-chord）；SUFFIX=_v040 之类给输出文件名加后缀。
# 需要 runs/site_media/task_table.tsv（python3 stages/evaluate/site_task_table.py --tsv runs/site_media/task_table.tsv）。
# 9/30 probe 实测需要的绕过：PYGLET_HEADLESS=1（否则 pyglet 找 X display）；markers.enabled=false（标记点绘制在当前 Newton 上报类型错）；
# 支撑面文件是空 Xform 的序列加 scene.support=false（与训练相同）；任务表里支撑面是 missing 的直接报错，不猜。
# 输出 runs/site_media/ref_epNN<后缀>.{mp4,jpg,json}；原始渲染与日志在 runs/site_media/raw/；已存在的跳过。
# 超时（每条 30 分钟）按容器名 docker stop，免得 timeout 只杀掉 docker 客户端、容器还在占卡。
set -uo pipefail
[ -x scripts/dev/in-flash-chord ] || { echo "不在仓库根" >&2; exit 2; }
TABLE=runs/site_media/task_table.tsv
[ -f "$TABLE" ] || { echo "缺 $TABLE" >&2; exit 2; }
SUFFIX="${SUFFIX:-}"
[[ "$SUFFIX" =~ ^[A-Za-z0-9_]*$ ]] || { echo "SUFFIX 非法" >&2; exit 2; }
mkdir -p runs/site_media/raw
fail=0
for n in "$@"; do
  [[ "$n" =~ ^[0-9]+$ ]] || { echo "ep 序号非法：$n" >&2; exit 2; }
  ep=$(printf 'ep%02d' "$n"); tag="ref_${ep}${SUFFIX}"
  out="runs/site_media/$tag.mp4"
  if [ -s "$out" ] && [ -s "runs/site_media/$tag.jpg" ] && [ -s "runs/site_media/$tag.json" ]; then echo "$ep 已有，跳过"; continue; fi
  row="$(awk -F'\t' -v n="$n" '$1 == n' "$TABLE")"
  seq="$(cut -f2 <<< "$row")"; frames="$(cut -f5 <<< "$row")"; support="$(cut -f6 <<< "$row")"
  [ -n "$seq" ] && [[ "$frames" =~ ^[0-9]+$ ]] || { echo "$ep 不在任务表里" >&2; fail=1; continue; }
  [ "$support" != missing ] || { echo "$ep 支撑面文件缺失，跳过" >&2; fail=1; continue; }
  sup=""; [ "$support" = none ] && sup="scene.support=false"
  pq="/repo/runs/motion/tier1/whole_body/soma/sequence_id=$seq/robot_name=g1/data.parquet"
  raw="runs/site_media/raw/$tag.mp4"; log="runs/site_media/raw/$tag.log"; cname="sitemedia-$tag-$$"
  rm -f "$raw" "runs/site_media/raw/$tag.json"
  echo "=== $(date '+%F %T %Z') $ep $seq frames=$frames support=$support V2D_FC_SRC=${V2D_FC_SRC:-}"
  cmd="export PYGLET_HEADLESS=1 && cd /workspace && python /repo/stages/policy/site_ref_replay.py --config-name=sonic_replay \"task.parquet='$pq'\" viewer.backend=mp4 viewer.output_path=/repo/$raw viewer.num_frames=$frames viewer.video_fps=30 viewer.realtime=false viewer.width=640 viewer.height=480 viewer.headless=true start_paused=false markers.enabled=false $sup"
  V2D_GPU=1 V2D_CONTAINER_NAME="$cname" timeout 1800 scripts/dev/in-flash-chord "$cmd" > "$log" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then
    docker stop "$cname" > /dev/null 2>&1 || true
    echo "$ep 渲染退出码 $rc，见 $log（末行：$(tail -1 "$log")）"; fail=1
  fi
  grep -h -E '^(Video:|REF_SUMMARY)' "$log" || true
  if [ ! -s "$raw" ]; then echo "$ep 没有渲染输出"; fail=1; continue; fi
  conv="FF=\$(python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())') && \
\$FF -nostdin -loglevel error -y -i /repo/$raw -an -map_metadata -1 -vf 'setpts=0.5*PTS' -r 30 -c:v libx264 -preset slow -crf 32 -pix_fmt yuv420p -movflags +faststart /repo/runs/site_media/.$tag.tmp.mp4 && \
mv /repo/runs/site_media/.$tag.tmp.mp4 /repo/$out && \
mid=\$(python -c \"import imageio_ffmpeg as f; print(f.count_frames_and_secs('/repo/$raw')[1] / 2)\") && \
\$FF -nostdin -loglevel error -y -ss \$mid -i /repo/$raw -frames:v 1 -map_metadata -1 -q:v 6 /repo/runs/site_media/.$tag.tmp.jpg && \
mv /repo/runs/site_media/.$tag.tmp.jpg /repo/runs/site_media/$tag.jpg"
  V2D_GPU=0 scripts/dev/in-flash-chord "$conv" > "runs/site_media/raw/conv_$tag.log" 2>&1
  [ -s "runs/site_media/raw/$tag.json" ] && cp "runs/site_media/raw/$tag.json" "runs/site_media/$tag.json"
  if [ -s "$out" ] && [ -s "runs/site_media/$tag.jpg" ]; then
    echo "$ep → $out $(stat -c %s "$out") B"
  else
    echo "$ep 转码失败，见 runs/site_media/raw/conv_$tag.log"; fail=1
  fi
done
echo "SITE_MEDIA_REF_DONE fail=$fail"
exit $fail
