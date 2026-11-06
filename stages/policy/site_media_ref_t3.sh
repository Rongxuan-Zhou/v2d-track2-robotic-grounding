#!/usr/bin/env bash
# SOP gallery“人与 G1 同一动作”的 G1 参考栏：按 Tier 3 run 的 run_record 取它训练用的参考 parquet、fc_src 与 scene 覆盖项，
# 用 stages/policy/site_ref_replay.py 把参考第 f 帧直接放进场景（不跑物理，1 世界），无头渲成 1 倍速 640×480 30 fps 的 mp4。
# 计划 agent-doc/plan-gallery-soma-g1-2026-10-29.md 第 1.1 版（复核第 3、6、13 条）。仓库根、宿主机上跑：
#   PY=<带 pyarrow 的 python> stages/policy/site_media_ref_t3.sh <t3 run>...
# 只接受 /repo/runs/motion/tier3_* 的参考；有 runs/.chain-active-* 或 GPU 上有进程就拒绝，自己写 runs/.chain-active-gallery、退出时删掉。
# 输出 runs/site_media/gallery_ref/<run>.{mp4,json,log}；最后一行 SITE_MEDIA_REF_T3_DONE fail=<n>。
set -uo pipefail
[ -x scripts/dev/in-flash-chord ] || { echo "不在仓库根" >&2; exit 2; }
PY=${PY:?PY 必须是带 pyarrow 的 python}
ls runs/.chain-active-* > /dev/null 2>&1 && { echo "有链在跑：$(ls runs/.chain-active-* | tr '\n' ' ')" >&2; exit 6; }
others="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')"
[ -z "$others" ] || { echo "GPU 上有别的进程（PID ${others//$'\n'/,}）" >&2; exit 6; }
MARK=runs/.chain-active-gallery
touch "$MARK"; trap 'rm -f "$MARK"' EXIT
OUT=runs/site_media/gallery_ref; mkdir -p "$OUT"
fail=0
for run in "$@"; do
  [[ $run =~ ^t3-[a-z0-9_-]{2,60}$ ]] || { echo "run 名非法：$run" >&2; exit 2; }
  rec="runs/flash_chord/$run/run_record.jsonl"
  [ -f "$rec" ] || { echo "$run 没有 run_record" >&2; fail=1; continue; }
  read -r pq src ovr <<< "$("$PY" -c 'import json,sys
for l in open(sys.argv[1]):
    r=json.loads(l)
    if r.get("event")=="start":
        print(r["parquet"], r["fc_src"], r.get("overrides","").replace(" ",",")); break' "$rec")"
  [[ $pq == /repo/runs/motion/tier3_*/data.parquet ]] || { echo "$run 的参考不是 Tier 3：$pq" >&2; fail=1; continue; }
  [ -f "$src/SOURCE_V2D.txt" ] || { echo "$run 的 fc_src 不可用：$src" >&2; fail=1; continue; }
  host_pq="runs/${pq#/repo/runs/}"
  frames="$("$PY" -c 'import pyarrow.parquet as pq,sys; t=pq.read_table(sys.argv[1],columns=["robot_root_position"]); print(len(t.column(0)[0]))' "$host_pq")"
  [[ $frames =~ ^[0-9]+$ ]] || { echo "$run 读不出帧数" >&2; fail=1; continue; }
  sup=""; [[ ",$ovr," == *,scene.support=false,* ]] && sup="scene.support=false"
  raw="$OUT/$run.mp4"; log="$OUT/$run.log"; cname="gallery-ref-$run-$$"
  rm -f "$raw" "$OUT/$run.json"
  echo "=== $(date '+%F %T %Z') $run frames=$frames $sup fc_src=$src"
  cmd="export PYGLET_HEADLESS=1 && cd /workspace && python /repo/stages/policy/site_ref_replay.py --config-name=sonic_replay \"task.parquet='$pq'\" viewer.backend=mp4 viewer.output_path=/repo/$raw viewer.num_frames=$frames viewer.video_fps=30 viewer.realtime=false viewer.width=640 viewer.height=480 viewer.headless=true start_paused=false markers.enabled=false $sup"
  V2D_FC_SRC="$src" V2D_GPU=1 V2D_CONTAINER_NAME="$cname" timeout 1800 scripts/dev/in-flash-chord "$cmd" > "$log" 2>&1
  rc=$?
  if [ "$rc" -ne 0 ]; then docker stop "$cname" > /dev/null 2>&1 || true; echo "$run 渲染退出码 $rc，见 $log（末行：$(tail -1 "$log")）"; fail=1; fi
  grep -h -E '^(Video:|REF_SUMMARY)' "$log" || true
  if [ -s "$raw" ]; then echo "$run → $raw $(stat -c %s "$raw") B，参考 $frames 帧"; else echo "$run 没有渲染输出"; fail=1; fi
done
echo "SITE_MEDIA_REF_T3_DONE fail=$fail"
exit $fail
