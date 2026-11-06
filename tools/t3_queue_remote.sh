# 服务器上的只读快照：同组成员在 ml40 上跑的 Track 2 Tier 3 CARI4D 重建队列走到哪一集、各集的检查结论与产物。经 stdin 送过去执行。
# 用途：估计 ml40 的 L40S 还要被占多久，以及哪些集的重建已经过了检查、可以接着 retarget 和训练；10/11 起另报重新选物与每帧重新配准的日志尾部与跟丢检查。
# 只读：不改文件、不进别人的容器、不读环境变量。队列目录按 /lab_data/<目录>/v2d/tier3 找，跳过本用户的目录。
# 记录要进 git：别人的目录名、用户名与 home 目录名一律换成 <其他用户>（同 tools/gpu_procs_remote.sh）。
# 用法（Mac，仓库根）：scripts/dev/record ops/t3_queue_ml40_<时间> ssh ml40 'bash -s' < tools/t3_queue_remote.sh
set -uo pipefail
me="$(id -un)"
mapfile -t roots < <(ls -d /lab_data/*/v2d/tier3 2>/dev/null | grep -v "^/lab_data/$me/")
names=()
for T in "${roots[@]}"; do
  d="${T#/lab_data/}"; d="${d%%/*}"; o="$(stat -c %U "$T")"
  h="$(getent passwd "$o" | cut -d: -f6)"; h="${h##*/}"
  names+=("$d" "$o" "$h")
done
# 长的名字先换，免得短名字是长名字的一部分时只换掉一半
mapfile -t names < <(printf '%s\n' "${names[@]}" | awk 'NF && $0 != "'"$me"'"' | sort -u | awk '{ print length, $0 }' | sort -rn | cut -d' ' -f2-)
red() { local s; s="$(cat)"; for u in "${names[@]}"; do s="${s//$u/<其他用户>}"; done; printf '%s\n' "$s"; }
{
  echo "== $(hostname) $(date '+%F %T %Z')"
  nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
  echo "## GPU 计算进程（pid、显存）"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
  [ ${#roots[@]} -gt 0 ] || echo "没有找到别的用户的 tier3 目录"
  for T in "${roots[@]}"; do
    o="$(stat -c %U "$T")"
    echo "## 队列目录 $T"
    echo "## 脚本类进程（pid、已运行、开始时间、命令行）"
    ps -u "$o" -o pid=,etime=,lstart=,args= 2>/dev/null | grep -E '\.sh|\.py|docker' | grep -v grep | cut -c1-400
    echo "## PAUSE 文件：$([ -e "$T/PAUSE" ] && echo 有 || echo 无)"
    echo "## logs 下最近修改的 10 个文件"
    ls -lt --time-style=+%F_%T "$T/logs" 2>/dev/null | sed -n 2,11p
    for f in "$T/logs/after_validation.log" "$T/logs/t3_cari4d_queue.log" "$T/logs/t3_reselect.log" "$T/logs/t3_fp_pilot.log" "$T/logs/t3_gpu_watch.log"; do
      echo "## $f（最后 40 行）"
      tail -n 40 "$f" 2>/dev/null | cut -c1-300
    done
    # 10/11 起：重新选物（track_check_v2sel）与每帧重新配准（track_check_fp）各有一份跟丢检查，与基线（track_check）并列
    for k in track_check track_check_v2sel track_check_fp; do
      echo "## 跟丢检查（runs/$k/*.json：集、lost_fraction、顶层键）"
      for c in "$T"/runs/$k/*.json; do
        [ -f "$c" ] || continue
        echo "$(basename "$c" .json) $(python3 -c 'import json,sys
j = json.load(open(sys.argv[1]))
print("lost_fraction=%s keys=%s" % (j.get("lost_fraction"), ",".join(sorted(j))[:200]))' "$c")"
      done
    done
    for k in checks checks_v2 checks_fp; do
      echo "## 各集剪影检查（runs/$k/*/object_check.json：ok、可见 IoU / 深度因子 / 质心偏移的中位、原因）"
      for c in "$T"/runs/$k/*/object_check.json; do
        [ -f "$c" ] || continue
        echo "$(basename "$(dirname "$c")") $(python3 -c 'import json,sys
j = json.load(open(sys.argv[1])); r = j.get("all_p10_p50_p90") or {}
g = lambda k: (r.get(k) or [None] * 3)[1]
print("ok=%s iou_p50=%s depth_factor_p50=%s centroid_p50=%s %s" % (j.get("ok"), g("iou_visible"), g("depth_factor"), g("centroid_px"), j.get("why") or ""))' "$c")"
      done
    done
    c="$(ls "$T"/runs/checks_v2/*/object_check.json 2>/dev/null | head -1)"
    [ -n "$c" ] && echo "## checks_v2 报告的顶层键：$(python3 -c 'import json,sys; print(sorted(json.load(open(sys.argv[1]))))' "$c")"
    echo "## 各集 pipeline_report 结论"
    for p in "$T"/runs/work/*/cari4d/episode_*/pipeline_report.json; do
      [ -f "$p" ] || continue
      echo "$(basename "$(dirname "$(dirname "$(dirname "$p")")")") $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("verdict"))' "$p")"
    done
    # 不只列 tier3_cari4d 与 tier3_cari4d_suspect：地面、焦距校正后的版本也在 runs/tier3_cari4d* 下
    for p in "$T"/runs/tier3_cari4d*/; do
      [ -d "$p" ] || continue
      d="$(basename "$p")"
      echo "## runs/$d 里的 parquet（$(find "$p" -name '*.parquet' 2>/dev/null | wc -l) 个；文件名、修改时间）"
      find "$p" -name '*.parquet' -printf '%f %TF_%TT\n' 2>/dev/null | cut -c1-42 | sort
    done
    echo "## 部署版本与队列脚本修改时间"
    cat "$T/code/DEPLOYED" 2>/dev/null
    ls -l --time-style=+%F_%T "$T"/code/stages/recovery/tier3/*.sh 2>/dev/null
  done
  echo "## /lab_data"
  df -h /lab_data | tail -1
} 2>&1 | red
