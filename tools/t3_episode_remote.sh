# 服务器上的只读核对：同组成员的 Tier 3 重建产物（/lab_data/<目录>/v2d/tier3/runs/tier3_cari4d）里给定几集的文件、本用户能不能读、元数据内容。
# 经 stdin 送过去执行。只读：不改文件、不进容器。别人的目录名、用户名与 home 目录名一律换成 <其他用户>（同 tools/t3_queue_remote.sh）。
# V2D_T3_SUBDIR 选目录，默认 tier3_cari4d；白名单与 stages/recovery/tier3_pack_remote.sh 相同（计划 agent-doc/plan-t3-pilot-v2-2026-10-10.md；10/14 加 best_dyn4、best_dyn4s，计划 agent-doc/plan-t3dyn-pilot-2026-10-14.md；10/17 加 gb3 交付版 tier3_cari4d_gb_best_dr2s_dyn7，计划 agent-doc/plan-t3-gb3-switch-2026-10-17.md）。
# V2D_T3_SUPPORT=delivery（只许空或 delivery；计划 agent-doc/plan-t3-hm-p2-2026-10-12.md）：每集多打一行支撑面文件，格式同 parquet 行
#   “可读 … sha256 <64 位> support_surface/<文件名>.usda”（路径不合要求或读不了另打一行说明）；另打 support_surface/support_usda.json 的
#   sha256 与内容，那一行以 “## ” 开头，不会被 tier3_expect.py 当成文件行。不设时输出与以前相同。
# V2D_T3_OWN=<本用户 /lab_data/<本用户>/v2d/ 下的目录>（同 stages/recovery/tier3_pack_remote.sh）：读 $V2D_T3_OWN/runs/$V2D_T3_SUBDIR（只许 tier3_kfix_*），
#   输入网格与地面仍按同组成员的 tier3/input 核 md5。不设时行为不变。
# 用法（Mac，仓库根）：scripts/dev/record ops/<名字> ssh ml40 '[V2D_T3_SUBDIR=<目录>] bash -s -- <episode> ...' < tools/t3_episode_remote.sh
set -uo pipefail
me="$(id -un)"
mapfile -t roots < <(ls -d /lab_data/*/v2d/tier3 2>/dev/null | grep -v "^/lab_data/$me/")
names=()
for T in "${roots[@]}"; do
  d="${T#/lab_data/}"; d="${d%%/*}"; o="$(stat -c %U "$T")"
  h="$(getent passwd "$o" | cut -d: -f6)"; h="${h##*/}"
  names+=("$d" "$o" "$h")
done
mapfile -t names < <(printf '%s\n' "${names[@]}" | awk 'NF && $0 != "'"$me"'"' | sort -u | awk '{ print length, $0 }' | sort -rn | cut -d' ' -f2-)
red() { local s; s="$(cat)"; for u in "${names[@]}"; do s="${s//$u/<其他用户>}"; done; printf '%s\n' "$s"; }
{
  echo "== $(hostname) $(date '+%F %T %Z')"
  [ ${#roots[@]} = 1 ] || { echo "别的用户的 tier3 目录不是恰好一个：${roots[*]:-无}"; exit 1; }
  SUB="${V2D_T3_SUBDIR:-tier3_cari4d}"
  case "$SUB" in tier3_cari4d|tier3_cari4d_suspect|tier3_cari4d_floor|tier3_cari4d_floor_suspect|tier3_cari4d_f830_floor|tier3_cari4d_f830_floor_suspect|tier3_cari4d_f830_simreg|tier3_cari4d_f830_simreg_suspect|tier3_cari4d_f830_best|tier3_cari4d_f830_best_suspect|tier3_cari4d_f830_best_hm|tier3_cari4d_f830_best_hm_suspect|tier3_cari4d_f830_best_dyn4|tier3_cari4d_f830_best_dyn4_suspect|tier3_cari4d_f830_best_dyn4s|tier3_cari4d_f830_best_dyn4s_suspect|tier3_cari4d_gb_best_dr2s_dyn7|tier3_cari4d_gb_best_dr2s_dyn7_suspect|tier3_cari4d_gb4_best_dr2s_dyn7|tier3_cari4d_gb4_best_dr2s_dyn7_suspect|tier3_cari4d_gb5_best_dr2s_dyn7|tier3_cari4d_gb5_best_dr2s_dyn7_suspect|tier3_kfix_F) ;;
    *) echo "V2D_T3_SUBDIR 不在白名单里：$SUB"; exit 2 ;;
  esac
  SUP="${V2D_T3_SUPPORT:-}"
  case "$SUP" in ""|delivery) ;; *) echo "V2D_T3_SUPPORT 只许空或 delivery：$SUP"; exit 2 ;; esac
  OWN="${V2D_T3_OWN:-}"
  if [ -n "$OWN" ]; then
    [[ $OWN =~ ^/lab_data/$me/v2d/[A-Za-z0-9_/-]+$ && $OWN != *..* ]] || { echo "V2D_T3_OWN 要在本用户的 /lab_data/<本用户>/v2d/ 下"; exit 2; }
    [[ $SUB == tier3_kfix_* ]] || { echo "本用户目录模式只许 tier3_kfix_*：$SUB"; exit 2; }
    D="$OWN/runs/$SUB"
  else
    [[ $SUB != tier3_kfix_* ]] || { echo "$SUB 在本用户目录里，要设 V2D_T3_OWN"; exit 2; }
    D="${roots[0]}/runs/$SUB"
  fi
  echo "## 目录 runs/$SUB"
  echo "## $D 的前两层"
  find "$D" -maxdepth 2 -printf '%M %TF_%TT %s %P\n' | cut -c1-160 | sort -k4 | head -80
  for ep in "$@"; do
    e="$(printf %06d "$ep")"
    echo "## episode $ep"
    for f in "$D/data/chunk-000/episode_$e.parquet" "$D/meta/episode_$e.metadata.json"; do
      if [ -r "$f" ]; then echo "可读 $(stat -c '%s 字节 %y' "$f" | cut -c1-40) sha256 $(sha256sum "$f" | cut -d' ' -f1) ${f#$D/}"
      else echo "读不了或不存在 ${f#$D/}"; fi
    done
    [ -r "$D/meta/episode_$e.metadata.json" ] || continue
    cat "$D/meta/episode_$e.metadata.json"; echo
    if [ "$SUP" = delivery ]; then
      rel="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("support_surface") or "")' "$D/meta/episode_$e.metadata.json")"
      if [[ $rel =~ ^support_surface/[A-Za-z0-9._-]+\.usda$ ]] && [ -r "$D/$rel" ]; then
        echo "可读 $(stat -c '%s 字节 %y' "$D/$rel" | cut -c1-40) sha256 $(sha256sum "$D/$rel" | cut -d' ' -f1) $rel"
      else echo "支撑面读不了、不存在或路径不合要求：$rel"; fi
    fi
    # 重建时用的输入网格与地面（元数据里的相对路径，相对 tier3/input）的 md5，用来与官方 Tier 2 目录里的同名文件比对
    for k in mesh ground_plane; do
      rel="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$D/meta/episode_$e.metadata.json" "$k")"
      f="${roots[0]}/input/$rel"
      echo "输入 $k $( [ -r "$f" ] && md5sum "$f" | cut -d' ' -f1 || echo 读不了或不存在) $rel"
    done
  done
  if [ "$SUP" = delivery ]; then
    j="$D/support_surface/support_usda.json"
    if [ -r "$j" ]; then echo "## support_surface/support_usda.json sha256 $(sha256sum "$j" | cut -d' ' -f1)"; cat "$j"; echo
    else echo "## support_surface/support_usda.json 读不了或不存在"; fi
  fi
} 2>&1 | red
