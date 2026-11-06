# 服务器上的只读盘点：同组成员在 ml40 上的 Tier 3 重建输出目录现在有什么（顶层结构、最近 3 天动过的文件、交付目录、报告与说明、
# 正在跑的几条队列的日志尾部、各集产物数量、报告类 JSON 的顶层键与数值摘要）。经 stdin 送过去执行。
# 只读：不改文件、不进别人的容器、不读环境变量；只读文本类小文件（≤ 200 KB）。别人的目录名、用户名与 home 目录名一律换成 <其他用户>（同 tools/t3_queue_remote.sh）。
# 用法（Mac，仓库根）：scripts/dev/record ops/t3_recon_survey_ml40_<时间> ssh ml40 'bash -s' < tools/t3_recon_survey_remote.sh
#   ssh ml40 'bash -s -- <相对 tier3 的路径>…' < tools/t3_recon_survey_remote.sh：只打印这几个文本文件（各 ≤ 200 KB，取前 120 行），不做盘点。
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
if [ $# -gt 0 ]; then
  { for T in "${roots[@]}"; do for rel in "$@"; do
      case "$rel" in /*|*..*) echo "跳过不合格的路径 $rel"; continue ;; esac
      f="$T/$rel"; echo "### $rel"
      if [ -f "$f" ] && [ "$(stat -c %s "$f")" -le 204800 ]; then head -n 120 "$f" | cut -c1-300; else echo "（不在，或超过 200 KB）"; fi
  done; done; } 2>&1 | red
  echo T3_RECON_SURVEY_DONE; exit 0
fi
{
  echo "== $(hostname) $(date '+%F %T %Z')"
  [ ${#roots[@]} -gt 0 ] || echo "没有找到别的用户的 tier3 目录"
  for T in "${roots[@]}"; do
    echo "## 目录 $T：顶层（名字、大小、修改时间）"
    for x in "$T"/*; do echo "  $(basename "$x")  $(du -sh "$x" 2>/dev/null | cut -f1)  $(date -r "$x" '+%F %T')"; done
    echo "## runs/ 下的子目录（名字、条目数、修改时间）"
    for x in "$T"/runs/*/; do echo "  $(basename "$x")  $(ls "$x" 2>/dev/null | wc -l)  $(date -r "$x" '+%F %T')"; done
    echo "## 最近 3 天改过的文件（最多 80 个，按时间倒序；不含 .pkl/.npz/.h5/.mp4/.png 之外也列，只列路径）"
    find "$T" -type f -mtime -3 -printf '%TY-%Tm-%Td %TH:%TM  %s  %P\n' 2>/dev/null | sort -r | head -n 80
    echo "## 报告与说明（*.md、*README*、*report*.txt、*summary*，最近 7 天，最多 15 个，各取前 60 行）"
    mapfile -t docs < <(find "$T" -type f \( -name '*.md' -o -iname '*readme*' -o -iname '*report*.txt' -o -iname '*summary*' \) -mtime -7 -size -200k 2>/dev/null | head -n 15)
    for f in "${docs[@]}"; do echo "### ${f#$T/}"; head -n 60 "$f" | cut -c1-300; done
    echo "## 报告类 JSON（*summary*.json、*report*.json、*metrics*.json、*eval*.json，最近 7 天，最多 25 个：顶层键与数值）"
    mapfile -t js < <(find "$T" -type f \( -iname '*summary*.json' -o -iname '*report*.json' -o -iname '*metric*.json' -o -iname '*eval*.json' \) -mtime -7 -size -200k 2>/dev/null | head -n 25)
    for f in "${js[@]}"; do
      echo "### ${f#$T/}"
      python3 - "$f" <<'PY' 2>&1 | head -n 25
import json, sys
j = json.load(open(sys.argv[1]))
def flat(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from flat(v, f"{p}.{k}" if p else str(k))
    elif isinstance(x, (int, float, str, bool)) or x is None:
        yield p, x
    elif isinstance(x, list) and len(x) <= 6 and all(isinstance(v, (int, float)) for v in x):
        yield p, x
items = list(flat(j))
for k, v in items[:25]:
    print(f"  {k} = {str(v)[:120]}")
if len(items) > 25: print(f"  …共 {len(items)} 项")
PY
    done
    echo "## logs 下最近改过的 12 个日志（各取最后 25 行）"
    mapfile -t lg < <(ls -t "$T"/logs/* 2>/dev/null | head -n 12)
    for f in "${lg[@]}"; do echo "### ${f#$T/}（$(date -r "$f" '+%F %T')）"; tail -n 25 "$f" | cut -c1-300; done
    echo "## 逐集产物数量（runs/work_dump/*、runs/work_fp/* 下每集的文件数与最新修改时间）"
    for k in work_dump work_fp; do
      for x in "$T"/runs/$k/*/; do
        [ -d "$x" ] || continue
        echo "  $k/$(basename "$x")  文件 $(find "$x" -type f 2>/dev/null | wc -l)  最新 $(find "$x" -type f -printf '%TY-%Tm-%Td %TH:%TM\n' 2>/dev/null | sort | tail -n 1)"
      done
    done
  done
} 2>&1 | red
echo "T3_RECON_SURVEY_DONE"
