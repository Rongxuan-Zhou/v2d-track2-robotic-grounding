# ml40 上的只读核对：另一位用户 10/16 起用标定内参重新表达（t3_focal_all.sh convert convert_kgb intrinsics_gb.json）的进度与做法。
# 经 stdin 送过去执行，输出要经 tools/redact_stdin.sh 脱敏再落盘。不改文件、不起容器。列：intrinsics_gb.json 内容、dev/kgb 下的文件、
# t3_focal_all.sh 与 t3_convert_ep.py focal 一节的改动（git diff 最近的提交，或文件头）、convert_kgb 下每集已有的输出与日志尾。
# 用法（Mac，仓库根）：ssh -o BatchMode=yes ml40 'bash -s' < tools/t3_kgb_remote.sh | tools/redact_stdin.sh
set -uo pipefail
me="$(id -un)"
mapfile -t roots < <(ls -d /lab_data/*/v2d/tier3 2>/dev/null | grep -v "^/lab_data/$me/")
[ ${#roots[@]} = 1 ] || { echo "别的用户的 tier3 目录不是恰好一个"; exit 1; }
cd "${roots[0]}" || exit 1
echo "== $(hostname) $(date '+%F %T %Z')"
echo "## runs/floor/intrinsics_gb.json"; cat runs/floor/intrinsics_gb.json 2>/dev/null; echo
echo "## dev/kgb"; ls -la dev/kgb 2>/dev/null | cut -c1-200
echo "## code 的 git 状态与最近 5 个提交"; (cd code && git log --oneline -5 2>/dev/null; git status --short 2>/dev/null | head -10)
echo "## t3_convert_ep.py 里 focal 一节（含 intrinsics 字样的行）"; grep -n -i "intrinsic\|cx\|cy\|principal" code/stages/recovery/tier3/t3_convert_ep.py | head -40 | cut -c1-220
echo "## t3_focal_all.sh 头 30 行"; sed -n 1,30p code/stages/recovery/tier3/t3_focal_all.sh 2>/dev/null
echo "## runs/convert_kgb 每集的文件"
for d in runs/convert_kgb/ep*; do [ -d "$d" ] && echo "${d##*/}: $(ls "$d" | tr '\n' ' ' | cut -c1-200)"; done
echo "## logs/focal_*kgb* 最近 12 个的尾行"
ls -t logs/focal_*kgb* 2>/dev/null | head -12 | while read -r f; do echo "${f}：$(tail -1 "$f" | cut -c1-250)"; done
echo "T3_KGB_REMOTE_DONE"
