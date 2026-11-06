# ml40 上的只读核对：另一位用户的初始化试点（runs/work_init）与它的来源（runs/work_fp）在 CARI4D 输出目录里各有什么文件，pipeline_report.json 的内容。
# 经 stdin 送过去执行，输出要经 tools/redact_stdin.sh 脱敏再落盘。不改文件、不起容器。
# 用法（Mac，仓库根）：ssh -o BatchMode=yes ml40 'bash -s -- ep13_snack_box ep28_squeegee ep04_baseball_bat ep03_contigo_coffee_mug' < tools/t3_init_outputs_remote.sh | tools/redact_stdin.sh
set -uo pipefail
me="$(id -un)"
mapfile -t roots < <(ls -d /lab_data/*/v2d/tier3 2>/dev/null | grep -v "^/lab_data/$me/")
[ ${#roots[@]} = 1 ] || { echo "别的用户的 tier3 目录不是恰好一个"; exit 1; }
cd "${roots[0]}" || exit 1
echo "== $(hostname) $(date '+%F %T %Z')"
echo "## code/stages/recovery/tier3/versions.json 里提到 init 的行"
grep -n -i "init" code/stages/recovery/tier3/versions.json | cut -c1-300
for name in "$@"; do
  [[ $name =~ ^ep[0-9]{2}_[a-z0-9_]+$ ]] || { echo "集名非法：$name"; exit 2; }
  seq=$(printf episode_%06d "$((10#${name:2:2}))")
  for w in work_fp work_init; do
    c="runs/$w/$name/cari4d/$seq"
    echo "## $w/$name：cari4d 目录（三层，文件的大小与修改时间）"
    find "$c" -maxdepth 3 -type f -printf '%TF_%TH:%TM %s %P\n' 2>/dev/null | sort -k3 | grep -v -E '\.(png|jpg)$' | head -40
    echo "## $w/$name：pipeline_report.json"
    cut -c1-3000 "$c/pipeline_report.json" 2>/dev/null; echo
  done
done
echo "T3_INIT_OUTPUTS_REMOTE_DONE"
