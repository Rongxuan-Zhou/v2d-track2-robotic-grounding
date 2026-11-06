# ml40 上的只读核对：另一位用户 10/16 凌晨跑的 CARI4D 初始化试点（t3_init_pilot.sh）留下了什么（经 stdin 送过去执行；输出要经 tools/redact_stdin.sh 脱敏再落盘）。
# 不改文件、不起容器。列：试点脚本的说明与主要步骤、日志全文、runs/work_init 的目录树（两层）、试点之后（10/16 14:00 CST 起）改过的报告类文件、
# runs 下文件名或内容里带 init 的 json/tsv/md/txt（只列路径与大小）。
# 用法（Mac，仓库根）：ssh -o BatchMode=yes ml40 'bash -s' < tools/t3_init_pilot_remote.sh | tools/redact_stdin.sh
set -uo pipefail
me="$(id -un)"
mapfile -t roots < <(ls -d /lab_data/*/v2d/tier3 2>/dev/null | grep -v "^/lab_data/$me/")
[ ${#roots[@]} = 1 ] || { echo "别的用户的 tier3 目录不是恰好一个"; exit 1; }
cd "${roots[0]}" || exit 1
echo "== $(hostname) $(date '+%F %T %Z')"
echo "## code/stages/recovery/tier3/t3_init_pilot.sh（前 80 行）"
sed -n 1,80p code/stages/recovery/tier3/t3_init_pilot.sh
echo "## logs/t3_init_pilot.log（全文，每行截到 400 字）"
cut -c1-400 logs/t3_init_pilot.log
echo "## runs/work_init 两层"
find runs/work_init -maxdepth 2 -printf '%M %TF_%TH:%TM %s %P\n' 2>/dev/null | sort -k4 | head -80
echo "## runs/work_init/<集>/ 下三层里的 json、npz、txt（路径、大小、修改时间）"
find runs/work_init -mindepth 2 -maxdepth 4 -type f \( -name '*.json' -o -name '*.npz' -o -name '*.txt' \) -printf '%TF_%TH:%TM %s %P\n' 2>/dev/null | sort -k3 | head -80
echo "## runs、logs 下 10/16 14:00 CST 之后改过的文件（三层以内）"
touch -d '2026-10-16 14:00' /tmp/.t3init_ref_$$
find runs logs -maxdepth 3 -type f -newer /tmp/.t3init_ref_$$ -printf '%TF_%TH:%TM %s %p\n' 2>/dev/null | sort | head -60
rm -f /tmp/.t3init_ref_$$
echo "## runs 下（三层以内）文件名带 init 的 json、tsv、md、txt"
find runs -maxdepth 3 -type f \( -name '*init*.json' -o -name '*init*.tsv' -o -name '*init*.md' -o -name '*init*.txt' \) -printf '%TF_%TH:%TM %s %p\n' 2>/dev/null | head -40
echo "## 代码里引用 work_init 或 init 列的脚本"
grep -rl -E "work_init|\binit\b.*(column|列)" code/stages/recovery/tier3 2>/dev/null | head -20
echo "T3_INIT_PILOT_REMOTE_DONE"
