# 服务器上的只读核对：一组序列在这台机器上的训练输入是否齐全、是否与某条 run 训练时用的同一份。经 stdin 送过去执行。
# 每条序列一行（TSV）：序列、参考 parquet 的 sha256、runs/motion_src 下 v2d_meta.json 的档位与 episode、
# 支撑面文件 <序列>_support.usda 的 md5（没有记 -）、有没有旧版的 <序列>_g1_support.usda（有就会被优先加载）、
# 给定 run 的 run_record.jsonl 第一条 start 记录里的 parquet_sha256（run 不在这台记 -），以及两者是否相同。
# 只读：不改文件、不读别的档位。仓库目录按这台机器的布局找（ml40 在 /lab_data，其余在 home）。
# 用法（Mac，仓库根）：scripts/dev/record ops/<名字> ssh <主机> 'bash -s -- <档位> <序列>=<run> ...' < tools/motion_inputs_remote.sh
#   run 写 - 表示只看输入、不比对。
set -uo pipefail
T=${1:?档位}; shift
for R in /lab_data/rxzhou/v2d/maniphys_lab_v2d_challenge "$HOME/v2d/maniphys_lab_v2d_challenge"; do [ -d "$R" ] && break; done
cd "$R" || exit 2
echo "== $(hostname) $(date '+%F %T %Z') 仓库 $R 来历 $(cat .v2d-source-commit 2>/dev/null)"
printf 'seq\tparquet_sha256\tv2d_meta\tsupport_md5\tg1_support\trun\trun_parquet_sha256\tsame\n'
n=0; bad=0
for kv in "$@"; do
  s="${kv%%=*}"; run="${kv#*=}"
  p="runs/motion/tier$T/whole_body/soma/sequence_id=$s/robot_name=g1/data.parquet"
  sha="$( [ -f "$p" ] && sha256sum "$p" | cut -d' ' -f1 || echo MISSING)"
  m="runs/motion_src/tier$T/$s/v2d_meta.json"
  meta="$( [ -f "$m" ] && python3 -c 'import json,sys; j=json.load(open(sys.argv[1])); print("tier=%s,ep=%s" % (j["tier"], j["episode"]))' "$m" || echo MISSING)"
  st="runs/motion/tier$T/whole_body/reconstructed_stage"
  sup="$( [ -f "$st/${s}_support.usda" ] && md5sum "$st/${s}_support.usda" | cut -d' ' -f1 || echo -)"
  g1="$( [ -e "$st/${s}_g1_support.usda" ] && echo YES || echo no)"
  rs=-
  if [ "$run" != - ] && [ -f "runs/flash_chord/$run/run_record.jsonl" ]; then
    rs="$(python3 -c 'import json,sys
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r.get("event") == "start":
        print(r.get("parquet_sha256", "?")); break' "runs/flash_chord/$run/run_record.jsonl")"
  fi
  same=-; [ "$rs" != - ] && { [ "$rs" = "$sha" ] && same=yes || same=NO; }
  { [ "$sha" = MISSING ] || [ "$meta" = MISSING ] || [ "$g1" = YES ] || [ "$same" = NO ]; } && bad=$((bad + 1))
  n=$((n + 1))
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$s" "$sha" "$meta" "$sup" "$g1" "$run" "$rs" "$same"
done
echo "MOTION_INPUTS 序列 $n 条，有问题 $bad 条（缺 parquet 或 v2d_meta、有旧版 g1 支撑面、或与 run 记录的 sha256 不同）"
