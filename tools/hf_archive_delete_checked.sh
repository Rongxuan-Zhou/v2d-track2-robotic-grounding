#!/usr/bin/env bash
# HF 归档的删除一步，连同删除前的对齐核对与演习，一条命令按登记的判据做完（训练机上执行，放在 runs/tools/ 下跑；
# 凭据只在环境里，经 tools/hf_remote.sh 注入，本脚本不读 .env、不打印它们；输出照前几批滤掉含 @ 或 token 样式的行）。
# 判据出自 agent-doc/plan-hf-archive-batch4-2026-10-12.md 第 2 节第 6 步与第 3 节“删除前”“删除”两条；
# 串成一条命令、由 stages/policy/arm_after.sh 在链结束后调用，见 agent-doc/plan-mf4090-handover-2026-10-13.md（第 2 版）。
# 任何一步不过就停，后面的步骤都不跑：
#   1. 对齐核对：hf_archive.py verify 按清单把仓库与本地重新核对一遍（写 <输出前缀>.precheck.json，输出落 .precheck.log），退出码 0 且有 HF_VERIFY_OK；
#   2. 演习：hf_archive.py delete 不加 --yes（按现在的本地文件逐个重算大小与 sha256，输出落 .dryrun.log），退出码 0，
#      “将删”的行数等于清单的文件数，“跳过”的行 0 行；
#   3. 删除：V2D_HF_ARCHIVE_STEPS=delete 跑 hf_archive_run.sh（它再核对一遍仓库、过三道闸、逐个重算 sha256、比对删除前后的文件清单），
#      输出边跑边落 runs/logs/<输出前缀的文件名>_delete.log（中途断了也有记录）。要：退出码 0、最后一行 HF_ARCHIVE_DONE、
#      “其中没删的 0 个”且“清单外消失的文件：无”、可用空间的增加与清单总大小（MiB）相差不超过 50；
#   4. 删除后的本地状态：hf_archive_report.py（只报，记下退出码）。
# 最后一行 HF_DELETE_CHECKED_DONE，或 HF_DELETE_CHECKED_STOPPED <哪一步>。删除日志已存在就不做（同一批只删一次）。
# 用法：bash runs/tools/hf_archive_delete_checked.sh <清单.json> <用户/仓库> <机器名> <输出前缀，如 runs/hf_archive_batch4_mf4090>
#   V2D_HF_VISIBILITY 照传给 hf_archive_run.sh（第 4 批是 public）。
# 测试：tools/test_hf_archive_delete_checked.py。
set -uo pipefail
M=${1:?用法见文件头}; REPO=${2:?}; PREFIX=${3:?}; OUT=${4:?}
cd "$(dirname "$0")/../.."
T=runs/tools; DL="runs/logs/$(basename "$OUT")_delete.log"
ts() { date '+%F %T %Z'; }
stop() { echo "=== $(ts) HF_DELETE_CHECKED_STOPPED $1"; exit 1; }
filt() { grep --line-buffered -v -E '@|hf_[A-Za-z0-9]{20}'; }
[ -f "$M" ] || stop "参数（缺清单 ${M}）"
[ -s "$OUT.verified.json" ] && [ -s "$OUT.spotcheck.json" ] || stop "参数（缺第一段的核对或抽查结果 $OUT.verified.json、$OUT.spotcheck.json）"
[ ! -e "$DL" ] || stop "参数（$DL 已存在：同一批只删一次）"
mkdir -p runs/logs
read -r n want < <(python3 -c 'import json, sys; m = json.load(open(sys.argv[1])); print(len(m), round(sum(r["size"] for r in m) / 1048576))' "$M")
[[ ${n:-} =~ ^[1-9][0-9]*$ && ${want:-} =~ ^[0-9]+$ ]] || stop "参数（清单读不懂）"
echo "=== $(ts) 删除前的对齐核对：清单 ${M}（${n} 个文件、${want} MiB，md5 $(md5sum "$M" | cut -d' ' -f1)），hf_archive.py md5 $(md5sum "$T/hf_archive.py" | cut -d' ' -f1)"

python3 "$T/hf_archive.py" verify --manifest "$M" --repo "$REPO" --prefix "$PREFIX" --out "$OUT.precheck.json" 2>&1 | filt | tee "$OUT.precheck.log"; rc=${PIPESTATUS[0]}
[ "$rc" = 0 ] && grep -q HF_VERIFY_OK "$OUT.precheck.log" || stop "对齐核对（退出码 ${rc}）"

python3 "$T/hf_archive.py" delete --verified "$OUT.precheck.json" --spotcheck "$OUT.spotcheck.json" 2>&1 | filt | tee "$OUT.dryrun.log"; rc=${PIPESTATUS[0]}
will="$(grep -c '^将删 ' "$OUT.dryrun.log")"; skip="$(grep -c '^跳过 ' "$OUT.dryrun.log")"
echo "=== $(ts) 演习：将删 ${will} 个（清单 ${n} 个），跳过 ${skip} 个，退出码 ${rc}"
[ "$rc" = 0 ] && [ "$will" = "$n" ] && [ "$skip" = 0 ] || stop "演习（将删 ${will} / ${n}，跳过 ${skip}，退出码 ${rc}）"

V2D_HF_ARCHIVE_STEPS=delete bash "$T/hf_archive_run.sh" "$M" "$REPO" "$PREFIX" "$OUT" 2>&1 | filt | tee "$DL"; rc=${PIPESTATUS[0]}
[ "$rc" = 0 ] && tail -1 "$DL" | grep -q HF_ARCHIVE_DONE || stop "删除（退出码 ${rc}）"
grep -qF '其中没删的 0 个）；清单外消失的文件：无' "$DL" || stop "删除（有清单里没删的文件，或清单外的文件消失）"
b="$(sed -n 's/.*；可用 \([0-9][0-9]*\) MB；xet 缓存.*/\1/p' "$DL" | head -1)"; a="$(sed -n 's/.*删除后可用 \([0-9][0-9]*\) MB.*/\1/p' "$DL" | tail -1)"
[[ $b =~ ^[0-9]+$ && $a =~ ^[0-9]+$ ]] || stop "删除（读不出删除前后的可用空间）"
d=$((a - b - want))
echo "=== $(ts) 可用空间：删除前 ${b} MB、删除后 ${a} MB，增加 $((a - b)) MB；清单 ${want} MiB；相差 ${d}"
[ "${d#-}" -le 50 ] || stop "删除（可用空间的增加与清单相差 ${d} MB，超过 50）"

python3 "$T/hf_archive_report.py" "$M" "$OUT" 2>&1 | filt; echo "=== $(ts) 删除后报告的退出码 ${PIPESTATUS[0]}"
echo "=== $(ts) HF_DELETE_CHECKED_DONE"
