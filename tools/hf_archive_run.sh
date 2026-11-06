#!/usr/bin/env bash
# 归档的完整流程（训练机上、仓库根执行，setsid nohup 后台跑）：上传 → 核对 → 抽查 → 删除，任何一步不过就停。
# 计划 agent-doc/plan-hf-archive-2026-10-01.md（state，第 2 版）与 agent-doc/plan-hf-archive-policies-2026-10-08.md（policy 中间份）。
# 凭据只经环境变量进来：启动方经 stdin 读入后 export HF_TOKEN、https_proxy、HTTPS_PROXY、HF_ENDPOINT=https://huggingface.co，
# 本脚本不读 .env、不打印它们。
# V2D_HF_ARCHIVE_STEPS：all（默认，一口气做完）、verify（上传、核对、抽查后停在删除前，等用户点头）、
#   check（同 verify 但不上传：清单里的文件已经在仓库里，如第 3 批撞上私有存储额度后用 hubsubset 挑出的已提交部分）、
#   delete（删除前先重新核对一遍仓库，落到 <输出前缀>.reverified.json，再按它与前一次的 .spotcheck.json 删）。
# V2D_HF_SPOTCHECK_N：抽查下载几个文件，默认 3；V2D_HF_SPOTCHECK_PER_RUN=1 时改成每条 run 随机挑 1 个（有最终 state 的 run 固定挑它，再随机补 1 个）。
# V2D_HF_VISIBILITY：private（默认）或 public，原样传给 upload 的 --visibility，仓库的实际可见性不同就停在上传（第 3 批第 4 版：
#   用户 10/11 同意把归档仓库改为公开，第 3b 批传到公开仓库）。
# V2D_HF_SPOT_MIN_FREE_MB、V2D_HF_SPOT_MAX_CACHE_MB：抽查的两道闸（默认 0，不查），与训练链同时运行时用（第 3 批，
#   计划 agent-doc/plan-hf-archive-batch3-2026-10-10.md）：下载前可用不到 M MB 加文件大小、xet 缓存涨过 C MB，就停在抽查。
# 删除前这几种情况就停：有训练链在跑（runs/.chain-active-*）、GPU 上有进程、有导出评测训练相关的进程
# （fc_export、fc_eval、fc_run、export_chain、case_clip），它们会删 runs/flash_chord 下的临时文件，让下面的比对误报。
# 删除前后各记一份 runs/flash_chord 下全部文件的清单与磁盘余量，判据：删除期间消失的文件只能是清单里的，其他文件一个都不能少；
# 另外报出清单里没删掉的个数（被重算的保护或本地改动挡下的）。
# 用法：bash runs/tools/hf_archive_run.sh <清单.json> <用户/仓库> <机器名> <输出前缀，如 runs/hf_archive_mf4090>
# 最后一行 HF_ARCHIVE_DONE、HF_ARCHIVE_VERIFIED、HF_ARCHIVE_FILES_CHECK_FAILED 或 HF_ARCHIVE_STOPPED <哪一步>。
set -uo pipefail
M=${1:?清单}; REPO=${2:?仓库}; PREFIX=${3:?机器名}; OUT=${4:?输出前缀}
STEPS="${V2D_HF_ARCHIVE_STEPS:-all}"; N="${V2D_HF_SPOTCHECK_N:-3}"; PER_RUN="${V2D_HF_SPOTCHECK_PER_RUN:-0}"
case "$STEPS" in all|verify|check|delete) ;; *) echo "V2D_HF_ARCHIVE_STEPS 只能是 all、verify、check、delete" >&2; exit 2 ;; esac
[[ $N =~ ^[1-9][0-9]*$ ]] || { echo "V2D_HF_SPOTCHECK_N 必须是正整数" >&2; exit 2; }
case "$PER_RUN" in 0) SPOT=(--n "$N"); SPOT_DESC="${N} 个" ;; 1) SPOT=(--per-run); SPOT_DESC="每条 run 1 个" ;;
  *) echo "V2D_HF_SPOTCHECK_PER_RUN 只能是 0 或 1" >&2; exit 2 ;; esac
VIS="${V2D_HF_VISIBILITY:-private}"
case "$VIS" in private|public) ;; *) echo "V2D_HF_VISIBILITY 只能是 private 或 public" >&2; exit 2 ;; esac
GMIN="${V2D_HF_SPOT_MIN_FREE_MB:-0}"; GCACHE="${V2D_HF_SPOT_MAX_CACHE_MB:-0}"
[[ $GMIN =~ ^[0-9]+$ && $GCACHE =~ ^[0-9]+$ ]] || { echo "V2D_HF_SPOT_MIN_FREE_MB、V2D_HF_SPOT_MAX_CACHE_MB 必须是非负整数" >&2; exit 2; }
SPOT+=(--min-free-mb "$GMIN" --max-cache-growth-mb "$GCACHE")
T=runs/tools/hf_archive.py
trap 'rm -rf runs/hf_spotcheck' EXIT   # 抽查被 timeout 或别的信号打断时，下载到一半的文件不留在盘上（第 3 批审查 m3）
ts() { date '+%F %T %Z'; }
stop() { echo "=== $(ts) HF_ARCHIVE_STOPPED $1"; exit 1; }
free_mb() { df -Pk . | awk 'NR == 2 {print int($4 / 1024)}'; }
xet_mb() { du -sm "${HF_XET_CACHE:-${HF_HOME:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}/xet}" 2>/dev/null | cut -f1; }
all_files() { find runs/flash_chord -type f -printf '%p\n' | sort; }
echo "=== $(ts) 开始（步骤 ${STEPS}，抽查 ${SPOT_DESC}，可见性 ${VIS}）；工具 md5 $(md5sum "$T" | cut -d' ' -f1)；清单 $M（$(python3 -c 'import json, sys; print(len(json.load(open(sys.argv[1]))))' "$M") 个文件，md5 $(md5sum "$M" | cut -d' ' -f1)）；可用 $(free_mb) MB；xet 缓存 $(xet_mb) MB"
if [ "$STEPS" != delete ]; then
  [ "$STEPS" = check ] || python3 "$T" upload --manifest "$M" --repo "$REPO" --prefix "$PREFIX" --visibility "$VIS" || stop upload
  python3 "$T" verify --manifest "$M" --repo "$REPO" --prefix "$PREFIX" --out "$OUT.verified.json" || stop verify
  python3 "$T" spotcheck --verified "$OUT.verified.json" "${SPOT[@]}" --out "$OUT.spotcheck.json" || stop spotcheck
fi
if [ "$STEPS" = verify ] || [ "$STEPS" = check ]; then
  echo "=== $(ts) $([ "$STEPS" = check ] && echo 核对、抽查都通过（没有上传） || echo 上传、核对、抽查都通过)，按约定停在删除前，等用户确认；可用 $(free_mb) MB；xet 缓存 $(xet_mb) MB；HF_ARCHIVE_VERIFIED"
  exit 0
fi
[ -s "$OUT.verified.json" ] && [ -s "$OUT.spotcheck.json" ] || stop "delete（缺核对或抽查结果）"
if [ "$STEPS" = delete ]; then
  python3 "$T" verify --manifest "$M" --repo "$REPO" --prefix "$PREFIX" --out "$OUT.reverified.json" || stop reverify
  V="$OUT.reverified.json"
else
  V="$OUT.verified.json"
fi
ls runs/.chain-active-* >/dev/null 2>&1 && stop "delete（有训练链在跑）"
gpu="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>&1)" || stop "delete（nvidia-smi 查不了 GPU）"
[ -z "$gpu" ] || stop "delete（GPU 上有进程：$(tr '\n' ' ' <<< "$gpu")）"
busy="$(pgrep -af '[f]c_export|[f]c_eval|[f]c_run|[e]xport_chain|[c]ase_clip' || true)"
[ -z "$busy" ] || stop "delete（有导出、评测或训练相关的进程：$(tr '\n' ' ' <<< "$busy" | cut -c1-300)）"
all_files > "$OUT.files_before.txt"
python3 "$T" delete --verified "$V" --spotcheck "$OUT.spotcheck.json" --yes || stop delete
all_files > "$OUT.files_after.txt"
python3 -c 'import json, sys
for r in json.load(open(sys.argv[1])):
    print("runs/flash_chord/%s/%s" % (r["run"], r["file"]))' "$V" | sort > "$OUT.allowed.txt"
comm -23 "$OUT.files_before.txt" "$OUT.files_after.txt" > "$OUT.gone.txt"
odd="$(comm -23 "$OUT.gone.txt" "$OUT.allowed.txt")"
kept="$(comm -13 "$OUT.gone.txt" "$OUT.allowed.txt" | grep -c . || true)"
echo "=== $(ts) runs/flash_chord：删除前 $(wc -l < "$OUT.files_before.txt") 个文件、删除后 $(wc -l < "$OUT.files_after.txt") 个；消失 $(wc -l < "$OUT.gone.txt") 个（清单 $(wc -l < "$OUT.allowed.txt") 个，其中没删的 ${kept} 个）；清单外消失的文件：${odd:-无}"
echo "=== $(ts) 删除后可用 $(free_mb) MB"
if [ -n "$odd" ]; then echo "=== $(ts) HF_ARCHIVE_FILES_CHECK_FAILED"; exit 1; fi
echo "=== $(ts) HF_ARCHIVE_DONE"
