#!/usr/bin/env bash
# 用官方打包器把一档的导出打成提交 CSV，再用官方上传器的 --dry-run 只做校验与只读预检（Mac 上、仓库根执行）。不上传。
# 计划 agent-doc/plan-submission-exports-2026-10-02.md 第 3.5 节。
# 用法：stages/evaluate/pack_tier.sh <档位 1|2|3> <结果目录> <名字> [episode …]
#   Tier 3（计划 agent-doc/plan-r3o-2026-10-15.md 2.4、2.4.1）：必须用 V2D_FRAME_REFS=<目录> 给 --frame-references（kit 的 tracks/track_2/README.md：
#   用自己的重建时，要把策略输入表达在发布的 Track 2 世界系里）。目录由 stages/evaluate/make_frame_refs_t3.py 在训练机上生成、
#   pull_submission.sh frefs 拉回，里面要有 manifest_*.tsv；打包前 stages/evaluate/frefs_t3_match.py 逐集核：manifest 恰好一行、run 与步数、
#   档位目录与序列都和导出 JSON 里的相同、npz 的 sha256 对，不过就拒绝。别的档位设了 V2D_FRAME_REFS 就拒绝。
#   打包器按干净参考算的 “locally computed” 那一行同 Tier 2 滤掉（Tier 3 的选择只用 Tier 3 的数据）。
#   结果目录：pull_submission.sh 拉下来的 runs/submission/final/t<档位> 或 runs/submission/compat/t<档位>（里面只放这一档的 episode_XXXXXX.{json,parquet}）。
#   给了 episode：复制一份 kit 的 config/tracks.json，把这一档的 scored_episodes 改成这几条，用 --config 传给打包器（只用于预演）；
#     预演不跑上传器预检（行数与官方样例对不上，必然不过）。
#   V2D_PACK_NEGCTL=1（只许 Tier 2 预演用）：反向对照，不给 --frame-references，打包器改用发布的 Tier 2 带噪输入换算，
#     期望报 “do not differ by one rigid transform”；报了就 NEGCTL_OK、退出码 0，没报（或打包成功）就 NEGCTL_FAIL、退出码 1。
# 打包：uv 起 Python 3.12，装 kit 的 requirements.txt（另限 pandas<3，与 kit 写时的 pandas 2 一致），跑 kit 的 tools/pack_submission.py：
#   --track track_2_tier<档位> --results <结果目录> --dataset-root data/challenge
#   --code-commit https://github.com/Manifold-ai/maniphys_lab_v2d_challenge/commit/<HEAD>（HEAD 已推到 origin、除 results/ 外工作树干净，否则拒绝）；
#   Tier 2 另加 --frame-references runs/submission/frame_refs_t2（v5.4，make_frame_references.py 生成）。
#   Tier 2：打包器按干净参考算的 “locally computed” 那一行在落盘、显示之前滤掉（计划第 5 节：Tier 2/3 的本地分数不拿来做任何选择，属解读，待用户确认）。
# 预检：kit 的 submit/v2d_submit.py <csv> --track track_2_tier<档位> -m … --dry-run：本地校验 CSV 的行与官方样例一致，只读查 Kaggle 的报名、身份与配额，
#   然后在任何上传之前返回（v2d_submit.py 517–519 行）。本脚本没有不带 --dry-run 的调用。
# kit 用 third_party/v2d_submission_kit/update_0f2f9f4b（10/1 13:43 重发的那版），先核 zip 的 sha256。
# 输出 runs/submission/csv/track2_tier<档位>_<名字>.csv（已存在就拒绝）；打包与预检的输出经 scripts/dev/record 落盘到 results/submission/pack/t<档位>_<名字>{,_dryrun}.txt。
set -uo pipefail
cd "$(dirname "$0")/../.."
# V2D_KIT_DIR、V2D_KIT_ZIP_SHA、V2D_PACK_PYTHON 只给测试换成假的 kit 与本机 python
KIT=${V2D_KIT_DIR:-third_party/v2d_submission_kit/update_0f2f9f4b/v2d_submission_kit}
KIT_ZIP_SHA=${V2D_KIT_ZIP_SHA:-0f2f9f4bb01082a48ab684750cbcbddd985c2de7031847928ae124e53fa605d7}
PY=(uv run --no-project --python 3.12 --with-requirements "$KIT/requirements.txt" --with "pandas<3" python)
[ -n "${V2D_PACK_PYTHON:-}" ] && PY=("$V2D_PACK_PYTHON")

if [ "${1:-}" = --inner ]; then   # 由 record 调用：真正跑打包器，Tier 2 滤掉本地分数那一行
  shift; T=$1; RES=$2; CSV=$3; URL=$4; CFG=$5; NEG=${6:-0}; FREF=${7:-}
  args=(--track "track_2_tier$T" --results "$RES" --dataset-root data/challenge --code-commit "$URL" -o "$CSV")
  [ "$T" = 2 ] && [ "$NEG" != 1 ] && [ -z "$FREF" ] && FREF=runs/submission/frame_refs_t2
  [ -n "$FREF" ] && args+=(--frame-references "$FREF")
  [ -n "$CFG" ] && args+=(--config "$CFG")
  "${PY[@]}" -c 'import importlib.metadata as m, sys
def v(p):
    try:
        return m.version(p)
    except m.PackageNotFoundError:
        return "无"
print("# python", sys.version.split()[0], " ".join(f"{p}={v(p)}" for p in ("numpy", "pandas", "pyarrow", "trimesh", "kaggle")))'
  if [ "$T" = 2 ] || [ "$T" = 3 ]; then
    # -u：不缓冲，stdout 与 stderr 不会把那一行切开；grep 不锚定行首，混进了别的输出也照样滤掉整行
    "${PY[@]}" -u "$KIT/tools/pack_submission.py" "${args[@]}" 2>&1 | { grep -v 'locally computed' || true; }
    rc=${PIPESTATUS[0]}
  else
    "${PY[@]}" -u "$KIT/tools/pack_submission.py" "${args[@]}" 2>&1; rc=$?
  fi
  [ "$rc" = 0 ] || exit "$rc"
  [ -s "$CSV" ] || { echo "pack_tier: 打包器退出码 0 但没有 $CSV"; exit 1; }
  echo "CSV ${CSV}：$(($(wc -l < "$CSV") - 1)) 行，sha256 $(shasum -a 256 "$CSV" | cut -d' ' -f1)"
  exit 0
fi

T=${1:?用法见文件头}; RES=${2:?用法见文件头}; NAME=${3:?用法见文件头}; shift 3; EPS=("$@"); NEG=${V2D_PACK_NEGCTL:-0}
[ "$NEG" != 1 ] || { [ "$T" = 2 ] && [ $# -gt 0 ]; } || { echo "pack_tier: V2D_PACK_NEGCTL=1 只许 Tier 2 预演用" >&2; exit 2; }
[[ $T =~ ^[123]$ ]] || { echo "pack_tier: 档位只能是 1、2 或 3" >&2; exit 2; }
[[ $NAME =~ ^[A-Za-z0-9_-]+$ ]] || { echo "pack_tier: 名字只能用字母、数字、_、-" >&2; exit 2; }
[ -d "$RES" ] || { echo "pack_tier: 没有结果目录 $RES" >&2; exit 2; }
bad="$(find "$RES" -name '*.parquet' ! -name 'episode_[0-9][0-9][0-9][0-9][0-9][0-9].parquet' | head -3)"
[ -z "$bad" ] || { echo "pack_tier: 结果目录里有不是 episode_XXXXXX.parquet 的 parquet：$bad" >&2; exit 2; }
[ "$(shasum -a 256 "$KIT/../v2d_submission_kit.zip" | cut -d' ' -f1)" = "$KIT_ZIP_SHA" ] || { echo "pack_tier: kit 的 zip 不是 0f2f9f4b 那版" >&2; exit 2; }
for need in data/challenge/track_2/tier_1_multiview_caption/meta $([ "$T" = 2 ] && echo runs/submission/frame_refs_t2/manifest.tsv); do
  [ -e "$need" ] || { echo "pack_tier: 缺 ${need}（先跑 stages/evaluate/pull_submission.sh data / refs）" >&2; exit 2; }
done
FREF=${V2D_FRAME_REFS:-}; MATCH=""
if [ "$T" = 3 ]; then
  [ -n "$FREF" ] || { echo "pack_tier: Tier 3 要用 V2D_FRAME_REFS=<目录> 给帧参考（make_frame_refs_t3.py 生成、pull_submission.sh frefs 拉回；kit 的 tracks/track_2/README.md：用自己的重建要给 --frame-references）" >&2; exit 2; }
  compgen -G "$FREF/manifest_*.tsv" > /dev/null || { echo "pack_tier: 帧参考目录 $FREF 里没有 manifest_*.tsv" >&2; exit 2; }
  # 逐集核：manifest 恰好一行、run 与步数和导出的 checkpoint 相同、档位目录与序列和导出的 motion parquet 相同、npz 的 sha256 对（Fable 5.1 的审查）
  MATCH="$(python3 stages/evaluate/frefs_t3_match.py "$RES" "$FREF" 2>&1)" \
    || { echo "$MATCH" >&2; echo "pack_tier: 帧参考与结果目录逐集对不上（stages/evaluate/frefs_t3_match.py）" >&2; exit 2; }
else
  [ -z "$FREF" ] || { echo "pack_tier: V2D_FRAME_REFS 只给 Tier 3 用（Tier 2 固定用 runs/submission/frame_refs_t2）" >&2; exit 2; }
fi
CSV="runs/submission/csv/track2_tier${T}_${NAME}.csv"
[ ! -e "$CSV" ] || { echo "pack_tier: $CSV 已存在，拒绝覆盖" >&2; exit 2; }
[ -z "$(git status --porcelain -- . ':(exclude)results')" ] || { echo "pack_tier: 工作树有未提交的改动（results/ 除外），--code-commit 指不回这次的代码" >&2; exit 2; }
HEAD="$(git rev-parse HEAD)"
[ -n "$(git branch -r --contains "$HEAD")" ] || { echo "pack_tier: HEAD $HEAD 还没推到 origin" >&2; exit 2; }
URL="https://github.com/Manifold-ai/maniphys_lab_v2d_challenge/commit/$HEAD"
mkdir -p runs/submission/csv
CFG=""
if [ ${#EPS[@]} -gt 0 ]; then
  CFG="runs/submission/csv/tracks_tier${T}_${NAME}.json"
  python3 - "$KIT/config/tracks.json" "$CFG" "track_2_tier$T" "${EPS[@]}" <<'PY' || exit 2
import json, sys
src, dst, track, *eps = sys.argv[1:]
c = json.load(open(src))
c["tracks"][track]["scored_episodes"] = [int(e) for e in eps]
json.dump(c, open(dst, "x"), indent=2)
print(f"预演：{track} 只计 episode {[int(e) for e in eps]}（{dst}）")
PY
fi
echo "pack_tier: Tier ${T}，结果目录 ${RES}，代码 $URL"
[ -z "$FREF" ] || echo "pack_tier: 帧参考 ${FREF}：$(ls "$FREF"/episode_*.npz | wc -l | tr -d ' ') 个 npz，$(cd "$FREF" && for m in manifest_*.tsv; do printf '%s sha256 %s  ' "$m" "$(shasum -a 256 "$m" | cut -c1-16)"; done)逐集核对 $(tail -1 <<< "$MATCH")"
if [ "$NEG" = 1 ]; then
  if scripts/dev/record "submission/pack/t${T}_${NAME}" bash stages/evaluate/pack_tier.sh --inner "$T" "$RES" "$CSV" "$URL" "$CFG" 1; then
    echo "NEGCTL_FAIL（不给帧参考也打包成功了）"; exit 1
  fi
  grep -q "do not differ by one rigid transform" "results/submission/pack/t${T}_${NAME}.txt.failed" \
    && { echo "NEGCTL_OK（不给帧参考时打包器报不是一个刚体变换）"; exit 0; }
  echo "NEGCTL_FAIL（打包失败，但不是因为刚体变换，见 results/submission/pack/t${T}_${NAME}.txt.failed）"; exit 1
fi
scripts/dev/record "submission/pack/t${T}_${NAME}" bash stages/evaluate/pack_tier.sh --inner "$T" "$RES" "$CSV" "$URL" "$CFG" 0 "$FREF" || exit 1
tail -4 "results/submission/pack/t${T}_${NAME}.txt"
if [ -n "$CFG" ]; then echo "PACK_OK（预演，不跑上传器预检）"; exit 0; fi
scripts/dev/record "submission/pack/t${T}_${NAME}_dryrun" "${PY[@]}" "$KIT/submit/v2d_submit.py" "$CSV" --track "track_2_tier$T" \
  -m "dry run, not submitted" --dry-run || { echo "PACK_OK_DRYRUN_FAIL（CSV 打好了，上传器预检不过，见 results/submission/pack/t${T}_${NAME}_dryrun.txt.failed）"; exit 1; }
tail -8 "results/submission/pack/t${T}_${NAME}_dryrun.txt"
echo "PACK_OK"
