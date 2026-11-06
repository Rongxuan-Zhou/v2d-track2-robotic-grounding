#!/usr/bin/env bash
# “v0.4.0 训出的 checkpoint 能否在 8c778f9 评测器里加载、导出、打包”的静态事实（Mac 上执行，只读：本地新 kit 加 gh api 读上游源码）。
# 计划 agent-doc/plan-box6-and-eval-runtime-2026-10-01.md 第 2 节；每行带版本、文件与行号，供 jev 核对。
#   1. v0.4.0 → 8c778f9：提交、flash_chord 下改了哪些文件；训练路径上唯一改到的 training/flash_sac/evaluation.py 的原文 diff；依赖文件有没有动
#   2. 新 kit 自带的 evaluate_policy.py 与三份 Hydra 配置，和 8c778f9 的同名文件是否逐字节相同；8c778f9 评测代码的两个摘要
#   3. 评测器怎样从 checkpoint 重建实验（harness.py 的加载行，8c778f9）
#   4. kit 的导出与打包规矩（examples/run_policy_evaluation.py、tools/pack_submission.py、config/tracks.json、data/ 样例、Track 2 说明）
# 用法（仓库根）：scripts/dev/record submission/eval_runtime_facts stages/evaluate/eval_runtime_facts.sh
set -uo pipefail
K=third_party/v2d_submission_kit/update_0f2f9f4b/v2d_submission_kit
REPO=nvidia-isaac/video_to_data; B=robotic_grounding/flash_chord
OLD=33129dd; NEW=8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef
src() { gh api "repos/$REPO/contents/$B/$2?ref=$1" --jq .content | base64 -d; }

echo "== 1. v0.4.0（${OLD}）→ 8c778f9 的提交与改动"
gh api "repos/$REPO/compare/$OLD...$NEW" --jq '"  ahead " + (.ahead_by|tostring) + "，behind " + (.behind_by|tostring) + "，改动文件 " + (.files|length|tostring) + " 个"'
gh api "repos/$REPO/compare/$OLD...$NEW" --jq '.commits[] | "  提交 " + .sha[0:10] + " " + .commit.committer.date + " " + (.commit.message|split("\n")[0])'
echo "flash_chord 下改动的文件（状态 +增 -删）："
gh api "repos/$REPO/compare/$OLD...$NEW" --jq ".files[] | select(.filename|startswith(\"$B/\")) | \"  \" + .status + \" +\" + (.additions|tostring) + \" -\" + (.deletions|tostring) + \" \" + .filename"
echo "flash_chord 以外改动的文件："
gh api "repos/$REPO/compare/$OLD...$NEW" --jq ".files[] | select(.filename|startswith(\"$B/\")|not) | \"  \" + .status + \" \" + .filename"
n_dep="$(gh api "repos/$REPO/compare/$OLD...$NEW" --jq '[.files[] | select(.filename|test("pyproject.toml$|uv.lock$|requirements[^/]*.txt$|Dockerfile|environment.ya?ml$"))] | length')"
echo "依赖与镜像文件（pyproject.toml、uv.lock、requirements*.txt、Dockerfile、environment.yml）改动数：$n_dep"
echo "flash_chord/src 下、评测目录（evaluation/、configs/evaluation/）以外改动的文件："
gh api "repos/$REPO/compare/$OLD...$NEW" --jq ".files[] | select(.filename|startswith(\"$B/src/\")) | select(.filename|test(\"/evaluation/\")|not) | \"  \" + .filename"
echo "其中 training/flash_sac/evaluation.py 的全部改动（原文）："
gh api "repos/$REPO/compare/$OLD...$NEW" --jq ".files[] | select(.filename==\"$B/src/flash_chord/training/flash_sac/evaluation.py\") | .patch" | sed 's/^/  | /'

echo "== 2. 新 kit（zip sha256 $(shasum -a 256 "$K/../v2d_submission_kit.zip" | cut -c1-16)…）自带的评测入口与配置，对比 8c778f9 的同名文件"
for f in scripts/evaluate_policy.py src/flash_chord/configs/evaluate_policy.yaml src/flash_chord/configs/evaluation/flash_sac.yaml src/flash_chord/configs/logging/wandb.yaml; do
  a="$(shasum -a 256 "$K/flash_chord/$f" | cut -d' ' -f1)"; b="$(src "$NEW" "$f" | shasum -a 256 | cut -d' ' -f1)"
  [ "$a" = "$b" ] && echo "  相同 ${f}（sha256 ${a:0:16}…）" || echo "  不同 ${f}：kit ${a:0:16}…，8c778f9 ${b:0:16}…"
done
echo "kit 里 flash_chord 目录的全部文件："; (cd "$K" && find flash_chord -type f | sort | sed 's/^/  /')
echo "8c778f9 评测代码的摘要（tools/make_fc_patch.sh 按 harness.evaluation_code_sha256 的算法算，存 third_party/patches/flash_chord-8c778f9.eval.json）："
sed 's/^/  /' third_party/patches/flash_chord-8c778f9.eval.json

echo "== 3. 评测器从 checkpoint 重建实验（harness.py @ 8c778f9）"
src "$NEW" src/flash_chord/evaluation/harness.py | grep -n -E 'def evaluate_checkpoint|use_checkpoint_config|load_checkpoint|checkpoint_config|read_checkpoint|safetensors|def _load|restore' | sed 's/^/  harness.py:/'
grep -n -E 'use_checkpoint_config|world_count|reset_mode' "$K/flash_chord/src/flash_chord/configs/evaluation/flash_sac.yaml" | sed 's/^/  kit configs\/evaluation\/flash_sac.yaml:/'

echo "== 4. kit 的导出与打包规矩"
grep -n -E 'world-count", type=int, default|evaluation.reset_mode|evaluation.object_trajectories_output|logging.mode=disabled|cwd=workdir' "$K/examples/run_policy_evaluation.py" | sed 's/^/  examples\/run_policy_evaluation.py:/'
grep -n -E '_TRACK2_POSITION_TOLERANCE_M =|_TRACK2_ANGLE_TOLERANCE_RAD =|first_evaluator = evaluator_reference\[0\]|do not differ by one rigid transform|scored episodes have no trajectory|environment_valid=false|One submission|--frame-references|locally computed|Local scores|evaluator metrics use the policy input|def crosscheck|do not line up with the reference' "$K/tools/pack_submission.py" | sed 's/^/  tools\/pack_submission.py:/'
python3 - "$K/config/tracks.json" <<'PY'
import json, sys
t = json.load(open(sys.argv[1]))["tracks"]
for k in ("track_2_tier1", "track_2_tier2", "track_2_tier3"):
    s = t.get(k, {})
    print(f"  config/tracks.json {k}: tier {s.get('tier')}，data_root {s.get('data_root')}，scored_episodes {s.get('scored_episodes')}，"
          f"worlds_per_episode {s.get('worlds_per_episode')}，private_fraction {s.get('private_fraction')}")
PY
python3 - "$K/data/track_2_sample_submission.csv" <<'PY'
import csv, re, sys
rows = list(csv.reader(open(sys.argv[1])))[1:]
keys = [re.match(r"^track_2/e(\d{6})/w(\d{3})/f(\d{6})/o(\d+)$", r[0]).groups() for r in rows]
print(f"  data/track_2_sample_submission.csv：{len(rows)} 行，episode {len({k[0] for k in keys})} 条，世界编号 {sorted({int(k[1]) for k in keys})}")
PY
grep -n -E 'frame-references|released_track2_world|one rigid transform|Use the same CSV|Local scores use the clean reference' "$K/tracks/track_2/README.md" | sed 's/^/  tracks\/track_2\/README.md:/'
grep -n -E '^numpy|^pandas|^pyarrow|^trimesh|Python 3' "$K/requirements.txt" | sed 's/^/  requirements.txt:/'
