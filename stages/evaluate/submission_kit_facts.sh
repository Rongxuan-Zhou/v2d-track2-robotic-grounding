#!/usr/bin/env bash
# 官方 submission kit（2026-09-28 邮件发布）里与 Track 2 评测口径有关的原文摘录（Mac 上执行，只读；上游补丁经 gh api 读取）。
# 每行带“文件:行号”，供 jev 核对：评测按 checkpoint 自己的配置重建、每条 episode 几个世界、导出轨迹的帧号怎么编、
# 打包器怎么对齐参考帧、样例 CSV 的格子、Tier 的计分设置、官方接受的评测代码哈希。
# 用法（仓库根）：scripts/dev/record submission/kit_facts stages/evaluate/submission_kit_facts.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
Z=third_party/v2d_submission_kit/v2d_submission_kit.zip
K=third_party/v2d_submission_kit/v2d_submission_kit
URL=https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_submission_kit.zip
[ -f "$Z" ] && [ -d "$K" ] || { echo "先下载并解压 $URL 到 third_party/v2d_submission_kit/"; exit 1; }
g() { grep -nE "$1" "$K/$2" | sed "s|^|$2:|"; }

echo "== 0. 来源"
echo "kit $URL"
echo "kit sha256 $(shasum -a 256 "$Z" | cut -d' ' -f1)，$(unzip -l "$Z" | tail -1 | awk '{print $2}') 个文件"
g "commit|public runtime|Public runtime|included evaluator files match" README.md

echo "== 1. 评测按 checkpoint 自己的配置重建（kit 的 evaluate_policy 配置）"
g "rebuilt from the checkpoint" flash_chord/src/flash_chord/configs/evaluate_policy.yaml
g "use_checkpoint_config|deterministic|world_count|start_frame" flash_chord/src/flash_chord/configs/evaluation/flash_sac.yaml

echo "== 2. 示例批量评测：每条 episode 默认几个世界"
g "world-count|evaluation.world_count|one trained checkpoint per episode" examples/run_policy_evaluation.py

echo "== 3. 官方计分设置（config/tracks.json）"
python3 - "$K/config/tracks.json" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1]))
print("config/tracks.json: 顶层键", sorted(k for k in c if k != "_comment"))
for key in ("accepted_eval_script_sha256", "accepted_eval_code_sha256"):
    print(f"config/tracks.json: {key} = {c.get(key)}")
for name, spec in c["tracks"].items():
    if name.startswith("track_2"):
        keep = {k: spec.get(k) for k in ("scored_episodes", "private_fraction", "worlds_per_episode", "metrics", "reference", "data_root")}
        print(f"config/tracks.json: {name} {json.dumps(keep, ensure_ascii=False)}")
EOF

echo "== 4. 导出轨迹的帧号（上游 main aee95f2，FlashCHORD rollout.py 补丁里新增的行）"
gh api repos/nvidia-isaac/video_to_data/commits/aee95f25ae0943f335d8bb42fa95b6a9acfa266f \
  --jq '.files[] | select(.filename | endswith("evaluation/rollout.py")) | .patch' \
  | grep -nE '^\+.*("reference_frame"|"time_s"|control_fps\)|row_order)' | sed 's|^|upstream aee95f2 rollout.py patch:|'
gh api repos/nvidia-isaac/video_to_data/commits/aee95f25ae0943f335d8bb42fa95b6a9acfa266f \
  --jq '.commit.message' | grep -nE "Public v0.4.0|lacks" | sed 's|^|upstream aee95f2 commit message:|'

echo "== 5. kit 读轨迹：frame_index 取自哪一列"
g "reference_frame|time_s  " v2dlb/submission_parquet.py

echo "== 6. 打包器：要求的（世界, 帧, 物体）格子、缺行报错、参考不一致的警告、缺 episode 报错"
g "range\(worlds_expected\), ref.frame_index|absent = wanted.difference|evaluator numbers rows by reference_frame|reference the evaluator recorded does not match|scored episodes have no trajectory" tools/pack_submission.py
g "frame_index=frame_index\[keep\]|A frame is dropped" v2dlb/trajectory.py

echo "== 7. 样例提交 CSV 的格子"
python3 - "$K/data/track_2_sample_submission.csv" <<'EOF'
import collections, csv, sys
frames = collections.defaultdict(list); worlds = set(); rows = 0
with open(sys.argv[1]) as f:
    reader = csv.reader(f); header = next(reader)
    for row in reader:
        rows += 1
        _, episode, world, frame, _ = row[0].split("/")
        frames[episode].append(int(frame[1:])); worlds.add(world)
print(f"data/track_2_sample_submission.csv: {rows} 行，{len(frames)} 条 episode，世界 {sorted(worlds)}")
for episode in ("e000020", "e000026", "e000013", "e000010", "e000011"):
    fr = sorted(frames[episode])
    print(f"data/track_2_sample_submission.csv: {episode} {len(fr)} 帧，第 {fr[0]}–{fr[-1]} 帧，连续 {fr == list(range(fr[0], fr[-1] + 1))}")
EOF

echo "== 8. Track 2 说明：每条 episode 一个 checkpoint、全部计分 episode、SP-SR / MP-SR 的口径"
g "one checkpoint per episode|Include all scored episodes|Fraction of episodes|averages across episodes|Accept the rules" tracks/track_2/README.md

echo "== 9. 打包器与上传：本地重算指标、带哈希、同一个 CSV 传这一档的四个 Kaggle 比赛"
g "recomputes Track 2 metrics locally|carries evaluator and checkpoint hashes|checks scored episodes, frames and objects" tools/pack_submission.py
g "four Kaggle metric competitions|same CSV for all four|sends it to all four|packer adds its own SHA-256" tracks/track_2/README.md
