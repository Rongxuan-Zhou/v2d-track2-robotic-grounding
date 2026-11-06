#!/usr/bin/env bash
# 组委会 10/1 更新 kit 之后的事实摘录（Mac 上执行，只读：本地两份 kit 加 gh api 读上游源码）。给“组委会三问怎么定”的决定当证据，
# 决定记录 agent-doc/decision-organizer-q1-q3-2026-10-01.md。每行带版本、文件与行号，供 jev 核对。
#   1. 运行时：新旧两份 kit 的 README 各钉哪个提交；上游 #188 的说明；aee95f2 → 8c778f9 之间 flash_chord 改了哪些文件；三处 support.py 的 md5
#   2. 终止：评测器怎样关掉任务失败终止（阈值改成多少），ReconBody 的终止判断怎么比较，harness 何时报错
#   3. 空支撑面：support.py 对空文件报什么错；setup.py 在 include_support 为 false / 找不到支撑面时各做什么
# 用法（仓库根）：scripts/dev/record submission/runtime_pin_facts stages/evaluate/runtime_pin_facts.sh
set -uo pipefail
KOLD=third_party/v2d_submission_kit/update_a2de2aee
KNEW=third_party/v2d_submission_kit/update_0f2f9f4b
B=robotic_grounding/flash_chord
OLD=aee95f25ae0943f335d8bb42fa95b6a9acfa266f; NEW=8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef
src() { gh api "repos/nvidia-isaac/video_to_data/contents/$B/$2?ref=$1" --jq .content | base64 -d; }

echo "== 1. 运行时"
for k in "$KOLD" "$KNEW"; do
  echo "kit ${k}：zip sha256 $(shasum -a 256 "$k/v2d_submission_kit.zip" | cut -c1-16)…"
  grep -n -E '^ +commit +[0-9a-f]{40}|^Public runtime:' "$k/v2d_submission_kit/README.md" | sed "s|^|  README.md:|"
done
grep -i 'last-modified' "$KNEW/headers.txt" | tr -d '\r' | sed 's/^/新 kit 的 HTTP 头 /'
echo "上游 #188（e2df8819fa）提交说明里关于 kit 的几行："
gh api repos/nvidia-isaac/video_to_data/commits/e2df8819fa --jq '.commit.committer.date + " " + (.commit.message | split("\n")[0])' | sed 's/^/  /'
gh api repos/nvidia-isaac/video_to_data/commits/e2df8819fa --jq .commit.message | grep -A8 '^## Submission kit fixes' | sed 's/^/  /'
echo "aee95f2 → 8c778f9 之间改动的文件（GitHub compare）："
gh api "repos/nvidia-isaac/video_to_data/compare/$OLD...$NEW" --jq '.files[] | "  " + .status + " " + .filename'
for ref in "$OLD" "$NEW"; do echo "support.py @ ${ref:0:7} md5 $(src "$ref" src/flash_chord/scene/support.py | md5 -q)"; done
grep -h -E 'support.py|4d6eab11' results/ml40/diverge/support_import_p185.txt 2>/dev/null | head -3 | sed 's/^/ml40 上的 #185 补丁副本：/'

echo "== 2. 终止（8c778f9；harness.py 与 aee95f2 的 md5 相同）"
for ref in "$OLD" "$NEW"; do echo "harness.py @ ${ref:0:7} md5 $(src "$ref" src/flash_chord/evaluation/harness.py | md5 -q)"; done
src "$NEW" src/flash_chord/evaluation/harness.py | grep -n -E '_DISABLED_THRESHOLD = |The tracking-failure terms are disabled|keeps advancing instead of freezing|termination\[name\] = _DISABLED_THRESHOLD|completion.terminated.any|disable task failure termination' | sed 's/^/  harness.py:/'
src "$NEW" src/flash_chord/lifecycle/termination.py | grep -n -E 'pelvis_position_error > pelvis_position_threshold|max_palm_position_error > palm_position_threshold|max_object_position_error > object_position_threshold|terminated\[world\] = wp.max\(pelvis_failed' | sed 's/^/  termination.py:/'
src "$NEW" src/flash_chord/configs/termination/recon_body.yaml | grep -n -E 'threshold' | sed 's/^/  configs\/termination\/recon_body.yaml:/'

echo "== 3. 空支撑面（8c778f9）"
src "$NEW" src/flash_chord/scene/support.py | grep -n -E 'contains no Cylinder or Cube collision shapes' | sed 's/^/  support.py:/'
src "$NEW" src/flash_chord/scene/setup.py | grep -n -E 'support_usda = support_usda_for_reference|if include_support and support_usda is None|scene support is enabled, but no support asset' | sed 's/^/  setup.py:/'
grep -n 'use_checkpoint_config' "$KNEW/v2d_submission_kit/flash_chord/src/flash_chord/configs/evaluation/flash_sac.yaml" | sed 's/^/  kit configs\/evaluation\/flash_sac.yaml:/'
