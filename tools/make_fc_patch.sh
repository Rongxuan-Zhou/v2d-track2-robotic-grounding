#!/usr/bin/env bash
# 从 GitHub 生成“v0.4.0（33129dd）→ 上游 8c778f905c”在 robotic_grounding/flash_chord 下的完整补丁、打补丁后应有的文件 sha256 清单，
# 以及 8c778f9 评测代码的两个摘要（Mac 上执行，只读：gh api 读上游源码）。计划 agent-doc/plan-box6-and-eval-runtime-2026-10-01.md 第 3 节。
# 8c778f9 是组委会 10/1 新 kit 钉的评测运行时；比 v0.4.0 多 6 个提交，flash_chord 下只改了评测代码、scene/support.py 与测试。
# 输出（都提交进仓库）：
#   third_party/patches/flash_chord-v0.4.0-to-8c778f9.patch  逐文件 diff -u 拼成，路径 a/robotic_grounding/flash_chord/…，在副本根上 -p3 打
#   third_party/patches/flash_chord-8c778f9.sha256            改动文件在 8c778f9 的 sha256（路径相对 flash_chord 目录），fc_src_check.sh 用
#   third_party/patches/flash_chord-8c778f9.eval.json         8c778f9 的 eval_code_sha256（按 harness.evaluation_code_sha256 的算法）
#                                                             与 scripts/evaluate_policy.py 的 sha256（导出轨迹里的 eval_script_sha256 应等于它）
# 用法（仓库根）：tools/make_fc_patch.sh
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=nvidia-isaac/video_to_data
OLD=33129dd; NEW=8c778f905cb35f76f8e6cd6e2ea486da4fb9e0ef
B=robotic_grounding/flash_chord
P=third_party/patches/flash_chord-v0.4.0-to-8c778f9.patch
M=third_party/patches/flash_chord-8c778f9.sha256
J=third_party/patches/flash_chord-8c778f9.eval.json
T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
src() { gh api "repos/$REPO/contents/$2?ref=$1" --jq .content | base64 -d; }

gh api "repos/$REPO/compare/$OLD...$NEW" --jq ".files[] | select(.filename | startswith(\"$B/\")) | .status + \" \" + .filename" > "$T/files"
[ -s "$T/files" ] || { echo "compare 里没有 $B 下的文件" >&2; exit 1; }
{
  echo "# 上游 nvidia-isaac/video_to_data 从 v0.4.0（33129dd）到 8c778f905c（2026-09-29，#180–#185）在 $B 下的全部改动，"
  echo "# 逐文件取自 GitHub 后用 diff -u 拼成（tools/make_fc_patch.sh 生成，不要手改）。改动只在评测代码、scene/support.py 与测试。"
  echo "# 用途：给 v0.4.0 打补丁，生成与 8c778f9 逐字节相同的评测运行时副本（stages/policy/make_fc_src_8c778f9.sh），不改 third_party/video_to_data。"
  echo "# 打补丁后各文件应与 flash_chord-8c778f9.sha256 一致。"
} > "$T/patch"
: > "$T/manifest"
while read -r status f; do
  rel="${f#$B/}"
  case "$status" in
    modified) src "$OLD" "$f" > "$T/old"; src "$NEW" "$f" > "$T/new"; from="a/$f" ;;
    added)    : > "$T/old"; src "$NEW" "$f" > "$T/new"; from=/dev/null ;;
    *) echo "不支持的改动类型 ${status}：$f" >&2; exit 1 ;;
  esac
  rc=0; diff -u --label "$from" --label "b/$f" "$T/old" "$T/new" >> "$T/patch" || rc=$?
  [ "$rc" -le 1 ] || { echo "diff 出错（${rc}）：$f" >&2; exit 1; }
  printf '%s  %s\n' "$(shasum -a 256 "$T/new" | cut -d' ' -f1)" "$rel" >> "$T/manifest"
done < "$T/files"

mkdir -p "$T/eval"
for n in __init__.py harness.py metrics.py rollout.py; do src "$NEW" "$B/src/flash_chord/evaluation/$n" > "$T/eval/$n"; done
src "$NEW" "$B/scripts/evaluate_policy.py" > "$T/evaluate_policy.py"
python3 - "$T/eval" "$T/evaluate_policy.py" "$NEW" > "$T/eval.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
d, script, ref = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
h = hashlib.sha256()
for name in sorted(("__init__.py", "harness.py", "metrics.py", "rollout.py")):   # 同 harness.py 的 _EVALUATION_CODE_MODULES
    h.update(name.encode()); h.update(b"\0"); h.update((d / name).read_bytes())
print(json.dumps({"upstream": ref, "eval_code_sha256": h.hexdigest(),
                  "eval_script_sha256": hashlib.sha256(script.read_bytes()).hexdigest()}, indent=1))
PY
cp "$T/patch" "$P"; cp "$T/manifest" "$M"; cp "$T/eval.json" "$J"
echo "写出 ${P}（$(grep -c "^+++ " "${P}") 个文件）、${M}、${J}"
cat "$M" "$J"
