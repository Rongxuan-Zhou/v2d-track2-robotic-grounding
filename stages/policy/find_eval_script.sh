#!/usr/bin/env bash
# 查官方评测脚本（eval_robotic_grounding.py 等）有没有公开发布：HF 数据集的全部文件与提交记录、GitHub 上游全部分支、代码搜索、发布附件。
# 在 Mac 上执行（要 gh 已登录；只读）；输出经 scripts/dev/record 落到 results/playback/。
# 用法（仓库根）：scripts/dev/record playback/eval_script_search stages/policy/find_eval_script.sh
set -euo pipefail
DS=nvidia/video_to_data_challenge
GH=nvidia-isaac/video_to_data
PAT='eval_robotic_grounding|eval_e2e|eval_reconstruction'

echo "== HF 数据集 ${DS}"
curl -s --max-time 60 "https://huggingface.co/api/datasets/${DS}" | python3 -c '
import json, sys
d = json.load(sys.stdin)
print("  lastModified", d.get("lastModified"), "sha", (d.get("sha") or "")[:10])'
curl -s --max-time 60 "https://huggingface.co/api/datasets/${DS}/commits/main" | python3 -c '
import json, sys
for c in json.load(sys.stdin):
    print("  commit", c.get("date"), c.get("id", "")[:10], (c.get("title") or "")[:80])'
curl -s --max-time 60 "https://huggingface.co/api/datasets/${DS}/tree/main?recursive=true" | python3 -c '
import json, re, sys
d = json.load(sys.stdin)
py = [x["path"] for x in d if x["path"].endswith(".py")]
hit = [x["path"] for x in d if re.search(sys.argv[1], x["path"])]
print(f"  全部 {len(d)} 个条目；.py 文件 {len(py)} 个；路径含评测脚本名的 {len(hit)} 个 {hit}")' "$PAT"

echo "== GitHub ${GH}：各分支的完整文件树里有没有评测脚本"
for b in $(gh api "repos/${GH}/branches?per_page=100" --jq '.[].name'); do
  gh api "repos/${GH}/git/trees/${b}?recursive=1" | python3 -c '
import json, re, sys
d = json.load(sys.stdin)
hit = [x["path"] for x in d["tree"] if re.search(sys.argv[2], x["path"])]
t, n = d["truncated"], len(d["tree"])   # Mac 自带的 python3 不许 f-string 表达式里有反斜杠
print("  " + sys.argv[1] + "（truncated=" + str(t) + "，" + str(n) + " 个条目）：" + (" ".join(hit) if hit else "没有"))' "$b" "$PAT"
done
echo "== GitHub 代码搜索（默认分支）：eval_robotic_grounding"
gh search code "eval_robotic_grounding" --repo "${GH}" --limit 20 | cut -c1-160 | sed 's/^/  /'
echo "== GitHub 发布与附件"
gh api "repos/${GH}/releases?per_page=20" --jq '.[] | "  \(.tag_name) \(.published_at) 附件：\([.assets[].name] | if length == 0 then "无" else join(", ") end)"'
