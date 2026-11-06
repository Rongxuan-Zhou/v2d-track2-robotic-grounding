#!/usr/bin/env bash
# 提交前检查的 R5、R6（计划 agent-doc/plan-submission-check-2026-10-03.md）：Mac 上、仓库根执行，只读。
# R5：代码链接能不能被组委会打开——仓库可见性（gh api，只读）、CSV 里的提交是否在 origin、kit 里要求代码链接与公开代码的原文（带 文件:行号）。
# R6：给用户的提交清单——提交次数、决赛份数、截止时间、队伍名的原文（带 文件:行号），与 Kaggle 报名状态记录里的截止时间。
# 用法：scripts/dev/record submission/rules_facts_1003 stages/evaluate/submission_rules_facts.sh
set -uo pipefail
cd "$(dirname "$0")/../.."
K=third_party/v2d_submission_kit/update_0f2f9f4b/v2d_submission_kit
CSV=runs/submission/csv/track2_tier1_full30_1003.csv
g() { grep -nE "$1" "$K/$2" | head -"${3:-3}" | sed "s|^|$2:|" | cut -c1-260; }

echo "== R5 代码链接"
URL="$(sed -n 2p "$CSV" | awk -F, '{print $NF}')"
COMMIT="${URL##*/}"
echo "CSV 里的 code_commit_url：$URL"
echo "提交 $COMMIT 所在的远端分支：$(git branch -r --contains "$COMMIT" | tr -d ' ' | tr '\n' ' ')"
echo "仓库可见性（gh api repos/Manifold-ai/maniphys_lab_v2d_challenge，只读）：$(gh api repos/Manifold-ai/maniphys_lab_v2d_challenge --jq '"private=\(.private) visibility=\(.visibility)"' 2>&1)"
g "repo shared with the V2D account" tools/pack_submission.py
g "Section 2.9.a" v2dlb/report_schema.py
g "must be publicly released under" config/tracks.json 1
g "code_release_license" config/tracks.json 1

echo "== R6 提交清单"
g '"freeze"' config/tracks.json 1
g '"submission_limit"' config/tracks.json 1
g '"submission_policy"' config/tracks.json 1
g '"max_daily_submissions"' config/tracks.json 1
g '"max_final_submissions"' config/tracks.json 1
g "for metric, slug in competitions.items\(\)" submit/v2d_submit.py 2
g "submit_one\(api, slug, args.submission" submit/v2d_submit.py 1
g "team_name" submit/v2d_submit.py 4
g "Rename your team" submit/v2d_submit.py 1
echo "Kaggle 显示的截止（results/submission/kaggle_join_status_1003.txt）："
grep -E "track2-tier[12]" results/submission/kaggle_join_status_1003.txt | sed 's/^/  /'
echo "RULES_FACTS_DONE"
