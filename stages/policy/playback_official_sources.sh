#!/usr/bin/env bash
# 官方文档与各发布版本的代码里，关于 G1 参考“按什么速度回放”的原文摘录（回答“官方评测按什么速度播放”）。
# 在 Mac 上执行（要 gh 已登录；只读 GitHub，不碰训练机）；输出经 scripts/dev/record 落到 results/playback/。
# 摘：赛事页面源码（main 分支 docs/v2d_challenge/index.html）里 Track 2 与评测相关的句子；
#     release/0.2.0、release/0.3.0、v0.4.0 三个版本里 Isaac Lab 全身 tracking command 怎样取参考帧；
#     FlashCHORD G1 配方逐帧回放的原文见 results/playback/upstream_playback_facts.txt（v0.4.0，训练机上的副本）。
# 用法（仓库根）：scripts/dev/record playback/official_playback_sources stages/policy/playback_official_sources.sh
set -euo pipefail
R=nvidia-isaac/video_to_data
T=robotic_grounding/source/robotic_grounding/robotic_grounding
C=${T}/tasks/v2d_whole_body/mdp/commands
get() { gh api "repos/${R}/contents/${2}?ref=${1}" --jq '.content' | base64 -d; }
show() {  # show <ref> <path> <正则>：印出所有匹配行，每行带“版本 文件名:行号”，按版本过滤时不会串行
  echo "--- ${1}:${2}（匹配 ${3}）"
  get "$1" "$2" | grep -n -E -- "$3" | cut -c1-200 | sed "s|^|${1} $(basename "$2"):|" || echo "${1} $(basename "$2"):（没有匹配）"
}
exists() { gh api "repos/${R}/contents/${2}?ref=${1}" --jq '.path' > /dev/null 2>&1; }

echo "== 0. 版本"
git ls-remote "https://github.com/${R}" HEAD refs/heads/main refs/heads/release/0.2.0 refs/heads/release/0.3.0 refs/tags/v0.4.0

echo "== 1. 赛事页面（main:docs/v2d_challenge/index.html）里与 Track 2、评测、回放有关的句子"
get main docs/v2d_challenge/index.html | python3 -c '
import html, re, sys
t = sys.stdin.read()
t = re.sub(r"<script.*?</script>|<style.*?</style>", "", t, flags=re.S)
t = html.unescape(re.sub(r"<[^>]+>", "\n", t))
seen = set()
for ln in (re.sub(r"\s+", " ", x).strip() for x in t.split("\n")):
    if ln and ln not in seen and re.search(r"(?i)robotic grounding|eval_robotic|recorded evaluation|simulator|evaluated against|object-tracking|playback|fps|frame rate|hz\b|real.?time|speed", ln):
        seen.add(ln)
        print("  " + ln[:400])
'
echo "  页面源码里 playback / fps / frame rate / Hz / real time / speed 的出现次数："
get main docs/v2d_challenge/index.html | grep -o -i -E "playback|[0-9 ]fps|frame rate|[0-9] ?hz|real.?time|motion speed" | sort | uniq -c || echo "  0"

count() {  # count <ref> <path> <正则>：这个版本的文件里匹配的行数（没有匹配也打出 0）
  echo "${1} $(basename "$2"): 含 /${3}/ 的行数 $(get "$1" "$2" | grep -c -E -- "$3" || true)"
}
for ref in release/0.2.0 release/0.3.0 v0.4.0; do
  echo "== 2. ${ref}：Isaac Lab 全身 tracking command 取参考帧的方式"
  show "$ref" "${C}/tracking_utils.py" 'def load_motion_data|resample|target_fps|resolve_playback_timing|step_dt'
  show "$ref" "${C}/tracking_command_cfg.py" 'motion_speed|target_fps|Resample|uses data as-is|dt: float|dt = '
  show "$ref" "${C}/tracking_command.py" 'target_fps|self\.timestep(\[[a-z_]+\])? \+= 1|playback_dt|load_motion_data\('
  show "$ref" "${T}/tasks/v2d_whole_body/base_env_cfg.py" 'decimation =|sim\.dt =|dt=0'
  count "$ref" "${C}/tracking_utils.py" 'resample'
  count "$ref" "${C}/tracking_command.py" 'resample\(|target_fps'
  echo "--- ${ref}: 参考帧号自增所在的方法（tracking_command.py）"
  get "$ref" "${C}/tracking_command.py" | grep -n -E 'def _update_command|self\.timestep(\[past_freeze\])? \+= 1' | sed "s|^|${ref} tracking_command.py:|"
  if exists "$ref" "${T}/motion_schema/timing.py"; then
    show "$ref" "${T}/motion_schema/timing.py" 'advance by one reference frame|sampled at|automatic resampling|return env_step_dt'
  else
    echo "--- ${ref}:${T}/motion_schema/timing.py：这个版本没有这个文件"
  fi
done
