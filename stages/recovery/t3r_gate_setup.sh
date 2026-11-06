#!/usr/bin/env bash
# t3takeover 第 1 级 B 部分（支撑面 usda）：在我们的目录里搭 gate 仓库（训练机 ml40 上执行；只写我们的目录）。计划 agent-doc/plan-t3takeover-v3-2026-10-21.md 第 1 节第 2 条与复核第 1 条。
# 1. 代码从 stdin 收我们仓库 f0ac98f 的 tar（Mac 上 `git archive f0ac98f libs stages scripts tools third_party`），解到 /lab_data/rxzhou/v2d/t3recon/gate/repo，记 .v2d-source-commit；
# 2. third_party/video_to_data 用我们自己训练用的那份 v0.4.0（33129dd，核 HEAD），rsync 工作树（不带 .git）；
# 3. scripts/dev/incontainer 换成她 gate 里的替身（不进 git、解释器写死成她 gate venv 里的 python，只读调用；复核第 1 条），核 sha256 等于登记值，原文件留作 incontainer.orig；
# 4. 打出 SOMA 资源与替身的 sha256（来历）。gate/repo 已存在就拒绝（退出码 2）。最后一行 T3R_GATE_SETUP_DONE。
# 用法：git archive f0ac98f libs stages scripts tools third_party | ssh ml40 "<仓库>/stages/recovery/t3r_gate_setup.sh f0ac98f"
# T3R_ENV=ours（第 1b 级，agent-doc/plan-t3takeover-1b-2026-10-21.md 第 1 节第 4 条）：搭在 t3recon/gate_e，incontainer 用我们的替身 stages/recovery/t3r_incontainer_ours
# （解释器是 t3recon/env/gate），不碰她的替身、只核她的目录；第 1 级用过的 t3recon/gate 不动。
set -euo pipefail
cd "$(dirname "$0")/../.."
REPO="$(pwd)"
COMMIT=${1:?提交号}
case "${T3R_ENV:-ours}" in hers) G=/lab_data/rxzhou/v2d/t3recon/gate ;; ours) G=/lab_data/rxzhou/v2d/t3recon/gate_e ;; *) echo "T3R_GATE_SETUP_FAILED T3R_ENV 只认 hers、ours"; exit 2 ;; esac
V="$REPO/third_party/video_to_data"
STANDIN_SHA=3b39d5ff07b6   # 她 gate/incontainer 与她 gate/repo/scripts/dev/incontainer 的 sha256 前 12 位（10/21 05:4x MDT 只读核对）
[ -e "$G/repo" ] && { echo "T3R_GATE_SETUP_FAILED $G/repo 已存在"; exit 2; }   # $G 本身已有 dyn7 要导入的 4 个 gate 模块（部署代码时放的），同她的布局
if [ "${T3R_ENV:-ours}" = hers ]; then
  H="$("$REPO/stages/recovery/t3r_other.sh")" || { echo "T3R_GATE_SETUP_FAILED 另一位成员的目录或解释器没过核对"; exit 2; }
fi
[ "$(git -C "$V" --no-optional-locks rev-parse --short=7 HEAD)" = 33129dd ] || { echo "T3R_GATE_SETUP_FAILED 我们的 video_to_data 不是 33129dd"; exit 2; }
mkdir -p "$G/repo" "$G/logs"
tar -x -C "$G/repo"
echo "$COMMIT" > "$G/repo/.v2d-source-commit"
rsync -a --exclude .git "$V/" "$G/repo/third_party/video_to_data/"
cp "$G/repo/scripts/dev/incontainer" "$G/incontainer.orig"
if [ "${T3R_ENV:-ours}" = hers ]; then
  cp --no-preserve=all "$H/gate/incontainer" "$G/repo/scripts/dev/incontainer"
  got="$(sha256sum < "$G/repo/scripts/dev/incontainer" | cut -c1-12)"
  [ "$got" = "$STANDIN_SHA" ] || { echo "T3R_GATE_SETUP_FAILED 替身的 sha256 是 $got，不是登记的 $STANDIN_SHA"; exit 3; }
else
  cp --no-preserve=all "$REPO/stages/recovery/t3r_incontainer_ours" "$G/repo/scripts/dev/incontainer"
fi
chmod +x "$G/repo/scripts/dev/incontainer"
echo "替身 incontainer sha256 $(sha256sum < "$G/repo/scripts/dev/incontainer" | cut -c1-64)"
S="$G/repo/third_party/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/body_models/soma"
find "$S" -type f ! -name .gitattributes -printf '%P\n' | sort | while read -r f; do echo "SOMA $(sha256sum < "$S/$f" | cut -c1-16) $f"; done
find "$G" -type l -printf '%p -> %l\n' | head -5
echo "T3R_GATE_SETUP_DONE"
