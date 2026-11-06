#!/usr/bin/env bash
# 生成与上游 8c778f905c 逐字节相同的 flash_chord 源码副本：组委会 10/1 新 kit 钉的评测运行时。导出与评测时用 V2D_FC_SRC 挂到 /workspace
# （scripts/dev/in-flash-chord）。计划 agent-doc/plan-box6-and-eval-runtime-2026-10-01.md 第 3 节。
# 做法同 make_patched_fc_src.sh：复制 third_party 里 v0.4.0（33129dd）的 flash_chord（不含 outputs/、logs/），打
# third_party/patches/flash_chord-v0.4.0-to-8c778f9.patch（tools/make_fc_patch.sh 从 GitHub 生成），写 SOURCE_V2D.txt，
# 再用 fc_src_check.py 核：与 v0.4.0 不同的文件正好是清单 flash_chord-8c778f9.sha256 的 11 个、sha256 全对。任何一步失败（含核不过）
# 都删掉整个目标目录，退出码非零。8c778f9 的依赖文件与 v0.4.0 相同（results/submission/eval_runtime_facts.txt），沿用
# flash_chord:latest 镜像。原 checkout 不动；目标已存在就退出。
# 用法（仓库根，训练机上）：stages/policy/make_fc_src_8c778f9.sh <目标目录>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"; cd "$ROOT"
DST=${1:?用法：make_fc_src_8c778f9.sh <目标目录>}
[ -e "$DST" ] && { echo "已存在，不覆盖：$DST" >&2; exit 2; }
SRC=third_party/video_to_data/robotic_grounding/flash_chord
PATCH=third_party/patches/flash_chord-v0.4.0-to-8c778f9.patch
MAN=third_party/patches/flash_chord-8c778f9.sha256
HEAD="$(git -C third_party/video_to_data rev-parse --short=7 HEAD)"
[ "$HEAD" = 33129dd ] || { echo "third_party/video_to_data 在 ${HEAD}，不是 v0.4.0（33129dd）" >&2; exit 2; }
mkdir -p "$DST"
done_ok=0; trap '[ "$done_ok" = 1 ] || rm -rf "$DST"' EXIT   # 中途失败就整个删掉，重跑时不会被“已存在”挡住；目标原本存在时上面已经退出，碰不到它
rsync -a --exclude outputs --exclude logs --exclude __pycache__ --exclude '*.pyc' "$SRC/" "$DST/"
# 补丁里的路径是 a/robotic_grounding/flash_chord/…：在副本根下用 -p3 去掉前三级；新增文件的一侧是 /dev/null
patch -p3 -d "$DST" --dry-run --quiet < "$PATCH"
patch -p3 -d "$DST" --quiet < "$PATCH"
printf 'flash_chord 8c778f905c（v0.4.0 33129dd + %s）\n与 %s 列的 11 个文件逐字节相同，其余与 v0.4.0 相同（stages/policy/fc_src_check.py）\n生成于 %s；仓库来历 %s\n' \
  "$PATCH" "$MAN" "$(date '+%F %T %Z')" "$(cat .v2d-source-commit 2>/dev/null || echo unknown)" > "$DST/SOURCE_V2D.txt"
if ! python3 stages/policy/fc_src_check.py "$DST" "$MAN"; then
  echo "核对不过，删掉整个 ${DST}（in-flash-chord 认不到它）" >&2; exit 1
fi
done_ok=1
echo "FC_SRC_8C778F9_OK $DST"
