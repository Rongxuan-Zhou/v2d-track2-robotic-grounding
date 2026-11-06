#!/usr/bin/env bash
# 生成“v0.4.0 + 上游 #185（8c778f9：方盒支撑面按 size × scale 导入）”的 flash_chord 源码副本，诊断 run 用 V2D_FC_SRC 挂到 /workspace
# （scripts/dev/in-flash-chord）。计划 agent-doc/plan-ep1-ep7-divergence-2026-09-30.md 附录 A 第 4 版。
# 只复制 third_party/video_to_data/robotic_grounding/flash_chord（不含 outputs/、logs/），打 third_party/patches/ 里的补丁，
# 核对打补丁后的 support.py 与上游 8c778f9 的同名文件逐字节相同（sha256），写 SOURCE_V2D.txt。原 checkout 不动；目标已存在就退出。
# 用法（仓库根，训练机上）：stages/policy/make_patched_fc_src.sh <目标目录> [<第二个补丁> <清单> [<第三个补丁> <清单>]]
#   给了第二个补丁（例如 third_party/patches/flash_chord-v0.4.0-nonfinite-guard.patch）：#185 之后再打它（同样 -p3，先 --dry-run），
#   再按清单（例如 third_party/patches/flash_chord-p185-nfg.sha256，每行“sha256  相对 flash_chord 目录的路径”）逐个核对 sha256，
#   SOURCE_V2D.txt 第一行写明两个补丁。不给时行为与原来完全相同。计划 agent-doc/plan-ep7-eval-and-guards-2026-10-02.md 的 B.3。
#   第三个补丁（例如 third_party/patches/flash_chord-v0.4.0-object-orientation.patch，计划 agent-doc/plan-r3o-2026-10-15.md 2.1）：
#   第二个之后再打，按第三份清单（要列出与 v0.4.0 不同的全部文件，同 fc_src_check.py）核对；只给第三个、不给第二个时拒绝。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd -P)"; cd "$ROOT"
DST=${1:?用法：make_patched_fc_src.sh <目标目录> [<第二个补丁> <清单> [<第三个补丁> <清单>]]}
PATCH2=${2:-}; MANIFEST2=${3:-}; PATCH3=${4:-}; MANIFEST3=${5:-}
if [ -n "$PATCH2" ]; then
  [ -f "$PATCH2" ] && [ -f "$MANIFEST2" ] || { echo "第二个补丁与清单都要给，且文件要在：$PATCH2 $MANIFEST2" >&2; exit 2; }
fi
if [ -n "$PATCH3" ]; then
  [ -n "$PATCH2" ] || { echo "给了第三个补丁就要先给第二个" >&2; exit 2; }
  [ -f "$PATCH3" ] && [ -f "$MANIFEST3" ] || { echo "第三个补丁与清单都要给，且文件要在：$PATCH3 $MANIFEST3" >&2; exit 2; }
fi
[ -e "$DST" ] && { echo "已存在，不覆盖：$DST" >&2; exit 2; }
SRC=third_party/video_to_data/robotic_grounding/flash_chord
PATCH=third_party/patches/flash_chord-v0.4.0-cube-support-scale-8c778f9.patch
WANT=8d82f6008f2310667bef6862dfa3f453d3bdd188c79c7283a26e288a5a19e1c2   # 上游 8c778f905c 的 scene/support.py
HEAD="$(git -C third_party/video_to_data rev-parse --short=7 HEAD)"
[ "$HEAD" = 33129dd ] || { echo "third_party/video_to_data 在 $HEAD，不是 v0.4.0（33129dd）" >&2; exit 2; }
mkdir -p "$DST"
rsync -a --exclude outputs --exclude logs "$SRC/" "$DST/"
# 补丁里的路径是 a/robotic_grounding/flash_chord/src/...：在副本根下用 -p3 去掉前三级
patch -p3 -d "$DST" --dry-run --quiet < "$PATCH"
patch -p3 -d "$DST" --quiet < "$PATCH"
GOT="$(sha256sum "$DST/src/flash_chord/scene/support.py" | cut -d' ' -f1)"
[ "$GOT" = "$WANT" ] || { echo "打补丁后 support.py 的 sha256 $GOT ≠ 上游 8c778f9 的 $WANT" >&2; exit 1; }
LABEL="$PATCH"
if [ -n "$PATCH2" ]; then
  patch -p3 -d "$DST" --dry-run --quiet < "$PATCH2"
  patch -p3 -d "$DST" --quiet < "$PATCH2"
  while read -r want path; do
    [ -n "$want" ] || continue
    got="$(sha256sum "$DST/$path" | cut -d' ' -f1)"
    [ "$got" = "$want" ] || { echo "打补丁后 ${path} 的 sha256 ${got} ≠ 清单 ${MANIFEST2} 的 ${want}" >&2; exit 1; }
  done < "$MANIFEST2"
  LABEL="$PATCH + $PATCH2"
fi
if [ -n "$PATCH3" ]; then
  patch -p3 -d "$DST" --dry-run --quiet < "$PATCH3"
  patch -p3 -d "$DST" --quiet < "$PATCH3"
  while read -r want path; do
    [ -n "$want" ] || continue
    got="$(sha256sum "$DST/$path" | cut -d' ' -f1)"
    [ "$got" = "$want" ] || { echo "打补丁后 ${path} 的 sha256 ${got} ≠ 清单 ${MANIFEST3} 的 ${want}" >&2; exit 1; }
  done < "$MANIFEST3"
  LABEL="$LABEL + $PATCH3"
fi
printf 'flash_chord v0.4.0（33129dd）+ %s\nscene/support.py sha256 %s，与上游 8c778f905c 相同\n生成于 %s；仓库来历 %s\n' \
  "$LABEL" "$GOT" "$(date '+%F %T %Z')" "$(cat .v2d-source-commit 2>/dev/null || echo unknown)" > "$DST/SOURCE_V2D.txt"
echo "PATCHED_FC_SRC_OK $DST"
