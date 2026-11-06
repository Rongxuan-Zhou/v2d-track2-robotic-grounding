#!/usr/bin/env bash
# 取官方 submission kit 的当前版本（Mac 上、仓库根执行）：下载 zip 与响应头到 third_party/v2d_submission_kit/update_<sha256 前 8 位>/ 并解压。
# 那个目录已存在就拒绝（不覆盖已有的 kit）；third_party/v2d_submission_kit/ 不进 git。只从官方地址下载，不执行 kit 里的任何东西。
# 用法：scripts/dev/record submission/fetch_kit_<日期> stages/evaluate/fetch_kit.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
URL=https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_submission_kit.zip
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
curl -fsSL -D "$TMP/headers.txt" -o "$TMP/v2d_submission_kit.zip" "$URL"
SHA="$(shasum -a 256 "$TMP/v2d_submission_kit.zip" | cut -d' ' -f1)"
DST="third_party/v2d_submission_kit/update_${SHA:0:8}"
echo "下载 ${URL}：$(wc -c < "$TMP/v2d_submission_kit.zip" | tr -d ' ') 字节，sha256 ${SHA}"
grep -iE '^(last-modified|etag|content-length):' "$TMP/headers.txt" | tr -d '\r' | sed 's/^/  /'
[ ! -e "$DST" ] || { echo "${DST} 已存在，不覆盖"; exit 2; }
mkdir -p "$DST"
mv "$TMP/v2d_submission_kit.zip" "$TMP/headers.txt" "$DST/"
unzip -q "$DST/v2d_submission_kit.zip" -d "$DST"
echo "解压到 ${DST}：$(find "$DST/v2d_submission_kit" -type f | wc -l | tr -d ' ') 个文件"
echo "FETCH_KIT_OK $DST"
