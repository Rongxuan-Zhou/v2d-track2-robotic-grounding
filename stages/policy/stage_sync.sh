#!/usr/bin/env bash
# 训练机上把暂存的源码树同步进仓库副本（仓库根执行；arm_after.sh 的 V2D_AFTER_SYNC_FROM 调用）。
# 计划 agent-doc/plan-t12night-2026-10-30.md 第 2 节：链在跑时 scripts/dev/remote 拒绝同步，新段的代码先由 Mac 用同一套过滤规则
# rsync 到 runs/stage/<来历>/（runs/ 受保护，主同步不碰它），等链都结束再由本脚本在机器本地做 scripts/dev/remote 那一步。
# 过滤规则与 scripts/dev/remote 逐条相同（P 保护 .env、data、runs、results、third_party/video_to_data、哨兵与来历文件；H 不传 results；
# 复用 .gitignore；不传 .git）；另外不传暂存目录里的 .v2d-source-commit，同步完才写进仓库副本。
# 用法：bash stages/policy/stage_sync.sh <暂存目录>   V2D_DRYRUN=1 只列出会传/会删什么（只读，有长链标记也能跑）
# 前提（任何一条不过就退出码 2，什么也不改）：仓库根有 .v2d-remote-root；暂存目录在 runs/stage/ 下、有 .v2d-source-commit 与
# stages/policy/arm_var_chain.sh；没有 runs/.chain-active-*（干跑不查）。最后一行 STAGE_SYNC_DONE <来历>。
set -euo pipefail
SRCD=${1:?用法：stage_sync.sh <暂存目录>}
die() { echo "stage_sync: $*" >&2; exit 2; }
[ -f .v2d-remote-root ] || die "不在仓库副本根（没有 .v2d-remote-root）"
SRCD="$(cd "$SRCD" 2>/dev/null && pwd -P)" || die "暂存目录不存在：$1"
case "$SRCD" in "$(pwd -P)"/runs/stage/?*) ;; *) die "暂存目录要在 runs/stage/ 下：$SRCD" ;; esac
[ -s "$SRCD/.v2d-source-commit" ] && [ -f "$SRCD/stages/policy/arm_var_chain.sh" ] || die "暂存目录不完整：$SRCD"
c="$(cat "$SRCD/.v2d-source-commit")"
[[ $c =~ ^[0-9a-f]{12}$ ]] || die "暂存的来历不是 12 位干净版本：$c"
[ "${V2D_DRYRUN:-0}" = 1 ] || [ -z "$(ls runs/.chain-active-* 2>/dev/null)" ] || die "还有长链标记：$(ls runs/.chain-active-* | tr '\n' ' ')"
RS=(-a -c --delete)   # -c：按内容比，不靠大小与修改时间（同一秒写下、大小相同的改动会被漏掉，test_arm_after.py 8 实测）
[ "${V2D_DRYRUN:-0}" = 1 ] && RS+=(-n -i)
[ "${V2D_DRYRUN:-0}" = 1 ] || printf '%s\n' "$c-syncing" > .v2d-source-commit
rsync "${RS[@]}" \
      --filter='P /.env' \
      --filter='P /data/***' \
      --filter='P /runs/***' \
      --filter='H /results/***' --filter='P /results/***' \
      --filter='P /third_party/video_to_data/***' \
      --filter='P /.v2d-remote-root' \
      --filter='- /.v2d-source-commit' --filter='P /.v2d-source-commit' \
      --filter='+ /.env.example' \
      --filter=':- .gitignore' \
      --exclude '.git/' \
      "$SRCD/" ./
[ "${V2D_DRYRUN:-0}" = 1 ] && exit 0
printf '%s\n' "$c" > .v2d-source-commit
echo "STAGE_SYNC_DONE $c"
