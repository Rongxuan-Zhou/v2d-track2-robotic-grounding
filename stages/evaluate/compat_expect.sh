#!/usr/bin/env bash
# 兼容性验证里两种“期望失败”的行（stages/evaluate/compat_8c778f9.sh 调用，经 scripts/dev/record 落盘）。计划 agent-doc/plan-box6-and-eval-runtime-2026-10-01.md 第 3.3 节。
#   guard  <run> <步数> <世界数>：守卫开着导出含 NaN 的 policy，期望 fc_export.sh 退出码 12（规矩一：含 NaN 的 policy 不导出）。
#   negctl <run> <步数> <世界数>：守卫关着导出（V2D_EXPORT_UNGUARDED=1）。符合期望的结局有三种：退出码 13（评测器报
#          must disable task failure termination，规矩二的触发条件）、退出码 3（别的评测失败）、退出码 0 但 check_export.py 的最后一行是 EXPORT_FAIL。
#          退出码 0 且核对通过，说明这套核对拦不住坏 policy，这一行失败。
# 退出码 0 = 符合期望，1 = 不符合，2 = 用法错。环境变量 V2D_FC_SRC 原样传给 fc_export.sh。
set -uo pipefail
cd "$(dirname "$0")/../.."
KIND=${1:-}; RUN=${2:-}; STEP=${3:-}; W=${4:-}
[ -n "$W" ] || { echo "用法：compat_expect.sh guard|negctl <run> <步数> <世界数>" >&2; exit 2; }
rc=0
case "$KIND" in
  guard)
    V2D_EXPORT_WORLDS="$W" V2D_EXPORT_UNGUARDED=0 stages/evaluate/fc_export.sh "$RUN" "$STEP" || rc=$?
    echo "fc_export.sh 退出码 ${rc}（期望 12）"
    [ "$rc" = 12 ] && { echo "COMPAT_EXPECT_OK guard"; exit 0; }
    echo "COMPAT_EXPECT_FAIL guard"; exit 1 ;;
  negctl)
    V2D_EXPORT_WORLDS="$W" V2D_EXPORT_UNGUARDED=1 stages/evaluate/fc_export.sh "$RUN" "$STEP" || rc=$?
    echo "fc_export.sh 退出码 ${rc}（期望 13、3，或 0 且核对不过）"
    case "$rc" in
      13) echo "COMPAT_EXPECT_OK negctl：评测器报 must disable task failure termination"; exit 0 ;;
      3)  echo "COMPAT_EXPECT_OK negctl：评测失败（不是终止报错），见 runs/logs/$RUN.export_8c778f9_s${STEP}_w${W}.log"; exit 0 ;;
      0)  out="$(stages/evaluate/check_export.sh "$RUN" "$STEP" "$W" 2>&1)"; crc=$?; echo "$out"
          # 核对脚本自己崩了（Traceback 等）不算“核对不过”：最后一行必须是 EXPORT_FAIL
          if [ "$crc" != 0 ] && tail -1 <<< "$out" | grep -q '^EXPORT_FAIL'; then
            echo "COMPAT_EXPECT_OK negctl：导出成功但核对不过（上面列了不过的项）"; exit 0
          fi
          echo "COMPAT_EXPECT_FAIL negctl：含 NaN 的 policy 导出成功，核对没有给出 EXPORT_FAIL（退出码 ${crc}）"; exit 1 ;;
      *)  echo "COMPAT_EXPECT_FAIL negctl：退出码 $rc 不在期望之内"; exit 1 ;;
    esac ;;
  *) echo "未知类型 $KIND" >&2; exit 2 ;;
esac
