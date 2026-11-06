#!/usr/bin/env bash
# 一个档位的几条序列逐条跑 one.sh（转换 → 重定向 → 对齐检查），最后打印每条的结果行与日志尾。训练机上、仓库根执行，放 tmux 经 record 落盘
# （Mac 断网时不留半截记录；计划 agent-doc/plan-t3-pilot-2026-10-09.md 第 3 节，审查 n8）。
# one.sh 的退出码非 0 只表示状态不是 OK（ALIGN、NOCONTACT 或 FAIL:*），这里不当失败：照样跑下一条，状态由计划的判据来判。
# 用法：stages/retarget/eps.sh <档位> <ep>...    例：scripts/dev/record t3p/retarget_mf4090 stages/retarget/eps.sh 3 8 24
# Tier 3 的校正版：V2D_T3_SET=<名> stages/retarget/eps.sh 3 8 24（目录与日志名带 _<名>，见 one.sh）。
set -uo pipefail
cd "$(dirname "$0")/../.."
T=${1:?用法：eps.sh <档位> <ep>...}; shift
[ $# -ge 1 ] || { echo "eps: 要至少一条序列" >&2; exit 2; }
DN=$(python3 -c "import sys; sys.path.insert(0, 'libs/v2d_infra'); from paths import tier_dirname; print(tier_dirname($T))") || exit 2
P=t${T}${DN#tier$T}
echo "来历 $(cat .v2d-source-commit 2>/dev/null)，$(date '+%F %T %Z')；目录 runs/motion_src/${DN}、runs/motion/$DN"
for e in "$@"; do
  stages/retarget/one.sh "$T" "$e"; echo "one.sh $T $e 退出码 $?（$(date '+%T')）"
done
echo "## 结果行（档位 ep 序列 帧数 fps 左手接触 右手接触 IK中位 间隙mm 支撑面 状态）"
for e in "$@"; do cat "runs/logs/batch/${P}_ep$(printf %02d "$e").tsv"; done
# 脚底高度那一行单独先打：支撑面的几何体多时它会被挤出最后 15 行（计划 agent-doc/plan-t3-v2-retarget18-2026-10-10.md，审查 6）
for e in "$@"; do
  f="runs/logs/batch/${P}_ep$(printf %02d "$e").log"; echo "## ${f}（脚底高度与最后 15 行）"
  grep -a "Saved robot sole Z range" "$f" | tail -n 1 | cut -c1-300; tail -n 15 "$f" | cut -c1-300
done
echo EPS_DONE
