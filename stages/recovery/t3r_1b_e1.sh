#!/usr/bin/env bash
# t3takeover 第 1b 级判据 E1 的整条重跑（训练机 ml40 上执行，放 tmux；只用 CPU）。计划 agent-doc/plan-t3takeover-1b-2026-10-21.md 第 1 节第 4–6 条、判据 E1。
# 依次（每步失败只记下、不拦后面的步）：
#   0. 回归：默认环境（她的解释器）在 t3 根上重跑比较驱动，记录名加 _reg；各段的判定行必须与第 1 级的记录逐行相同（脚本改动没改行为）；
#   1. 用我们的环境搭 gate_e（T3R_ENV=ours stages/recovery/t3r_gate_setup.sh，代码包是已核过 sha256 的 t3recon/gate_f0ac98f.tar）；
#   2. t3_e：只种输入，T3R_ENV=ours 从 decode 整条重跑，与 t3（第 1 级、她的解释器）比，记录名加 _1b；
#   3. t3_ew：种输入与参照的 world 产物，T3R_ENV=ours 从 assemble 起跑，与参照比（含三项对照与门槛自检，比较器用我们的宿主机环境）；
#   4. usda：T3R_ENV=ours 在 gate_e 上对 t3_ew 跑 mkroot、重定向与 t3_support_usda，与参照的 usda、support_usda.json 及第 1 级的 refw 结果行比；
#   5. 报警检查：她目录下属于我们的文件只有已知的那个、修改时间不变；t3recon 下的链接不出我们的目录。
# 用法：stages/recovery/t3r_1b_e1.sh   日志 runs/logs/t3r_1b_e1.log，最后一行 T3R_1B_E1_DONE。
set -uo pipefail
cd "$(dirname "$0")/../.."
R=/lab_data/rxzhou/v2d/t3recon
exec >> runs/logs/t3r_1b_e1.log 2>&1
say() { echo "=== $(date '+%F %T %Z') $*"; }
step() { local n=$1; shift; say "开始 $n"; "$@"; say "$n 退出码 $?"; }
verdicts() { grep -h -E '^(BYTES_EQUAL|NUMBERS_EQUAL|DIFFERENT|门槛自检|T3R_COMPARE_DONE)' "$1" 2>/dev/null; }   # 整行比：同样的根、同样的比较器

say "开始，来历 $(cat .v2d-source-commit 2>/dev/null || echo unknown)"
step regression env T3R_ENV=hers T3R_TAG=reg nice -n 10 stages/recovery/t3r_level1_compare.sh t3   # 回归照第 1 级：她的解释器（默认已改成 ours）
for f in results/t3recon/t3_vs_ref_cmp_reg_*.txt*; do
  n="${f#results/t3recon/t3_vs_ref_cmp_reg_}"; old="results/t3recon/t3_vs_ref_cmp_$n"
  [ -e "$old" ] || old="${old%.failed}"; [ -e "$old" ] || old="$old.failed"
  if [ "$(verdicts "$f")" = "$(verdicts "$old")" ]; then say "回归 $n 判定行与第 1 级相同"; else say "回归 $n 判定行与第 1 级不同"; fi
done
step gate_e_setup env T3R_ENV=ours scripts/dev/record t3recon/gate_e_setup stages/recovery/t3r_gate_setup.sh f0ac98f < "$R/gate_f0ac98f.tar"
step seed_t3_e scripts/dev/record t3recon/seed_t3_e stages/recovery/t3r_seed_root.sh t3_e inputs
step level1_t3_e env T3R_ENV=ours nice -n 10 stages/recovery/t3r_level1.sh t3_e
step compare_t3_e env T3R_ENV=ours T3R_TAG=1b nice -n 10 stages/recovery/t3r_level1_compare.sh t3_e t3
step seed_t3_ew scripts/dev/record t3recon/seed_t3_ew stages/recovery/t3r_seed_root.sh t3_ew world-from-ref
step level1_t3_ew env T3R_ENV=ours T3R_FROM=assemble nice -n 10 stages/recovery/t3r_level1.sh t3_ew
step compare_t3_ew env T3R_ENV=ours T3R_TAG=1b nice -n 10 stages/recovery/t3r_level1_compare.sh t3_ew
step usda_t3_ew env T3R_ENV=ours T3R_TAG=1b stages/recovery/t3r_level1_usda.sh t3_ew
say "报警检查"
find "$(T3R_OTHER_DIR_ONLY=1 stages/recovery/t3r_other.sh)" -user rxzhou -printf '%p %TY-%Tm-%Td %TH:%TM:%TS\n' 2>/dev/null | sed -e 's#^/lab_data/[^/]*/#/lab_data/<其他用户>/#'
find "$R" -type l | while read -r l; do r="$(readlink -f "$l")"; case "$r" in /lab_data/rxzhou/*|/usr/bin/*) ;; *) echo "外链 $l";; esac; done   # venv 的基础解释器链到系统的 /usr/bin/python3.10，不算外链（10/21 第一次跑时报过 6 条这样的）
say "T3R_1B_E1_DONE"
