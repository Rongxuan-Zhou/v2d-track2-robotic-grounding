#!/usr/bin/env bash
# 找出另一位成员的 v2d 目录并核对它与她的两个解释器可信，再打印目录路径（训练机上执行；只读）。计划 agent-doc/plan-t3takeover-v3-2026-10-21.md；
# 10/21 自动安全复核：不要靠通配符猜，候选必须恰好一个，解释器的属主与写权限要核。仓库里不写她的目录名，所以在运行时找。
# 核对（任一不过就打 T3R_OTHER_REFUSED <原因> 到 stderr、退出码 2）：
#   1. /lab_data/*/v2d 去掉 rxzhou 自己的，恰好一个候选；它是目录、不是软链接；
#   2. 目录属主不是 rxzhou；目录对“其他用户”不可写（目录本身可能是机器的管理账户建的，属主不一定是她）；
#   3. 宿主机解释器 <v2d>/venv/bin/python 的真实文件属 root、对其他用户不可写；
#      gate 解释器 <v2d>/gate/venv/bin/python 的真实文件属路径里 /lab_data/<用户>/ 的那个用户、对其他用户不可写；
#   4. 两个解释器真实文件的 sha256 打到 stderr（来历）。
# 残余风险（记在计划里）：她的文件对整组可写，组里其他成员能改 gate 解释器；第 1b 级换成我们自己的环境后消除。
# 用法：H="$(stages/recovery/t3r_other.sh)" || exit 2
# 第 1b 级（agent-doc/plan-t3takeover-1b-2026-10-21.md 复核改动第 8 条）：T3R_OTHER_DIR_ONLY=1 只做第 1、2 条（她的目录），不核她的解释器——1b 不调用它们，
# 她那边换了解释器也不该挡住 1b。
set -euo pipefail
refuse() { echo "T3R_OTHER_REFUSED $1" >&2; exit 2; }
mapfile -t C < <(ls -d /lab_data/*/v2d 2>/dev/null | grep -v '^/lab_data/rxzhou/')
[ "${#C[@]}" = 1 ] || refuse "候选目录有 ${#C[@]} 个，不是 1 个"
H="${C[0]}"
[ -d "$H" ] && [ ! -L "$H" ] || refuse "候选不是普通目录"
owner="$(stat -c %U "$H")"
[ "$owner" != rxzhou ] || refuse "候选目录属于我们自己"
user="$(basename "$(dirname "$H")")"
[ "$user" != rxzhou ] || refuse "候选路径是我们自己的"
perm="$(stat -c %A "$H")"; [ "${perm:8:1}" != w ] || refuse "候选目录对其他用户可写"
check_exe() {   # check_exe <路径> <期望属主>
  local p r o a
  p="$1"; r="$(realpath -e "$p")" || refuse "解释器不存在"
  o="$(stat -c %U "$r")"; a="$(stat -c %A "$r")"
  [ "$o" = "$2" ] || refuse "解释器真实文件的属主不是预期的"
  [ "${a:8:1}" != w ] || refuse "解释器真实文件对其他用户可写"
  echo "解释器 $(basename "$p") → sha256 $(sha256sum "$r" | cut -c1-16)…" >&2
}
if [ "${T3R_OTHER_DIR_ONLY:-0}" != 1 ]; then
  check_exe "$H/venv/bin/python" root
  check_exe "$H/gate/venv/bin/python" "$user"
fi
printf '%s\n' "$H"
