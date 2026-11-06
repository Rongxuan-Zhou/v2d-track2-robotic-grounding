#!/usr/bin/env bash
# 在另一位成员目录只读的前提下跑一条命令（训练机上执行，只用 Linux 的 unshare / mount / setpriv）。计划 agent-doc/plan-t3takeover-v3-2026-10-21.md 第 1 节第 1 条与复核第 3 条。
# 她的目录对我们可写（同属一个组），靠自觉防不住写入；这里在系统层把它变成只读：
#   第 0 级（外面）：查参数——每个含 “/” 的参数（与 key=value 的 value）按 realpath 落在 T3R_HER 下就拒绝，T3R_ALLOW（冒号分隔，按 realpath 比）里的除外；
#                    相对路径按 T3R_ROOT 解析（命令在那里跑）。T3R_ROOT 本身不能在 T3R_HER 下。
#   第 1 级（unshare -rm 的私有用户 + 挂载命名空间）：把 T3R_HER 绑定挂载并改成只读，用 /dev/null 盖住 /run/docker.sock（里面不能起容器），cd 到 T3R_ROOT；
#   第 2 级（setpriv 丢掉全部能力，不能再把挂载改回可写）：探针——[ -w T3R_HER ] 为假、[ -w T3R_ROOT ] 为真、/run/docker.sock 不是 socket；
#                    都过才 exec 命令，环境加 PYTHONDONTWRITEBYTECODE=1、GIT_OPTIONAL_LOCKS=0。
#   第 1b 级（agent-doc/plan-t3takeover-1b-2026-10-21.md 复核改动第 1 条）：T3R_MASK=<目录>:<目录>（都要在 T3R_HER 下）——第 1 级里把这些目录用空的只读 tmpfs 盖住，
#                    她的解释器与 site-packages 在命名空间里就不存在；第 2 级探针查它们是空的、不可写。T3R_ALLOW 里的程序打出 realpath 与 sha256（stderr），记下实际用的是谁。
# 用法：T3R_HER=<她的目录> T3R_ROOT=<我们的根> [T3R_ALLOW=<解释器>:<解释器>] stages/recovery/t3r_run.sh <命令...>
# 拒绝时打一行 T3R_REFUSED <原因>：参数或根的问题退出码 2，探针不过退出码 3；其余退出码是命令自己的。测试：stages/recovery/test_t3r_run.py（只在 Linux 上跑）。
set -euo pipefail
SELF="$(realpath -e "$0")"
refuse() { echo "T3R_REFUSED $2"; exit "$1"; }

if [ "${1:-}" = __t3r_stage1 ]; then
  shift
  mount --bind "$T3R_HER" "$T3R_HER"
  mount -o remount,bind,ro "$T3R_HER"
  if [ -n "${T3R_MASK:-}" ]; then
    IFS=: read -r -a masks <<< "$T3R_MASK"
    for m in "${masks[@]}"; do mount -t tmpfs -o ro,size=4k,mode=0555 t3r_mask "$m"; done
  fi
  if [ -e /run/docker.sock ]; then mount --bind /dev/null /run/docker.sock; fi
  cd "$T3R_ROOT"
  exec setpriv --bounding-set=-all --inh-caps=-all -- /bin/bash "$SELF" __t3r_stage2 "$@"
fi

if [ "${1:-}" = __t3r_stage2 ]; then
  shift
  [ ! -w "$T3R_HER" ] || refuse 3 "命名空间里她的目录仍可写"
  [ -w "$T3R_ROOT" ] || refuse 3 "命名空间里我们的根不可写"
  [ ! -S /run/docker.sock ] || refuse 3 "命名空间里 docker socket 没被盖住"
  grep -q '^CapEff:[[:space:]]*0*$' /proc/self/status || refuse 3 "能力没有丢干净"
  if [ -n "${T3R_MASK:-}" ]; then
    IFS=: read -r -a masks <<< "$T3R_MASK"
    for m in "${masks[@]}"; do
      [ -z "$(ls -A "$m" 2>/dev/null)" ] && [ ! -w "$m" ] || refuse 3 "命名空间里遮挡的目录不是空的或可写"
    done
  fi
  if [ -n "${T3R_ALLOW_REAL:-}" ]; then
    IFS=: read -r -a exes <<< "$T3R_ALLOW_REAL"
    for e in "${exes[@]}"; do [ -f "$e" ] && echo "T3R_EXE $e sha256 $(sha256sum < "$e" | cut -c1-16)" >&2; done
  fi
  export PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0
  exec "$@"
fi

HER="${T3R_HER:?要设 T3R_HER}"; ROOT="${T3R_ROOT:?要设 T3R_ROOT}"
[ $# -ge 1 ] || refuse 2 "没有命令"
HER="$(realpath -e "$HER")" || refuse 2 "T3R_HER 不存在"
ROOT="$(realpath -e "$ROOT")" || refuse 2 "T3R_ROOT 不存在"
case "$ROOT/" in "$HER"/*) refuse 2 "根在她的目录下" ;; esac
ALLOW=()
if [ -n "${T3R_ALLOW:-}" ]; then
  IFS=: read -r -a raw <<< "$T3R_ALLOW"
  for a in "${raw[@]}"; do [ -n "$a" ] && ALLOW+=("$(realpath -m -- "$a")"); done
fi
under_her() {
  local tok=$1 r a
  case "$tok" in */*) ;; *) return 1 ;; esac
  case "$tok" in /*) r="$(realpath -m -- "$tok")" ;; *) r="$(cd "$ROOT" && realpath -m -- "$tok")" ;; esac
  case "$r/" in "$HER"/*) ;; *) return 1 ;; esac
  for a in "${ALLOW[@]}"; do [ "$r" = "$a" ] && return 1; done
  return 0
}
MASK=()
if [ -n "${T3R_MASK:-}" ]; then
  IFS=: read -r -a raw <<< "$T3R_MASK"
  for m in "${raw[@]}"; do
    [ -n "$m" ] || continue
    r="$(realpath -e -- "$m")" && [ -d "$r" ] || refuse 2 "T3R_MASK 里有不存在或不是目录的项"
    case "$r/" in "$HER"/?*) ;; *) refuse 2 "T3R_MASK 只能盖她目录下的子目录" ;; esac
    MASK+=("$r")
  done
fi
for arg in "$@"; do
  under_her "$arg" && refuse 2 "参数落在她的目录下"
  case "$arg" in *=*) under_her "${arg#*=}" && refuse 2 "参数的值落在她的目录下" ;; esac
done
export T3R_HER="$HER" T3R_ROOT="$ROOT" T3R_MASK="$(IFS=:; echo "${MASK[*]}")" T3R_ALLOW_REAL="$(IFS=:; echo "${ALLOW[*]}")"
exec unshare -rm --propagation private /bin/bash "$SELF" __t3r_stage1 "$@"
