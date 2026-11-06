# t3takeover 的脚本 source 这个文件，再调 t3r_env <仓库根>：按 T3R_ENV 一次定好解释器、遮挡与 gate 目录（只定变量，不跑命令）。
# 计划 agent-doc/plan-t3takeover-1b-2026-10-21.md 复核改动第 1、3、8 条。
#   T3R_ENV=ours（不设时就是它；10/21 07:4x MDT 第 1b 级判据 E1 过了之后改成默认，agent-doc/plan-t3takeover-1b-2026-10-21.md 执行记录）：
#                 t3recon/env/host 与 t3recon/env/gate，gate 目录 t3recon/gate_e；她的 venv、gate/venv、gate/.py 经 T3R_MASK 在命名空间里被空的只读 tmpfs 盖住，
#                 她的解释器跑不起来；t3r_other.sh 只核她的目录。
#   T3R_ENV=hers（只在明确设了时；第 1 级的做法，复现第 1 级的记录或做回归时用）：她的宿主机 venv 与 gate venv（只读命名空间里调用），gate 目录 t3recon/gate，不遮挡。
# 设好：H（她的 v2d 目录）、HPY、GPY、GATE_VENV_BIN、GDIR；导出 T3R_HER，ours 时另导出 T3R_MASK。T3R_ROOT、T3R_ALLOW 由调用方定。
t3r_env() {
  local repo=$1 R=/lab_data/rxzhou/v2d/t3recon
  case "${T3R_ENV:-ours}" in
    hers)
      H="$("$repo/stages/recovery/t3r_other.sh")" || return 2
      HPY="$H/venv/bin/python"; GPY="$H/gate/venv/bin/python"; GATE_VENV_BIN="$H/gate/venv/bin"; GDIR="$R/gate"
      unset T3R_MASK ;;
    ours)
      H="$(T3R_OTHER_DIR_ONLY=1 "$repo/stages/recovery/t3r_other.sh")" || return 2
      HPY="$R/env/host/bin/python"; GPY="$R/env/gate/bin/python"; GATE_VENV_BIN="$R/env/gate/bin"; GDIR="$R/gate_e"
      [ -x "$HPY" ] && [ -x "$GPY" ] || { echo "T3R_ENV=ours 但 $R/env 下没有 host、gate 两个环境" >&2; return 2; }
      export T3R_MASK="$H/venv:$H/gate/venv:$H/gate/.py" ;;
    *)
      echo "T3R_ENV 只认 hers、ours" >&2; return 2 ;;
  esac
  T3R_HER="$(dirname "$H")"; export T3R_HER
  echo "T3R_ENV ${T3R_ENV:-ours}：宿主机 $(realpath -e "$HPY" 2>/dev/null)，gate $(realpath -e "$GPY" 2>/dev/null)，gate 目录 $GDIR" >&2
}
