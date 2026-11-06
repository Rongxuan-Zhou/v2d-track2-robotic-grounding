#!/usr/bin/env bash
# t3fix 的复制清单生成（训练机上执行；只读：只 find、stat，不写她的目录）。计划 agent-doc/plan-t3fix-2026-10-27.md 第 1 节 D1、第 0 节第 8 条。
# 在 T3R_SRC（另一位成员的 v2d 目录，运行时由调用方给，仓库里不写）下列出 gb5 交付层 30 集做 A、C 与重跑 dyn7 之后各步要的文件，打成 t3r_copy.sh 的清单：
#   类别 in、源（相对于 T3R_SRC）、目标（相对于我们的 t3recon 根，放在 t3fx/ 下，路径去掉开头的 tier3/）。
# 列的是：input 的 meta、mesh、ground_plane；runs 下 gb5_best 的选择表、dr2s 与 dyn7 两层（含 _suspect）及它们的 json 记录、dr2s 的物理检查、floor、support；
# 各集转换目录（convert*_gb5/<集>/，不要 *.pt）；各集掩码 masks_k0.h5（按选择表的 convert 找 work 目录，同 t3_align_check.WORK）；dev/walls。cari4d 大目录不拷。
# 用法：T3R_SRC=<她的目录> stages/recovery/t3r_fix_files.sh > <清单>   缺掩码就停（退出码 1）。
set -euo pipefail
SRC=${T3R_SRC:?要设 T3R_SRC}
T=tier3
cd "$SRC"
# 每个源：不含制表符、换行与 ..，realpath 落在 SRC 下（软链接解析后也一样），否则停（退出码 2）
emit() {
  local real_src; real_src="$(realpath "$SRC")"
  while IFS= read -r f; do
    case "$f" in *$'\t'*|*..*) echo "T3R_FIX_FILES_REFUSED 源路径不合格：$f" >&2; exit 2 ;; esac
    case "$(realpath "$f")" in "$real_src"/*) ;; *) echo "T3R_FIX_FILES_REFUSED 源在 T3R_SRC 外：$f" >&2; exit 2 ;; esac
    printf 'in\t%s\tt3fx/%s\n' "$f" "${f#"$T"/}"
  done
}
{
  find -L "$T/input/meta" "$T/input/mesh" "$T/input/ground_plane" -type f
  find -L "$T/runs" -maxdepth 1 -type f \( -name 'tier3_cari4d_gb5_best.selection.json' -o -name 'tier3_cari4d_gb5_best_dr2s.*json' -o -name 'tier3_cari4d_gb5_best_dr2s_dyn7.*json' \)
  for v in tier3_cari4d_gb5_best_dr2s tier3_cari4d_gb5_best_dr2s_suspect tier3_cari4d_gb5_best_dr2s_dyn7 tier3_cari4d_gb5_best_dr2s_dyn7_suspect; do
    [ -d "$T/runs/$v" ] && find -L "$T/runs/$v" -type f
  done
  [ -d "$T/runs/physics/tier3_cari4d_gb5_best_dr2s" ] && find -L "$T/runs/physics/tier3_cari4d_gb5_best_dr2s" -type f
  find -L "$T/runs/floor" -maxdepth 1 -type f
  [ -d "$T/runs/support" ] && find -L "$T/runs/support" -type f
  [ -d "$T/dev/walls" ] && find -L "$T/dev/walls" -type f
  # -I：不把当前目录（她的目录）放进 sys.path，也不读 PYTHON* 环境变量
  python3 -I - "$T/runs/tier3_cari4d_gb5_best.selection.json" <<'EOF'
import json, re, sys
from pathlib import Path
OK = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]*$")
work = {"convert": "work", "convert_v2": "work_v2", "convert_fp": "work_fp", "convert_pivot": "work_fp"}
sel = Path(sys.argv[1])
runs = sel.parent
for r in json.loads(sel.read_text()):
    conv, name = r["convert"], r["episode"]
    if not (OK.match(conv) and OK.match(name)) or ".." in conv or ".." in name:
        sys.exit(f"选择表里的名字不合格：{conv!r} {name!r}")
    base = conv.removesuffix("_gb5").removesuffix("_gb4").removesuffix("_gb3")
    for p in sorted((runs / conv / name).rglob("*")):
        if p.is_file() and p.suffix != ".pt":
            print(p)
    m = runs / work[base] / name / "masks_k0.h5"
    if not m.is_file():
        sys.exit(f"缺掩码：{m}")
    print(m)
EOF
} | sort -u | emit
