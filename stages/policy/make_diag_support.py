"""生成只换了支撑面的诊断用运动目录（只用标准库；计划 agent-doc/plan-ep1-ep7-divergence-2026-09-30.md 附录 A.3）。

flash_chord 按 <whole_body>/reconstructed_stage/<SEQ>[_<robot>]_support.usda 找支撑面（assets/registry.py:126–131），
所以把一条序列挂到一个新的运动根目录下、只换那份 usda，训练队列里把 parquet 路径换成新根下的即可：
  <新根>/tier1/whole_body/soma/sequence_id=<SEQ>/robot_name=g1/data.parquet   复制原 parquet（网格、URDF 是绝对路径，照旧指向原目录）
  <新根>/tier1/whole_body/soma/sequence_id=<SEQ>/object                       相对符号链接到原 object 目录
  <新根>/tier1/whole_body/reconstructed_stage/<SEQ>_support.usda             改过的支撑面
改法：--thicken <prim 名> <厚度 m> <保留的面 +x|-x|+y|-y|+z|-z>：把这个 Cube 沿该轴加厚到给定厚度，保留的那个面位置不变
（例如桌面保留 +z：顶面高度不变，往下加厚）。只处理 translate + scale 的轴对齐 Cube（与 support.py 建碰撞体的方式一致）。
原目录只读；新目录里已有同名文件就报错退出，不覆盖。
用法（仓库根，训练机上）：python3 stages/policy/make_diag_support.py <原运动根> <新运动根> <SEQ> --thicken tabletop 0.04 +z
  例：python3 stages/policy/make_diag_support.py runs/motion runs/diag/motion_thickbox 2026-05-29_15-20-43_iron_round_table_02 --thicken tabletop 0.04 +z
"""
import argparse
import os
import re
import shutil
import sys
from pathlib import Path

AXES = {"x": 0, "y": 1, "z": 2}


def vec(text: str) -> list:
    return [float(v) for v in text.split(",")]


def fmt(v: list) -> str:
    return "(" + ", ".join(f"{x:.6g}" for x in v) + ")"


def thicken(usda: str, prim: str, thickness: float, face: str) -> tuple:
    m = re.search(rf'def Cube "{re.escape(prim)}"[^{{]*\{{(.*?)\n    \}}', usda, re.S)
    if not m:
        raise SystemExit(f"找不到 Cube \"{prim}\"")
    body = m.group(1)
    size_m = re.search(r"double size = ([0-9.eE+-]+)", body)
    size = float(size_m.group(1)) if size_m else 2.0  # USD Cube 默认 size 2
    t = vec(re.search(r"xformOp:translate = \(([^)]*)\)", body).group(1))
    sm = re.search(r"xformOp:scale = \(([^)]*)\)", body)
    if not sm:
        raise SystemExit(f"Cube \"{prim}\" 没有 xformOp:scale，脚本只处理 translate + scale")
    s = vec(sm.group(1))
    if not re.fullmatch(r"[+-][xyz]", face):
        raise SystemExit(f"保留的面要写成 +x、-z 这样：{face}")
    sign, a = (1.0 if face[0] == "+" else -1.0), AXES[face[1]]
    half_old = 0.5 * size * s[a]
    kept = t[a] + sign * half_old
    half_new = 0.5 * thickness
    t2, s2 = list(t), list(s)
    t2[a] = kept - sign * half_new
    s2[a] = thickness / size
    new_body = re.sub(r"xformOp:translate = \([^)]*\)", f"xformOp:translate = {fmt(t2)}", body, count=1)
    new_body = re.sub(r"xformOp:scale = \([^)]*\)", f"xformOp:scale = {fmt(s2)}", new_body, count=1)
    out = usda[: m.start(1)] + new_body + usda[m.end(1):]
    report = (f"Cube \"{prim}\"：沿 {face[1]} 轴厚度 {2 * half_old * 1000:.1f} → {thickness * 1000:.1f} mm；"
              f"保留的 {face} 面 {kept:+.4f} → {t2[a] + sign * half_new:+.4f} m；中心 {fmt(t)} → {fmt(t2)}；缩放 {fmt(s)} → {fmt(s2)}")
    return out, report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src_root", type=Path)
    ap.add_argument("dst_root", type=Path)
    ap.add_argument("seq")
    ap.add_argument("--thicken", nargs=3, metavar=("PRIM", "THICKNESS_M", "KEEP_FACE"), required=True)
    ap.add_argument("--robot", default="g1")
    a = ap.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", a.seq):
        sys.exit(f"序列名非法：{a.seq}")
    wb_src, wb_dst = a.src_root / "tier1/whole_body", a.dst_root / "tier1/whole_body"
    seq_src = wb_src / "soma" / f"sequence_id={a.seq}"
    stage_src = wb_src / "reconstructed_stage"
    robot_specific = stage_src / f"{a.seq}_{a.robot}_support.usda"
    if robot_specific.exists():
        sys.exit(f"原目录有机器人专属的 {robot_specific.name}，训练会优先用它；本脚本只处理通用的那份，先人工确认")
    usda_src = stage_src / f"{a.seq}_support.usda"
    parquet_src = seq_src / f"robot_name={a.robot}" / "data.parquet"
    for p in (usda_src, parquet_src, seq_src / "object"):
        if not p.exists():
            sys.exit(f"缺 {p}")
    seq_dst = wb_dst / "soma" / f"sequence_id={a.seq}"
    usda_dst = wb_dst / "reconstructed_stage" / f"{a.seq}_support.usda"
    parquet_dst = seq_dst / f"robot_name={a.robot}" / "data.parquet"
    obj_dst = seq_dst / "object"
    for p in (usda_dst, parquet_dst, obj_dst):
        if p.exists() or p.is_symlink():
            sys.exit(f"已存在，不覆盖：{p}")
    new_usda, report = thicken(usda_src.read_text(), a.thicken[0], float(a.thicken[1]), a.thicken[2])
    parquet_dst.parent.mkdir(parents=True)
    usda_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(parquet_src, parquet_dst)
    link = os.path.relpath(seq_src / "object", seq_dst)  # 相对链接：宿主机与容器（仓库挂在 /repo）里都能解析
    os.symlink(link, obj_dst)
    usda_dst.write_text(new_usda)
    print(report)
    print(f"写出 {usda_dst}\n复制 {parquet_dst}\n链接 {obj_dst} → {link}")
    print(f"训练队列里的 parquet（容器内）：/repo/{parquet_dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
