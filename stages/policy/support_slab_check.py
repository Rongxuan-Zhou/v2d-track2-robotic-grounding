"""核支撑面在训练导入器里建出来的形状与 usda 原文一致（Mac 上跑，只用标准库，只读已落盘的记录）。
计划 agent-doc/plan-t3-hm-p2-2026-10-12.md 第 4 节第 9 步、第 5 节“支撑面在训练导入器里”（Fable 5.1 审查 1）。
输入：一份记录，里面先是 “## usda <文件名>” 开头的 usda 原文，再是 stages/policy/support_import_check.py 的输出
（“导入器 … md5 <md5>”，“== <文件名>”，每个形状一行 “<名>：类型 BOX，shape_scale（Box 为半尺寸）(x, y, z)，位置 (x, y, z)”）。
每个文件里每个 def Cube：size × |scale| / 2 要等于导入器给的半尺寸（差 ≤ 1e-6 m），xformOp:translate 要等于位置（差 ≤ 1e-4 m，位置印到 4 位小数），
类型要是 BOX，形状的名字与个数两边相同。全部一致打 SLAB_OK <n>/<n>，否则打 SLAB_BAD 与不一致的文件、名字和原因，退出码 1。
只认不旋转的 Cube（交付的支撑面就是这种）；usda 里有别的形状（Cylinder 等）就报不认识、退出码 1。
用法：python3 stages/policy/support_slab_check.py <记录>
测试：stages/policy/test_support_slab_check.py。
"""
import re
import sys
from pathlib import Path

USDA = re.compile(r"^## usda (\S+)$")
BUILT = re.compile(r"^== (\S+)$")
SHAPE = re.compile(r"^\s*(\S+)：类型 (\w+)，shape_scale（Box 为半尺寸）\(([^)]*)\)，位置 \(([^)]*)\)")
DEF = re.compile(r'^\s*def\s+(\w+)\s+"([^"]+)"', flags=re.M)
SIZE_TOL, POS_TOL = 1e-6, 1e-4


def vec(s: str) -> list:
    return [float(x) for x in s.split(",") if x.strip()]


def parse(path: Path) -> tuple:
    usda, built, cur, kind, importer = {}, {}, None, None, None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = USDA.match(line)
        if m:
            cur, kind = m.group(1), "usda"
            usda[cur] = []
            continue
        m = BUILT.match(line)
        if m:
            cur, kind = m.group(1), "built"
            built[cur] = {}
            continue
        if line.startswith("导入器 "):
            importer, kind = line, None
            continue
        if kind == "usda":
            usda[cur].append(line)
        elif kind == "built":
            m = SHAPE.match(line)
            if m:
                built[cur][m.group(1)] = (m.group(2), vec(m.group(3)), vec(m.group(4)))
    return {k: "\n".join(v) for k, v in usda.items()}, built, importer


def cubes(text: str) -> dict:
    """{prim 名: (size, scale, translate)}；遇到 Cube 以外的形状就报错。"""
    heads = list(DEF.finditer(text))
    out = {}
    for i, m in enumerate(heads):
        if m.group(1) == "Xform":
            continue
        if m.group(1) != "Cube":
            raise ValueError(f"不认识的形状 {m.group(1)} {m.group(2)}")
        body = text[m.end():heads[i + 1].start() if i + 1 < len(heads) else len(text)]
        size = float(re.search(r"double size = ([\d.eE+-]+)", body).group(1))
        sc = re.search(r"xformOp:scale = \(([^)]*)\)", body)
        tr = re.search(r"xformOp:translate = \(([^)]*)\)", body)
        out[m.group(2)] = (size, vec(sc.group(1)) if sc else [1.0, 1.0, 1.0], vec(tr.group(1)) if tr else [0.0, 0.0, 0.0])
    return out


def main(argv=None) -> int:
    a = sys.argv[1:] if argv is None else argv
    if len(a) != 1:
        raise SystemExit("用法：support_slab_check.py <记录>")
    usda, built, importer = parse(Path(a[0]))
    if not usda:
        raise SystemExit(f"support_slab_check: {a[0]} 里没有 “## usda” 段")
    print(importer or "导入器：记录里没有")
    print("文件\t形状\t期望半尺寸\t建出的半尺寸\t期望位置\t建出的位置\t一致")
    bad = []
    for f in sorted(set(usda) | set(built)):
        if f not in usda or f not in built:
            bad.append(f"{f}（只在{'usda' if f in usda else '导入器输出'}里）")
            continue
        try:
            want = cubes(usda[f])
        except (ValueError, AttributeError) as e:
            bad.append(f"{f}（{e}）")
            continue
        got = built[f]
        if set(want) != set(got):
            bad.append(f"{f}（形状名字不同：usda {sorted(want)}，导入器 {sorted(got)}）")
            continue
        for name, (size, scale, tr) in sorted(want.items()):
            typ, half, pos = got[name]
            exp = [size * abs(s) / 2 for s in scale]
            ok = (typ == "BOX" and len(half) == 3 and len(pos) == 3
                  and all(abs(x - y) <= SIZE_TOL for x, y in zip(exp, half))
                  and all(abs(x - y) <= POS_TOL for x, y in zip(tr, pos)))
            if not ok:
                bad.append(f"{f} {name}")
            print("\t".join([f, name, str(tuple(round(x, 6) for x in exp)), str(tuple(half)), str(tuple(tr)), str(tuple(pos)),
                             "yes" if ok else "NO"]))
    n = len(set(usda) | set(built))
    if bad:
        print(f"SLAB_BAD {len(bad)} 处：{'；'.join(bad)}")
        return 1
    print(f"SLAB_OK {n}/{n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
