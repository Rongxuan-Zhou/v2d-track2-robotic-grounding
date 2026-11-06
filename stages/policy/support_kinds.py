"""列出一个目录里每个 <序列>_support.usda 的支撑面形状（只用标准库；训练机上经 scripts/dev/pipe-py 跑，不需要同步仓库）。

每个文件一行（TSV）：序列名、形状清单、有没有带 xformOp:scale 的 Cube、文件 md5。
  形状清单：support_surfaces 下各个 prim 的类型与个数，如 Cylinder×1、Cube×2；没有子 prim（空 Xform）记 empty。
  带 scale 的 Cube：v0.4.0 与 kit 指定的评测运行时 aee95f2 的 scene/support.py 对 Cube 只读 size、不读 xformOp:scale，
  会把这种方盒建成边长 1 m 的实心立方体（发散排查，agent-doc/experiments/2026-09-30-divergence.md）。
只读给定目录，不碰其他档位的文件。Tier 2 批量用它决定哪些序列只有地面（训练加 scene.support=false）、哪些是方盒支撑面（先不排）。
用法：python3 stages/policy/support_kinds.py <目录>
  落盘：scripts/dev/record t2/support_kinds_tier2_mf4090 scripts/dev/pipe-py stages/policy/support_kinds.py runs/motion/tier2/whole_body/reconstructed_stage
"""
import collections
import hashlib
import re
import sys
from pathlib import Path

DEF = re.compile(r'^\s*def\s+(\w+)\s+"([^"]+)"', flags=re.M)


def describe(text: str) -> tuple:
    """(形状清单, 有没有带 scale 的 Cube)。每个 def 块取到下一个 def 之前的文字。"""
    heads = list(DEF.finditer(text))
    counts = collections.Counter()
    scaled_cube = False
    for i, m in enumerate(heads):
        kind = m.group(1)
        if kind == "Xform":
            continue
        counts[kind] += 1
        body = text[m.end():heads[i + 1].start() if i + 1 < len(heads) else len(text)]
        if kind == "Cube" and "xformOp:scale" in body:
            scaled_cube = True
    shapes = "、".join(f"{k}×{v}" for k, v in sorted(counts.items())) or "empty"
    return shapes, scaled_cube


def main() -> int:
    d = Path(sys.argv[1])
    files = sorted(d.glob("*_support.usda"))
    if not files:
        print(f"{d} 里没有 *_support.usda")
        return 1
    print("sequence_id\tshapes\tscaled_cube\tmd5")
    tally = collections.Counter()
    for f in files:
        raw = f.read_bytes()
        shapes, scaled = describe(raw.decode("utf-8", errors="replace"))
        seq = f.name[: -len("_support.usda")]
        tally["empty" if shapes == "empty" else ("scaled_cube" if scaled else "other")] += 1
        print(f"{seq}\t{shapes}\t{'yes' if scaled else 'no'}\t{hashlib.md5(raw).hexdigest()}")
    print(f"# 共 {len(files)} 个：只有地面 {tally['empty']}，带 scale 的方盒 {tally['scaled_cube']}，其他 {tally['other']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
