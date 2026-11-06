"""用当前挂载的 flash_chord 导入器真正建一次支撑面，打印每个形状的类型、半尺寸与位置（容器内、只用 CPU）。

v0.4.0 的 scene/support.py 对 Cube 只读 USD `size`、不读 `xformOp:scale`（support.py:85–86、114–116），上游 8c778f9（#185）才改成
`size × |scale|`。这里不靠读代码，直接用 /workspace 上挂载的那份导入器建到一个空的 newton.ModelBuilder 里，读回 shape_scale：
Box 的 shape_scale 是三个半尺寸，Cylinder 是（半径，半高，…）。计划 agent-doc/plan-ep1-ep7-divergence-2026-09-30.md 附录 A（第 4 版）的 P1。
用法（容器内）：python /repo/stages/policy/support_import_check.py <support.usda>...
  宿主机上：scripts/dev/in-flash-chord "python /repo/stages/policy/support_import_check.py /repo/runs/motion/tier1/whole_body/reconstructed_stage/<SEQ>_support.usda"
"""
import hashlib
import sys
from pathlib import Path

import newton
from flash_chord.scene import support as sp


def main() -> None:
    src = Path(sp.__file__)
    print(f"导入器 {src}  md5 {hashlib.md5(src.read_bytes()).hexdigest()}")
    for usda in sys.argv[1:]:
        builder = newton.ModelBuilder()
        sp.add_support_surfaces(builder, usda)
        print(f"== {Path(usda).name}")
        for i in range(builder.shape_count):
            kind = builder.shape_type[i]
            try:
                kind = newton.GeoType(kind).name
            except (ValueError, TypeError, AttributeError):
                pass
            scale = tuple(round(float(v), 6) for v in builder.shape_scale[i])
            pos = tuple(round(float(v), 4) for v in builder.shape_transform[i][:3])
            print(f"  {builder.shape_label[i]}：类型 {kind}，shape_scale（Box 为半尺寸）{scale}，位置 {pos}")


if __name__ == "__main__":
    main()
