"""support_slab_check.py 的测试（守计划 agent-doc/plan-t3-hm-p2-2026-10-12.md 第 5 节“支撑面在训练导入器里”；暂无契约 ID）。只用标准库。
  1. 一份带 scale 的 Cube，导入器建出 size × |scale| / 2 的半尺寸、位置等于 translate：SLAB_OK 1/1
  2. 导入器按 v0.4.0 的旧行为建成半尺寸 (0.5, 0.5, 0.5)：SLAB_BAD，退出码 1；位置差 1 mm、类型不是 BOX、只在一边的文件：都算不一致
  3. usda 里有 Cylinder：报不认识，SLAB_BAD
  4. 真记录（若在）：results/t3hm/support_import_p185_mf4090.txt 的导入器输出换成旧行为 (0.5, 0.5, 0.5) 后，12 份都判不一致（只在内存里改，
     核的是这个检查抓得到 1 m 立方体，不拿真记录本身的结论当测试）
用法：python3 stages/policy/test_support_slab_check.py
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
USDA = """#usda 1.0
def Xform "support_surfaces"
{
    def Cube "support_0" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        double3 xformOp:translate = (-0.6119, 0.8796, 0.8610)
        double3 xformOp:scale = (0.0906, 0.0896, 0.02)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }
}
"""


def record(half="(0.0453, 0.0448, 0.01)", pos="(-0.6119, 0.8796, 0.861)", typ="BOX", usda=USDA, extra_built="") -> str:
    return ("# command  : ssh mf4090 …\n## usda a_support.usda\n" + usda +
            "导入器 /workspace/src/flash_chord/scene/support.py  md5 4d6eab112a442bb2b533dfb11bc7160c\n"
            f"== a_support.usda\n  support_0：类型 {typ}，shape_scale（Box 为半尺寸）{half}，位置 {pos}\n" + extra_built)


def run(text: str, rc: int = 0) -> str:
    with tempfile.TemporaryDirectory() as t:
        p = Path(t) / "r.txt"
        p.write_text(text, encoding="utf-8")
        r = subprocess.run([sys.executable, str(HERE / "support_slab_check.py"), str(p)], capture_output=True, text=True, timeout=60)
    assert r.returncode == rc, (r.returncode, r.stdout, r.stderr)
    return r.stdout + r.stderr


def main() -> int:
    out = run(record())
    assert "SLAB_OK 1/1" in out and "md5 4d6eab11" in out, out
    assert "SLAB_BAD 1 处：a_support.usda support_0" in run(record(half="(0.5, 0.5, 0.5)"), rc=1)
    assert "SLAB_BAD" in run(record(pos="(-0.6119, 0.8796, 0.862)"), rc=1)
    assert "SLAB_BAD" in run(record(typ="CYLINDER"), rc=1)
    out = run(record(extra_built="== b_support.usda\n  support_0：类型 BOX，shape_scale（Box 为半尺寸）(0.1, 0.1, 0.01)，位置 (0.0, 0.0, 0.5)\n"), rc=1)
    assert "b_support.usda（只在导入器输出里）" in out, out
    cyl = USDA.replace('def Cube "support_0"', 'def Cylinder "support_0"')
    assert "不认识的形状 Cylinder support_0" in run(record(usda=cyl), rc=1)
    real = REPO / "results/t3hm/support_import_p185_mf4090.txt"
    if real.exists():
        text = real.read_text(encoding="utf-8", errors="replace")
        old = re.sub(r"shape_scale（Box 为半尺寸）\([^)]*\)", "shape_scale（Box 为半尺寸）(0.5, 0.5, 0.5)", text)
        out = run(old, rc=1)
        assert "SLAB_BAD 12 处" in out, out
    print("ALL_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
