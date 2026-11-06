"""make_diag_support.py 的行为测试（暂无契约 ID：运行规范待成文）。只用标准库。

在临时目录里造一份 ep7 那样的桌面方盒（translate (-0.7757, 0.1698, 0.7771)、scale (0.84, 0.282, 0.01)、size 1）：
  1. 保留 +z 面加厚到 4 cm：顶面 0.7821 不变，中心 z 变成 0.7621，scale z 变成 0.04；parquet 复制、object 是能解析的相对链接；
  2. 再跑一次：已存在，拒绝覆盖；
  3. 原目录有机器人专属的 <SEQ>_g1_support.usda：拒绝。
用法：python3 stages/policy/test_make_diag_support.py
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "make_diag_support.py"
SEQ = "2026-05-29_15-20-43_iron_round_table_02"
USDA = '''#usda 1.0
(
    defaultPrim = "support_surfaces"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "support_surfaces"
{
    def Cube "tabletop" (
        prepend apiSchemas = ["PhysicsCollisionAPI"]
    )
    {
        double size = 1
        double3 xformOp:translate = (-0.7757, 0.1698, 0.7771)
        double3 xformOp:scale = (0.84, 0.282, 0.01)
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }
}
'''


def tree(root: Path) -> None:
    wb = root / "src/tier1/whole_body"
    seq = wb / "soma" / f"sequence_id={SEQ}"
    (seq / "robot_name=g1").mkdir(parents=True)
    (seq / "robot_name=g1/data.parquet").write_bytes(b"PAR1 dummy")
    (seq / "object").mkdir()
    (seq / "object/textured_mesh.obj").write_text("v 0 0 0\n")
    (wb / "reconstructed_stage").mkdir()
    (wb / "reconstructed_stage" / f"{SEQ}_support.usda").write_text(USDA)


def run(root: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), "src", "dst", SEQ, "--thicken", "tabletop", "0.04", "+z"],
                          cwd=root, capture_output=True, text=True, timeout=60)


def main() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        tree(root)
        r = run(root)
        assert r.returncode == 0, r.stderr
        out = (root / "dst/tier1/whole_body/reconstructed_stage" / f"{SEQ}_support.usda").read_text()
        tz = float(re.search(r"xformOp:translate = \(([^)]*)\)", out).group(1).split(",")[2])
        sz = float(re.search(r"xformOp:scale = \(([^)]*)\)", out).group(1).split(",")[2])
        assert abs(tz - 0.7621) < 1e-9 and abs(sz - 0.04) < 1e-9, (tz, sz)
        assert abs((tz + 0.5 * sz) - 0.7821) < 1e-9
        seq = root / "dst/tier1/whole_body/soma" / f"sequence_id={SEQ}"
        assert (seq / "robot_name=g1/data.parquet").read_bytes() == b"PAR1 dummy"
        assert (seq / "object").is_symlink() and not Path((seq / "object").readlink()).is_absolute()
        assert (seq / "object/textured_mesh.obj").exists()
        print("ok thicken")
        r2 = run(root)
        assert r2.returncode != 0 and "不覆盖" in (r2.stderr + r2.stdout), r2
        print("ok no overwrite")
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        tree(root)
        (root / "src/tier1/whole_body/reconstructed_stage" / f"{SEQ}_g1_support.usda").write_text(USDA)
        r3 = run(root)
        assert r3.returncode != 0 and "机器人专属" in (r3.stderr + r3.stdout), r3
        print("ok robot-specific refused")
    print("ALL_OK")


if __name__ == "__main__":
    main()
