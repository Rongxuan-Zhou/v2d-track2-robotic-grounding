"""参考轨迹里物体网格与支撑面、地面的穿透（Mac 上用 uv 带 numpy、pyarrow 跑；只读下载到本地的数据，Tier 1）。

每条序列一个目录，放三个文件：data.parquet（重定向后的 motion_v1，取 object_body_position、object_body_wxyz 的第一个刚体）、
textured_mesh.obj（物体网格，物体局部系）、<SEQ>_support.usda（支撑面）。支撑面按文本读，不用 pxr：
Cube 默认按标注尺寸 size × xformOp:scale 算（上游 8c778f9 / #185 之后的导入方式）；加 --cube-as-v040 则按 v0.4.0 实际建出来的
碰撞体算：只读 size、忽略 scale，三边都是 size（results/ml40/diverge/support_import_v040.txt 实测为 1 m 立方体）。
Cylinder = radius / height、axis "Z"，两种导入方式一样。都只有 xformOp:translate。地面是 z = 0（重定向时已把地面找平到 0）。

逐帧把网格顶点（均匀抽约 4000 个）变到世界系，算：
  地面：最低顶点的高度，< 0 为穿地；
  每个支撑形状：顶点到形状的最小有符号距离（形状内部为负，即嵌入深度）；
  方盒另算“穿过薄板”：顶点落在板的两条长边张成的投影范围内，却跑到板的另一侧（相对这一帧物体中心所在的一侧）。
报：嵌入或穿地超过 5 mm 的帧数、最深值与所在帧、第 0 帧的值、离形状 1 cm 以内的帧数（有接触机会的帧）。
用法：uv run --no-project --with numpy --with pyarrow python stages/retarget/support_penetration.py [--cube-as-v040] <序列目录>...
计划 agent-doc/plan-ep1-ep7-divergence-2026-09-30.md 的 C5。
"""
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

THRESH = 0.005   # 5 mm
NEAR = 0.01      # 1 cm
MAX_VERTS = 4000


def load_vertices(obj: Path) -> tuple[np.ndarray, int]:
    verts = []
    with obj.open() as f:
        for line in f:
            if line.startswith("v "):
                verts.append([float(x) for x in line.split()[1:4]])
    v = np.asarray(verts, dtype=np.float64)
    stride = max(1, len(v) // MAX_VERTS)
    return v[::stride], len(v)


def load_poses(parquet: Path) -> tuple[np.ndarray, np.ndarray]:
    row = pq.read_table(parquet).slice(0, 1).to_pylist()[0]
    pos = np.asarray([b[0] for b in row["object_body_position"]], dtype=np.float64)
    wxyz = np.asarray([b[0] for b in row["object_body_wxyz"]], dtype=np.float64)
    return pos, wxyz


def rotations(wxyz: np.ndarray) -> np.ndarray:
    q = wxyz / np.linalg.norm(wxyz, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def load_supports(usda: Path, cube_as_v040: bool = False) -> list[dict]:
    text = usda.read_text()
    shapes = []
    for m in re.finditer(r'def (Cube|Cylinder) "([^"]+)"[^{]*\{(.*?)\n    \}', text, re.S):
        kind, name, body = m.groups()
        t = [float(x) for x in re.search(r"xformOp:translate = \(([^)]*)\)", body).group(1).split(",")]
        if kind == "Cube":
            size = float(re.search(r"double size = ([0-9.eE+-]+)", body).group(1))
            s = re.search(r"xformOp:scale = \(([^)]*)\)", body)
            scale = [float(x) for x in s.group(1).split(",")] if s else [1.0, 1.0, 1.0]
            half = 0.5 * size * (np.ones(3) if cube_as_v040 else np.abs(np.asarray(scale)))
            shapes.append({"kind": kind, "name": name, "center": np.asarray(t), "half": half})
        else:
            axis = re.search(r'axis = "([XYZ])"', body)
            if axis and axis.group(1) != "Z":
                raise SystemExit(f"{usda.name}：{name} 的轴是 {axis.group(1)}，脚本只处理 Z")
            r = float(re.search(r"double radius = ([0-9.eE+-]+)", body).group(1))
            h = float(re.search(r"double height = ([0-9.eE+-]+)", body).group(1))
            shapes.append({"kind": kind, "name": name, "center": np.asarray(t), "radius": r, "half_height": 0.5 * h})
    return shapes


def box_sdf(p: np.ndarray, c: np.ndarray, h: np.ndarray) -> np.ndarray:
    q = np.abs(p - c) - h
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(q.max(axis=-1), 0.0)
    return outside + inside


def cylinder_sdf(p: np.ndarray, c: np.ndarray, r: float, hh: float) -> np.ndarray:
    d = np.stack([np.linalg.norm(p[..., :2] - c[:2], axis=-1) - r, np.abs(p[..., 2] - c[2]) - hh], -1)
    return np.linalg.norm(np.maximum(d, 0.0), axis=-1) + np.minimum(d.max(axis=-1), 0.0)


def through_slab(p: np.ndarray, centroid: np.ndarray, c: np.ndarray, h: np.ndarray) -> np.ndarray:
    """每帧：落在板的长边投影范围内、却在板另一侧（越过远面）的顶点个数。"""
    a = int(np.argmin(h))
    others = [i for i in range(3) if i != a]
    in_footprint = np.all(np.abs(p[..., others] - c[others]) <= h[others], axis=-1)
    side = np.sign(centroid[:, a] - c[a])[:, None]            # 物体中心在哪一侧
    beyond = side * (p[..., a] - c[a]) < -h[a]                 # 越过远面
    return (in_footprint & beyond).sum(axis=-1)


def fmt_mm(x: float) -> str:
    return f"{x * 1000:+.1f} mm"


def analyse(d: Path, cube_as_v040: bool = False) -> None:
    usda = next(d.glob("*_support.usda"))
    seq = usda.name[: -len("_support.usda")]
    verts, n_all = load_vertices(d / "textured_mesh.obj")
    pos, wxyz = load_poses(d / "data.parquet")
    rot = rotations(wxyz)
    world = np.einsum("tij,nj->tni", rot, verts) + pos[:, None, :]   # (T, n, 3)
    centroid = world.mean(axis=1)
    T = len(pos)
    print(f"== {seq}（{T} 帧；网格 {n_all} 个顶点，抽 {len(verts)} 个；方盒按{'v0.4.0 实际导入（只读 size）' if cube_as_v040 else '标注尺寸（size × scale）'}）")
    zmin = world[..., 2].min(axis=1)
    k = int(np.argmin(zmin))
    print(f"  地面：最低顶点最低 {fmt_mm(zmin[k])}（第 {k} 帧）；穿地超过 5 mm 的帧 {int((zmin < -THRESH).sum())} 个；第 0 帧 {fmt_mm(zmin[0])}")
    for s in load_supports(usda, cube_as_v040):
        if s["kind"] == "Cube":
            sd = box_sdf(world, s["center"], s["half"]).min(axis=1)
            thr = through_slab(world, centroid, s["center"], s["half"])
            dims = "×".join(f"{2 * x * 100:.1f}" for x in s["half"])
            desc = f"Cube \"{s['name']}\"（{dims} cm，中心 {np.round(s['center'], 3).tolist()}）"
        else:
            sd = cylinder_sdf(world, s["center"], s["radius"], s["half_height"]).min(axis=1)
            thr = None
            desc = f"Cylinder \"{s['name']}\"（半径 {s['radius'] * 100:.1f} cm、高 {2 * s['half_height'] * 100:.1f} cm，中心 {np.round(s['center'], 3).tolist()}）"
        k = int(np.argmin(sd))
        line = (f"  {desc}：最小有符号距离 {fmt_mm(sd[k])}（第 {k} 帧）；嵌入超过 5 mm 的帧 {int((sd < -THRESH).sum())} 个；"
                f"离形状 1 cm 以内的帧 {int((sd < NEAR).sum())} 个；第 0 帧 {fmt_mm(sd[0])}")
        if thr is not None:
            kt = int(np.argmax(thr))
            line += f"；穿过薄板的帧 {int((thr > 0).sum())} 个（最多 {int(thr[kt])} 个顶点，第 {kt} 帧）"
        print(line)


def selftest() -> None:
    """几何函数的符号与数值：盒中心、盒外、圆柱顶面上方、穿过薄板。"""
    c, h = np.zeros(3), np.array([0.01, 0.5, 0.5])            # 2 cm 厚、沿 x 的薄板
    assert np.isclose(box_sdf(np.zeros((1, 1, 3)), c, h)[0, 0], -0.01)
    assert np.isclose(box_sdf(np.array([[[0.03, 0.0, 0.0]]]), c, h)[0, 0], 0.02)
    cc = np.array([0.0, 0.0, 0.7])
    assert np.isclose(cylinder_sdf(np.array([[[0.0, 0.0, 0.73]]]), cc, 0.1, 0.02)[0, 0], 0.01)
    assert np.isclose(cylinder_sdf(np.array([[[0.0, 0.0, 0.715]]]), cc, 0.1, 0.02)[0, 0], -0.005)
    pts = np.array([[[0.05, 0.0, 0.0], [-0.03, 0.1, 0.1], [-0.03, 0.9, 0.0]]])  # 中心在 +x 侧；第 2 点穿过、第 3 点在投影外
    assert through_slab(pts, np.array([[0.02, 0.0, 0.0]]), c, h)[0] == 1
    print("SELFTEST_OK")


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("用法：support_penetration.py [--cube-as-v040] <序列目录>... | --selftest")
    if sys.argv[1] == "--selftest":
        selftest()
        return
    args = sys.argv[1:]
    v040 = "--cube-as-v040" in args
    for d in (a for a in args if a != "--cube-as-v040"):
        analyse(Path(d), v040)


if __name__ == "__main__":
    main()
