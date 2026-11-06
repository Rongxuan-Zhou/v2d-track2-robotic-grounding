"""R1 数值检查：support surface 与重定向后的物体是否在同一坐标系。

挑战声称 support 在 g1_retarget_world，而官方重定向的机器人世界系经过两次数据相关修正
（ground_z_offset、robot_delta_z），两者是否重合没人验证过。replay_motion.py 不加载 support，
所以回放通过不能说明 R1；训练环境会加载它（scene_config.py:455）。

判据：物体在它"静置于支撑面上"的帧里（支撑面 = support 几何体顶面，或仿真地面 z=0），网格最低点应落在 support 顶面上，
且水平投影落在 support 的 xy 范围内。选"静置帧"用物体高度最低的前若干帧的中位数，
不假设第 0 帧一定静置。容器内跑（需要 pxr）。
用法：scripts/dev/incontainer stages/retarget/check_support_alignment.py <robot_name=g1 分区目录>
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from pxr import Usd, UsdGeom

part = Path(sys.argv[1])
row = pq.read_table(part / "data.parquet").to_pylist()[0]
seq = part.parent.name.split("=", 1)[1]
# 与 scene_config._discover_support_surface 同一查找顺序：机器人专属的优先
stage_dir = part.parent.parent.parent / "reconstructed_stage"
robot = part.name.split("=", 1)[1]
support = next((c for c in (stage_dir / f"{seq}_{robot}_support.usda", stage_dir / f"{seq}_support.usda")
                if c.exists()), None)
if len(sys.argv) > 2:          # 显式指定某个 usda，用于对照
    support = Path(sys.argv[2])
assert support is not None, "找不到 support"

stage = Usd.Stage.Open(str(support))
cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
boxes = []
for prim in stage.Traverse():
    if prim.IsA(UsdGeom.Gprim):
        r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        boxes.append((prim.GetPath(), np.array(r.GetMin()), np.array(r.GetMax())))
print(f"support: {support.name}，{len(boxes)} 个几何体")
for p, lo, hi in boxes:
    print(f"  {str(p):<40} x[{lo[0]:+.3f},{hi[0]:+.3f}] y[{lo[1]:+.3f},{hi[1]:+.3f}] z[{lo[2]:+.3f},{hi[2]:+.3f}]")

# 物体网格（重定向时生成的 obj，物体局部系）
V = []
for line in open(Path(row["object_mesh_paths"][0]) if Path(row["object_mesh_paths"][0]).is_absolute()
                 else part.parent / "object" / "textured_mesh.obj"):
    if line.startswith("v "):
        V.append([float(x) for x in line.split()[1:4]])
V = np.array(V)
pos = np.array([b[0] for b in row["object_body_position"]])       # (T,3)
q = np.array([b[0] for b in row["object_body_wxyz"]])             # (T,4) wxyz
w, x, y, z = q.T
R = np.stack([np.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)], -1),
              np.stack([2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)], -1),
              np.stack([2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], -1)], -2)
W = np.einsum("tij,vj->tvi", R, V) + pos[:, None, :]              # (T,V,3)
bottom = W[:, :, 2].min(1)
rest = np.argsort(bottom)[: max(5, len(bottom) // 20)]            # 最低的 5% 帧当"静置帧"
cx, cy = W[rest][:, :, 0].mean(), W[rest][:, :, 1].mean()
b = float(np.median(bottom[rest]))
print(f"\n物体静置时：网格最低点 z = {b:+.4f} m，中心 xy = ({cx:+.3f}, {cy:+.3f})")

under = [(p, lo, hi) for p, lo, hi in boxes if lo[0] <= cx <= hi[0] and lo[1] <= cy <= hi[1]]
cands = [("地面 z=0", 0.0)] + [(str(p), float(hi[2])) for p, _, hi in under]
name, top = min(cands, key=lambda c: abs(b - c[1]))       # 物体实际落在离它最近的那个面上
gap = b - top
print(f"候选支撑面: " + ", ".join(f"{n}@{z:+.3f}" for n, z in cands))
print(f"最近支撑面 {name} 顶面 z = {top:+.4f} m  →  间隙 = {gap*1000:+.1f} mm（正=悬空，负=穿入）")
ok = abs(gap) < 0.02
print("判定:", "对齐 ✓" if ok else "未对齐 ✗（|间隙| ≥ 20 mm）")
print(f"RESULT gap_mm={gap*1000:.1f} surface={'floor' if name.startswith('地面') else 'support'} ok={int(ok)}")
sys.exit(0 if ok else 1)
