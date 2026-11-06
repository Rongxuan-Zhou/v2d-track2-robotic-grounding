"""开训前核对训练输入（在 FlashCHORD 容器里跑）：parquet 能读、帧数与 fps、里面引用的资产路径在容器内都存在、
support usda 存在。用法：python check_inputs.py <parquet>...（容器内路径）。有缺失时退出码 1。"""
import hashlib
import os
import sys
from pathlib import Path

import pyarrow.parquet as pq

bad = 0
for p in sys.argv[1:]:
    t = pq.read_table(p)
    print(f"== {p}\n  行数 {t.num_rows}，列 {len(t.column_names)}，sha256 {hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]}…")
    for c in ("fps", "frequency"):
        if c in t.column_names:
            print(f"  {c} = {sorted(set(t.column(c).to_pylist()))[:3]}")
    for c in t.column_names:
        if c.startswith("object_") and ("path" in c or "name" in c):
            print(f"  列 {c} = {t.column(c).to_pylist()[0]!r}")
    # 物体资产按加载器的同一规则解析（flash_chord.data.motion_v1 → resolve_asset_path(reference_path=parquet)）
    from flash_chord.assets.registry import resolve_asset_path
    for c in ("object_mesh_paths", "object_urdf_paths"):
        for x in (t.column(c).to_pylist()[0] if c in t.column_names else []) or []:
            q = resolve_asset_path(str(x), reference_path=p)
            ok = os.path.exists(q)
            bad += not ok
            print(f"  {'OK ' if ok else '缺失'} {c}: {q}")
    root = Path(p).parents[3]  # .../whole_body/soma/sequence_id=<seq>/robot_name=g1/data.parquet → whole_body
    seq = Path(p).parents[1].name.split("=", 1)[1]
    sup = [root / "reconstructed_stage" / f"{seq}{s}_support.usda" for s in ("_g1", "")]
    found = [s for s in sup if s.exists()]
    print(f"  support：{found[0] if found else '无（只有地面）'}")
sys.exit(1 if bad else 0)
