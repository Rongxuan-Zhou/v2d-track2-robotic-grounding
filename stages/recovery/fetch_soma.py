import pathlib
"""Download the pinned SOMA-X asset bundle without needing robotic_grounding.

setup_soma_assets.py only wraps soma.assets.get_assets_dir(revision=...);
that call is all we need, plus a copy into the repo's BODY_MODELS_DIR.
"""
import os, shutil, sys
from pathlib import Path
REV = "466879a83d57eabf3d875ded2d869f2075f90348"   # setup_soma_assets.py:47
DEST = Path(str(pathlib.Path(__file__).resolve().parents[2] / "third_party/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/body_models/soma"))
from soma.assets import get_assets_dir
src = get_assets_dir(revision=REV).resolve()
print("downloaded to:", src)
DEST.mkdir(parents=True, exist_ok=True)
for item in src.iterdir():
    tgt = DEST / item.name
    if tgt.exists(): continue
    (shutil.copytree if item.is_dir() else shutil.copy2)(item, tgt)
print("staged to   :", DEST)
for p in sorted(DEST.rglob("*")):
    if p.is_file(): print(f"   {p.relative_to(DEST)}  {p.stat().st_size/1e6:.1f} MB")
