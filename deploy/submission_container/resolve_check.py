"""在提交容器里用评测器自己的函数，逐个 checkpoint 解析评测要读的文件，确认都在镜像里（不需要 GPU）。

和 motion_closure.py 互为对照：那边按规则列文件，这边调 FlashCHORD 8c778f9 的 load_reference（内部解析物体资产）
和 support_usda_for_reference，与 scene/setup.py 的 setup_scene 走同一段代码，只是不建场景。
解析出的每个路径都要是镜像里存在的文件，并且落在 /repo 或 /workspace 下。

用法（容器内）：/opt/venv/bin/python3 resolve_check.py /submission/weights
输出：每个 checkpoint 一行（ok / FAIL、parquet、支撑面、资产文件数），最后一行 RESOLVE_OK <数> 或 RESOLVE_FAILED <数>。
"""
import dataclasses
import json
import struct
import sys
from pathlib import Path

from flash_chord.assets.registry import support_usda_for_reference
from flash_chord.data import load_reference


# 只作来历记录、评测时不读的元数据字段：raw_motion_file 是生成 parquet 的源文件（motion_src/.../soma_params.npz），
# source_path 就是 parquet 本身。11/05 mf4090 实测：测试包里没有 motion_src，三集照样跑完。
SKIP_FIELDS = {"raw_motion_file", "source_path"}


def resolved_config(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n))["__metadata__"]
    return json.loads(meta["user.resolved_config_json"])


def asset_paths(obj, seen=None):
    """从 reference 里递归找出所有像文件路径的字符串（*_path 字段与路径列表）。"""
    seen = set() if seen is None else seen
    if id(obj) in seen:
        return []
    seen.add(id(obj))
    out = []
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            if f.name in SKIP_FIELDS:
                continue
            out += asset_paths(getattr(obj, f.name), seen)
    elif isinstance(obj, (list, tuple)):
        for x in obj:
            out += asset_paths(x, seen)
    elif isinstance(obj, (str, Path)) and str(obj).startswith("/") and Path(str(obj)).suffix:
        out.append(str(obj))
    return out


def main():
    weights = Path(sys.argv[1])
    bad = 0
    ckpts = sorted(weights.glob("tier*/*.safetensors"))
    for ckpt in ckpts:
        cfg = resolved_config(ckpt)
        parquet = cfg["task"]["parquet"]
        try:
            ref = load_reference(parquet, control_fps=None, motion_speed=cfg["task"].get("motion_speed", 1.0))
            support = support_usda_for_reference(ref) if cfg.get("scene", {}).get("support", True) else None
            if cfg.get("scene", {}).get("support", True) and support is None:
                raise FileNotFoundError("no support asset")
            paths = sorted(set(asset_paths(ref)) | ({str(support)} if support else set()))
            missing = [p for p in paths if not Path(p).is_file()]
            outside = [p for p in paths if not p.startswith(("/repo/", "/workspace/"))]
            if missing or outside:
                raise FileNotFoundError(f"missing {missing[:3]} outside {outside[:3]}")
            print(f"ok\t{ckpt.relative_to(weights)}\t{support}\t{len(paths)}", flush=True)
        except Exception as e:  # 逐个报，不在第一个失败处停
            bad += 1
            print(f"FAIL\t{ckpt.relative_to(weights)}\t{parquet}\t{type(e).__name__}: {e}", flush=True)
    print(f"RESOLVE_FAILED {bad}" if bad else f"RESOLVE_OK {len(ckpts)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
