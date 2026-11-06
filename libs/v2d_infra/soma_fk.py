"""SOMA-X 正运动学的唯一入口。

存在的理由：FK 曾在 recovery 的八个脚本里各抄一份，两类错误都是这样混进来的：
  1. 关节索引错位——SOMAPoseOutput.joints 已不含 Root，某份抄本又切了一次 [:,1:]；
  2. 漏传 observation.human.bone_length_flexibles——它改 spine/neck/shoulder/arm/
     hip/leg 六处骨长（soma/identity_model.py:331-341），漏掉会让关节整体偏 13.5 mm。
所以 flex 设成必填关键字：每个调用方都得显式表态，不能再静默丢掉。

坐标：返回 SOMA 世界系。挑战的物体位姿与地面平面在 OpenCV 世界系，两者差
diag(1,-1,-1)，用 R_CV 转换。
"""
import hashlib
import os
from pathlib import Path

import numpy as np
import torch

from paths import RUNS, SOMA_ASSETS

R_CV = np.diag([1.0, -1.0, -1.0])
_layer = None


def layer():
    """惰性构造，避免 import 本模块就付出加载 SOMA 资产的代价。"""
    global _layer
    if _layer is None:
        from soma import SOMALayer
        _layer = SOMALayer(data_root=str(SOMA_ASSETS), identity_model_type="mhr",
                           device="cpu", enable_procedural_transforms=False)
    return _layer


def joint_names() -> list[str]:
    """77 个关节名，与挑战数据的关节顺序逐字节一致（rig 的 78 个去掉 Root）。"""
    return [str(n) for n in layer().rig_data["joint_names"]][1:]


def col(t, c):
    v = t.column(c).to_numpy(zero_copy_only=False)
    return np.stack(v) if v.dtype == object else np.asarray(v)


_INPUT_COLS = ("observation.human.pose", "observation.human.translation",
               "observation.human.identity_coeffs", "observation.human.scale_params",
               "observation.human.bone_length_flexibles")
CACHE_DIR = RUNS / "cache" / "fk"
# 本文件的 md5 进缓存键：FK 逻辑一改，旧缓存自动失效，不会"看起来复现了"其实用的旧 FK。
_SELF_MD5 = hashlib.md5(Path(__file__).read_bytes()).hexdigest()


def _cache_key(table, flex: bool) -> str:
    h = hashlib.sha1(f"{_SELF_MD5}|flex={int(flex)}".encode())
    for c in _INPUT_COLS:
        h.update(np.ascontiguousarray(col(table, c)).tobytes())
    return h.hexdigest()


def fk(table, *, flex: bool) -> np.ndarray:
    """带磁盘缓存的 FK。缓存在 runs/cache/fk/，按输入内容寻址；V2D_FK_CACHE=0 关闭。

    存在的理由：每轮分析 10 个脚本各自对 30 条序列重算 FK，一批 20 分钟几乎全耗在这里。
    """
    if os.environ.get("V2D_FK_CACHE", "1") == "0":
        return _fk_compute(table, flex=flex)
    path = CACHE_DIR / f"{_cache_key(table, flex)}.npy"
    if path.exists():
        return np.load(path)
    j = _fk_compute(table, flex=flex)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp.npy")   # 先写临时文件再 rename：并行脚本同键写入也安全
    np.save(tmp, j)
    os.replace(tmp, path)
    return j


def _fk_compute(table, *, flex: bool) -> np.ndarray:
    """一个 episode 的 parquet 表 -> (T, 77, 3) SOMA 世界系关节位置，单位米。

    flex=True 时把 bone_length_flexibles 逐帧注入；它在序列内逐帧变化，
    不能只取第 0 帧。identity/scale 实测序列内恒定，逐帧传入与取第 0 帧等价。
    apply_correctives=False 与官方 read_soma.py:517 一致：correctives 只动
    rest-shape 顶点，不影响关节位置。
    """
    p = col(table, "observation.human.pose")
    T = len(p)
    kw = None
    if flex:
        b = col(table, "observation.human.bone_length_flexibles")
        kw = {"bone_length_flexibles": torch.tensor(b, dtype=torch.float32)}
    out = layer()(
        torch.tensor(p.reshape(T, 77, 3), dtype=torch.float32),
        torch.tensor(col(table, "observation.human.identity_coeffs"), dtype=torch.float32),
        scale_params=torch.tensor(col(table, "observation.human.scale_params"), dtype=torch.float32),
        transl=torch.tensor(col(table, "observation.human.translation"), dtype=torch.float32),
        apply_correctives=False,
        kwargs=kw,
    )
    j = out[0] if isinstance(out, (tuple, list)) else out
    if hasattr(j, "joints"):
        j = j.joints
    j = np.asarray(j.detach())
    assert j.shape == (T, 77, 3), f"FK 输出 {j.shape}，期望 ({T}, 77, 3)"
    return j
