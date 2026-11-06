"""包一层官方 read_soma.SOMA，让它把 bone_length_flexibles 传给 SOMALayer。

为什么要包：挑战数据给了逐帧的 bone_length_flexibles (T,6)，SOMA 用它改
spine/neck/shoulder/arm/hip/leg 六处骨长（soma/identity_model.py:331-341）。
官方 read_soma 的 _load_npz 不要求它、_reconstruct 调 layer 时也不传它
（read_soma.py:507-518），于是重定向看到的人体整体偏 13.5 mm、手偏 14.4 mm
（results/bone_flex_probe.txt）。官方代码不改源码（CLAUDE.md），所以在这里包。

做法：_load_npz 取出 flex；_reconstruct 期间把 self.layer 临时换成一个代理，
代理在每次调用时补上 kwargs={"bone_length_flexibles": flex}，其余属性原样转发。
SOMA 是普通类（非 nn.Module），换属性是安全的。

npz 里缺 flex 时直接报错，不回退到官方行为——静默回退正是这个 bug 的来源。
上游版本：video_to_data 仓 read_soma.py，_reconstruct 位于 472 行、
layer 调用位于 507-518 行。上游改动这两处时要重看本包装。
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from robotic_grounding.retarget.read_soma import SOMA

FLEX_KEY = "bone_length_flexibles"


class _FlexLayer:
    """SOMALayer 的代理：调用时注入 flex，其余属性转发给真 layer。"""

    def __init__(self, inner, flex: torch.Tensor):
        self._inner = inner
        self._flex = flex

    def __call__(self, poses, *args, **kw):
        if kw.get("kwargs") is not None:
            raise RuntimeError("上游已开始自己传 kwargs，本包装需要重看")
        if poses.shape[0] != self._flex.shape[0]:
            raise ValueError(f"flex 有 {self._flex.shape[0]} 帧，poses 有 {poses.shape[0]} 帧")
        kw["kwargs"] = {FLEX_KEY: self._flex}
        return self._inner(poses, *args, **kw)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class SOMAWithFlex(SOMA):
    def _load_npz(self, params_path):
        params = super()._load_npz(params_path)
        if FLEX_KEY not in params:
            raise ValueError(f"{params_path} 缺 {FLEX_KEY}；"
                             "挑战数据的转换器必须写入它，见 stages/retarget/soma_flex.py")
        flex = np.asarray(params[FLEX_KEY], dtype=np.float32)
        T = params["poses"].shape[0]
        if flex.shape != (T, 6):
            raise ValueError(f"{FLEX_KEY} 形状 {flex.shape}，期望 ({T}, 6)")
        self._flex = flex
        return params

    def _reconstruct(self, **kw):
        inner = self.layer
        self.layer = _FlexLayer(inner, torch.from_numpy(self._flex).to(self.device))
        try:
            return super()._reconstruct(**kw)
        finally:
            self.layer = inner


def load_official_soma_to_g1(upstream_root: Path):
    """按文件路径加载官方 scripts/retarget/soma_to_g1.py，并把其中的 SOMA 换成 SOMAWithFlex。

    soma_to_g1.main() 在函数体里才按全局名查 SOMA（soma_to_g1.py:407），
    所以替换模块属性即可生效。调用方设好 sys.argv 后执行 mod.main()。
    """
    path = Path(upstream_root) / "robotic_grounding/scripts/retarget/soma_to_g1.py"
    spec = importlib.util.spec_from_file_location("soma_to_g1_official", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    if getattr(mod, "SOMA", None) is not SOMA:
        raise RuntimeError("上游 soma_to_g1 不再按名引用 read_soma.SOMA，本包装需要重看")
    mod.SOMA = SOMAWithFlex
    return mod
