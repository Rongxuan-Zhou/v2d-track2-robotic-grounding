"""SOMAWithFlex 的测试。在 robotic-grounding 镜像里跑（官方包只装在容器里）。

守：转换器契约中"soma_params.npz 必须携带并实际使用 bone_length_flexibles"一条。
契约表尚未落到 docs/contracts/（仓库处于 P0），暂无契约 ID，落表后补上。

判据：包装后的官方读取路径，与我们自己的 FK（libs/v2d_infra/soma_fk.py）逐帧一致；
不包装时与"不带 flex 的 FK"一致；两者之差是毫米级而非零，证明 flex 确实生效。
"""
import os
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "libs" / "v2d_infra"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pyarrow.parquet as pq
import torch
from paths import SOMA_ASSETS, TIER1, UPSTREAM
from soma_fk import col, fk, joint_names
from robotic_grounding.retarget.read_soma import SOMA
from soma_flex import FLEX_KEY, SOMAWithFlex, load_official_soma_to_g1

CPU = torch.device("cpu")


def _episode():
    f = sorted((TIER1 / "data" / "chunk-000").glob("*.parquet"))[0]
    return pq.read_table(f)


def _write_npz(t, path, with_flex=True, flex_override=None):
    T = t.num_rows
    d = dict(
        poses=col(t, "observation.human.pose").reshape(T, 77, 3).astype(np.float32),
        transl=col(t, "observation.human.translation").astype(np.float32),
        identity_coeffs=col(t, "observation.human.identity_coeffs").astype(np.float32),
        scale_params=col(t, "observation.human.scale_params").astype(np.float32),
        joint_names=np.array(joint_names()),
        identity_model_type=np.array("mhr"),
        unit=np.array("meters"),
        keep_root=np.array(False),
        joint_orient=np.zeros((78, 3, 3), np.float32),   # 必填但 _reconstruct 从不使用
    )
    if with_flex:
        d[FLEX_KEY] = (flex_override if flex_override is not None
                       else col(t, "observation.human.bone_length_flexibles").astype(np.float32))
    np.savez(path, **d)


def test_wrapped_matches_our_fk_and_flex_is_applied():
    t = _episode()
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "soma_params.npz")
        _write_npz(t, p)
        off = SOMA(data_root=SOMA_ASSETS, device=CPU).load_motion(p, normalize=False)["joints"]
        wrp = SOMAWithFlex(data_root=SOMA_ASSETS, device=CPU).load_motion(p, normalize=False)["joints"]
    ours_flex, ours_noflex = fk(t, flex=True), fk(t, flex=False)
    e_w = np.abs(wrp - ours_flex).max()
    e_o = np.abs(off - ours_noflex).max()
    gap = np.median(np.linalg.norm(wrp - off, axis=-1)) * 1000
    print(f"  包装版 vs 我们带flex的FK   最大差 {e_w*1000:.4f} mm")
    print(f"  官方版 vs 我们不带flex的FK 最大差 {e_o*1000:.4f} mm")
    print(f"  包装版 vs 官方版           中位差 {gap:.1f} mm")
    assert e_w < 1e-4, "包装后的官方路径与带 flex 的 FK 不一致"
    assert e_o < 1e-4, "官方路径与不带 flex 的 FK 不一致，基准本身有问题"
    assert gap > 5.0, "包装前后几乎无差，flex 没有生效"


def test_missing_flex_is_rejected():
    t = _episode()
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "soma_params.npz")
        _write_npz(t, p, with_flex=False)
        try:
            SOMAWithFlex(data_root=SOMA_ASSETS, device=CPU).load_motion(p)
        except ValueError as e:
            assert FLEX_KEY in str(e)
            print("  缺 flex 时报错:", str(e)[:60], "…")
            return
    raise AssertionError("缺 flex 的 npz 被静默接受了")


def test_wrong_length_flex_is_rejected():
    t = _episode()
    bad = np.zeros((t.num_rows - 1, 6), np.float32)
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "soma_params.npz")
        _write_npz(t, p, flex_override=bad)
        try:
            SOMAWithFlex(data_root=SOMA_ASSETS, device=CPU).load_motion(p)
        except ValueError:
            print("  帧数不符时报错 ✓")
            return
    raise AssertionError("帧数不符的 flex 被静默接受了")


def test_official_script_uses_wrapper():
    mod = load_official_soma_to_g1(UPSTREAM)
    assert mod.SOMA is SOMAWithFlex
    print("  官方 soma_to_g1 模块内的 SOMA 已替换为 SOMAWithFlex ✓")


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in tests:
        print(f"[{fn.__name__}]")
        fn()
    print(f"\n{len(tests)} 个测试全部通过")
