"""Tier 2 降噪改进的测试（计划 agent-doc/plan-tier2-denoise-2026-10-04.md 第 2 版第 0、4、5 节；守档位隔离与 v5.4 不变；契约表未落，暂无契约 ID）。
  1. t2eval.py 只 import 允许的模块，字符串里没有 Tier 1 或 v5.2 / v5.3 产物的路径（只用标准库，Mac 上也能跑）
  2. static_mask 的 cover（候选 A1）：静止段边界清楚时，覆盖把通过的窗口中心向两边各扩 st_win // 2 帧，不越过运动段；
     guard（A1′）：纯静止时与 A1 相同；静止之后慢慢开始转动（0.5°/帧）时，A1 把转动的头 30 帧吞进静止，A1′ 在累计转角超出白噪声 P99 处停下
  3. correct 的 carry（候选 B1）：两个静止段之间没有未观测的增量时，沿用第一个段的整体修正，恰好还原第二个段的真漂移；
     carry 关闭时第二个段停在原始积分值
  4. enforce_tops（候选 C1）：静止时陷进桌面 2 cm 的方盒，约束后最低点不低于顶面 −1 cm；没有高于地面的顶时原样返回
2–4 要 numpy、scipy、trimesh，在训练机宿主上跑；Mac 上没有 numpy 时只跑 1。
用法：python3 stages/recovery/test_t2eval.py
"""
import ast
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "libs" / "v2d_infra")); sys.path.insert(0, str(HERE))
ALLOWED = {"os", "argparse", "json", "statistics", "sys", "itertools", "multiprocessing", "pathlib", "numpy", "pyarrow.parquet",
           "scipy.spatial.transform", "add_metrics", "paths", "soma_fk", "t2core", "v51", "nopen", "tier_guard"}
FORBIDDEN = ("tier_1", "TIER1", "tier2_v53", "tier2_v52", "denoise_v1", "denoise_v4", "noise_gen")
P = {"st_win": 61, "st_rot": 17.637, "st_tr": 0.023}


def test_t2eval_imports_and_literals():
    tree = ast.parse((HERE / "t2eval.py").read_text())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add(node.module)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            bad = [f for f in FORBIDDEN if f in node.value]
            assert not bad, f"t2eval.py 的字符串里有 {bad}"
        elif isinstance(node, ast.Name):
            assert node.id not in ("TIER1",), "t2eval.py 引用了 TIER1"
    assert mods <= ALLOWED, f"t2eval.py import 了不在允许清单里的模块：{sorted(mods - ALLOWED)}"


def test_static_mask_cover():
    import numpy as np
    from v51 import static_mask
    T = 200
    p = np.zeros((T, 7)); p[:, 3] = 1.0
    p[100:, 0] = 1.0 + 0.05 * np.arange(T - 100)        # 第 100 帧起跳开 1 m 并持续移动
    st = static_mask(p, P)
    assert np.flatnonzero(st).tolist() == list(range(70)), np.flatnonzero(st)
    cov = static_mask(p, P, cover=True)
    assert np.flatnonzero(cov).tolist() == list(range(100)), np.flatnonzero(cov)


def test_static_mask_guard():
    import numpy as np
    from scipy.spatial.transform import Rotation as Rot
    from v51 import static_mask
    T, sig = 200, (np.radians(0.649), 0.000846)
    p = np.zeros((T, 7)); p[:, 3] = 1.0
    p[100:, 0] = 1.0 + 0.05 * np.arange(T - 100)
    assert np.array_equal(static_mask(p, P, cover=True), static_mask(p, P, cover=True, guard=sig))
    q = np.zeros((T, 7)); q[:, 3] = 1.0
    ang = np.radians(0.5) * np.clip(np.arange(T) - 99, 0, None)          # 第 100 帧起绕 z 轴 0.5°/帧
    q[:, 3:7] = Rot.from_rotvec(np.stack([0 * ang, 0 * ang, ang], 1)).as_quat()[:, [3, 0, 1, 2]]
    a1, a1p = static_mask(q, P, cover=True), static_mask(q, P, cover=True, guard=sig)
    last_core = int(np.flatnonzero(~static_mask(q, P))[0]) - 1          # 第一段静止的最后一帧（序列末尾的截短窗口另会判出静止）
    assert a1[last_core + 30] and not a1p[last_core + 30], (last_core, np.flatnonzero(a1).max(), np.flatnonzero(a1p).max())
    assert (a1p <= a1).all() and a1p[:last_core + 1].all()


def test_carry_restores_drift_without_motion_increments():
    import numpy as np
    from scipy.spatial.transform import Rotation as Rot
    from t2core import correct
    rng = np.random.default_rng(0)
    T = 110
    DR = np.empty((T, 3, 3)); Dx = np.zeros((T, 3)); DR[0] = np.eye(3)
    for t in range(1, T):
        if 50 <= t < 60:                                # “运动段”：积分保持，真漂移也不变（没有未观测的增量）
            DR[t] = DR[t - 1]; Dx[t] = Dx[t - 1]
        else:
            DR[t] = Rot.from_rotvec(rng.normal(0, 0.01, 3)).as_matrix() @ DR[t - 1]; Dx[t] = Dx[t - 1] + rng.normal(0, 0.001, 3)
    R0, x0 = Rot.from_rotvec([0.1, -0.15, 0.05]).as_matrix(), np.array([0.01, -0.02, 0.005])
    Dtrue = np.einsum("tij,jk->tik", DR, R0); xtrue = Dx + x0     # 第 0 帧的漂移 R0、x0，之后的增量都已观测
    segs = [(0, 50), (60, T)]
    target = lambda s, e: (None, None, (Dtrue[(s + e) // 2], xtrue[(s + e) // 2])) if s == 0 else (None, None, None)
    n = np.array([0.0, 0.0, 1.0])
    m2 = (60 + T) // 2
    DRc, Dxc = correct(DR, Dx, segs, target, n, bridge=True, carry=True)
    assert np.allclose(DRc[m2], Dtrue[m2], atol=1e-9) and np.allclose(Dxc[m2], xtrue[m2], atol=1e-12)
    DRo, Dxo = correct(DR, Dx, segs, target, n, bridge=True, carry=False)
    assert np.allclose(DRo[m2], DR[m2]) and np.allclose(Dxo[m2], Dx[m2]) and not np.allclose(DRo[m2], Dtrue[m2], atol=1e-3)


def test_enforce_tops():
    import numpy as np
    import trimesh
    from scipy.spatial.transform import Rotation as Rot
    from nopen import enforce_tops, hull_vertices, min_height
    with tempfile.TemporaryDirectory() as t:
        path = str(Path(t) / "box.obj")
        trimesh.creation.box(extents=(0.3, 0.2, 0.1)).export(path)
        V = hull_vertices(path)
    n, d, H = np.array([0.0, 0.0, 1.0]), 0.0, 0.75
    T = 120
    q = np.zeros((T, 7))
    quat = Rot.from_rotvec([0.0, np.radians(10), 0.0]).as_quat()      # 倾斜 10°
    q[:, 3:7] = quat[[3, 0, 1, 2]]
    low0 = min_height(q, V, n, d)[0]
    q[:, 2] = H - 0.02 - low0                                          # 最低点陷进桌面 2 cm
    out = enforce_tops(q, q, V, n, d, [H], P)
    assert (min_height(out, V, n, d) >= H - 0.01 - 1e-9).all(), min_height(out, V, n, d).min()
    assert np.array_equal(enforce_tops(q, q, V, n, d, [], P), q)


if __name__ == "__main__":
    test_t2eval_imports_and_literals(); print("ok test_t2eval_imports_and_literals")
    try:
        import numpy  # noqa: F401
    except ImportError:
        print("跳过 2–4：这个解释器没有 numpy"); print("ALL_OK（部分）"); sys.exit(0)
    for fn in (test_static_mask_cover, test_static_mask_guard, test_carry_restores_drift_without_motion_increments, test_enforce_tops):
        fn(); print(f"ok {fn.__name__}")
    print("ALL_OK")
