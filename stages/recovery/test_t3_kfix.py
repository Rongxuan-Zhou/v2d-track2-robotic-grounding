"""t3_kfix.py 的单元测试（暂无契约 ID：运行规范待成文；计划 agent-doc/plan-t3-kfix-2026-10-16.md 方案 1）。Mac 上跑，只要 numpy；人体部分要 SOMA，
在容器里由另一位用户 `_move_body` 自带的逐帧核对守（关节误差上限 1 mm），这里不测。
  1. 口径 F 与口径 Z：随机点经 M 映射后在 K_t 下的像素与原来在 K_c 下的相同（差 ≤ 1e-9 px），点的深度 0.5–12 m、横向 ±6 m；
  1b. fx ≠ fy（放大到 1e-3）时两种口径仍精确（ep03 的 CARI4D 内参是 float32，fx 与 fy 末位不同）；
  2. 口径 F 的 M 与 K_t⁻¹K_c 乘 f_t/f_c 相同；口径 F 的深度比是 f_t/f_c，口径 Z 的深度不变；
  3. 反向对照：主点差的符号故意取反，pixel_check 报错（SystemExit）；
  4. 口径参数错（不是 F、Z）报 ValueError。
用法：uv run --no-project --python 3.12 --with numpy python stages/recovery/test_t3_kfix.py；最后一行 ALL_OK。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t3_kfix import kmat, mapping, pixel_check, project  # noqa: E402

KC = kmat(834.97, 768.0, 576.0)
KT = kmat(866.3, 701.8, 531.1)


def points(n: int = 2000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.5, 12.0, n)
    return np.c_[rng.uniform(-6, 6, n), rng.uniform(-1.5, 1.4, n), z]


def test_pixels() -> None:
    X = points()
    for mode in ("F", "Z"):
        M = mapping(KC, KT, mode)
        d = float(np.abs(project(KC, X) - project(KT, X @ M.T)).max())
        assert d <= 1e-9, (mode, d)
        assert pixel_check(KC, KT, X, M) <= 1e-9
    print("  1. 口径 F、Z：映射后在 K_t 下的像素与原来在 K_c 下的相同")


def test_fx_ne_fy() -> None:
    """CARI4D 的内参按 float32 存，fx 与 fy 末位可能不同；这里放大到 fy = fx·(1 + 1e-3)，映射仍须精确。"""
    kc = KC.copy()
    kc[1, 1] *= 1.001
    X = points(500, 3)
    for mode in ("F", "Z"):
        M = mapping(kc, KT, mode)
        d = float(np.abs(project(kc, X) - project(KT, X @ M.T)).max())
        assert d <= 1e-9, (mode, d)
    print("  1b. fx ≠ fy 时两种口径的映射仍精确（ep03 的 float32 内参就是这种情况）")


def test_formulas() -> None:
    MF, MZ = mapping(KC, KT, "F"), mapping(KC, KT, "Z")
    assert np.abs(MF - (KT[0, 0] / KC[0, 0]) * np.linalg.inv(KT) @ KC).max() < 1e-12
    X = points(50, 1)
    assert np.allclose((X @ MF.T)[:, 2], X[:, 2] * KT[0, 0] / KC[0, 0]) and np.allclose((X @ MZ.T)[:, 2], X[:, 2])
    print("  2. 口径 F 的 M = (f_t/f_c)·K_t⁻¹K_c；口径 F 深度乘 f_t/f_c，口径 Z 深度不变")


def test_negative() -> None:
    M = mapping(KC, KT, "F")
    bad = M.copy()
    bad[0, 2] *= -1   # 主点差的符号取反
    try:
        pixel_check(KC, KT, points(100, 2), bad)
    except SystemExit:
        pass
    else:
        raise AssertionError("主点差符号取反的映射没被 pixel_check 拦下")
    try:
        mapping(KC, KT, "X")
    except ValueError:
        pass
    else:
        raise AssertionError("口径参数错没报 ValueError")
    print("  3、4. 反向对照：错的映射被拦下；口径参数错报 ValueError")


if __name__ == "__main__":
    test_pixels()
    test_fx_ne_fy()
    test_formulas()
    test_negative()
    print("ALL_OK")
