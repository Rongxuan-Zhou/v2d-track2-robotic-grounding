"""t3_frame0_table.py 纯函数的测试（暂无契约 ID：运行规范待成文；计划 agent-doc/plan-t3b2-2026-10-28.md 第 2.1 版第 1 步第 4 条、B2 规则）。要 numpy、pyarrow。
  1. right_mul：右乘 g 绕质心，质心的世界位置不变、朝向是 R g；g 为恒等时位姿不变；
  2. B2 选支：第 0 帧位姿是真值右乘 g⁻¹（加 3° 噪声）时，固定对齐下选回 g；
  3. b2_choose 的每条规则：原差 ≤ 30° 不改、最好的群元后仍 > 30° 不改、只小 < 20° 不改、Tier 1 不可见不改、恒等最小不改；
  4. frame0_diff 与 align_frefs_t3.angle_deg 一致（R_T 为恒等时就是两个朝向的夹角）；
  5. near_choose（plan-t3b2n12）：唯一候选可选；空集、两个候选都 ≤ 30°、候选后仍 > 30°、原差 ≤ 30°、不可见、改进 < 20° 都拒绝。
用法：uv run --no-project --python 3.12 --with numpy --with pyarrow python stages/evaluate/test_t3_frame0_table.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import align_frefs_t3 as al   # noqa: E402
import t3_frame0_table as ft   # noqa: E402


def pose(R, p):
    return np.r_[p, al.R_to_q(R)]


def test_right_mul() -> None:
    rng = np.random.default_rng(0)
    c = np.array([0.05, -0.02, 0.3])
    g = ft.rot([0, 0, 1], 180)
    P = np.stack([pose(al.rotvec_to_R(rng.normal(size=3)), rng.normal(size=3)) for _ in range(5)])
    Q = ft.right_mul(P, g, c)
    for i in range(5):
        R, R2 = al.q_to_R(P[i, 3:]), al.q_to_R(Q[i, 3:])
        assert np.abs((P[i, :3] + R @ c) - (Q[i, :3] + R2 @ c)).max() < 1e-12
        assert np.abs(R2 - R @ g).max() < 1e-12
    assert np.abs(ft.right_mul(P, np.eye(3), c) - P).max() < 1e-12


def test_b2_picks_g() -> None:
    rng = np.random.default_rng(1)
    c = np.array([0.0, 0.1, 0.0])
    G = [np.eye(3), ft.rot([1, 0, 0], 180)]
    R1 = al.rotvec_to_R(rng.normal(size=3))
    noise = ft.rot([0, 1, 0], 3)
    P0 = pose(R1 @ G[1].T @ noise, np.zeros(3))[None]
    d = {k: ft.frame0_diff(ft.right_mul(P0, g, c), np.eye(3), R1) for k, g in enumerate(G)}
    assert d[0] > 170 and d[1] < 5, d
    assert ft.b2_choose(d, True)[0] == 1


def test_b2_rules() -> None:
    assert ft.b2_choose({0: 25.0, 1: 2.0}, True)[0] == 0
    assert ft.b2_choose({0: 120.0, 1: 40.0}, True)[0] == 0
    assert ft.b2_choose({0: 45.0, 1: 28.0}, True)[0] == 0
    assert ft.b2_choose({0: 170.0, 1: 3.0}, False)[0] == 0
    assert ft.b2_choose({0: 60.0, 1: 90.0}, True)[0] == 0
    assert ft.b2_choose({0: 170.0, 1: 3.0}, True) == (1, "170.0° → 3.0°")


def test_near_choose() -> None:
    assert ft.near_choose(179.9, [("a", 1.0)], True) == ("a", "179.9° → 1.0°")
    assert ft.near_choose(179.9, [], True)[0] is None
    assert ft.near_choose(179.9, [("a", 1.0), ("b", 20.0)], True)[0] is None  # 两个都 ≤ 30°：拒绝
    assert ft.near_choose(179.9, [("a", 40.0)], True)[0] is None
    assert ft.near_choose(25.0, [("a", 1.0)], True)[0] is None
    assert ft.near_choose(179.9, [("a", 1.0)], False)[0] is None
    assert ft.near_choose(45.0, [("a", 28.0)], True)[0] is None  # 只小 17°


def test_frame0_diff() -> None:
    A, B = ft.rot([0, 0, 1], 40), ft.rot([0, 0, 1], 100)
    assert abs(ft.frame0_diff(pose(A, np.zeros(3))[None], np.eye(3), B) - 60) < 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("TEST_T3_FRAME0_TABLE_OK")
