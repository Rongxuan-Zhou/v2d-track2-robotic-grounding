"""case_clip.py 挑世界规则的测试（不需要 GPU）：守 agent-doc/plan-site-cases-video-2026-10-08.md 第 4 节（三类各取最接近中位数的世界，空类如实报告）。"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from case_clip import pick_worlds  # noqa: E402


def test_three_kinds_and_median():
    success = np.array([True, True, True, False, False, False, False])
    pos = np.array([0.01, 0.02, 0.03, 0.40, 0.05, 0.06, 0.07])
    t_obj = np.array([-1, -1, -1, 100, -1, -1, -1])
    t_pel = np.array([-1, -1, -1, 180, -1, -1, -1])
    p = pick_worlds(success, t_obj, t_pel, pos, ["success", "fall", "nearmiss"])
    assert p["success"][0] == 1  # 中位数 0.02
    assert p["fall"][0] == 3
    assert p["nearmiss"][0] == 5  # 0.05、0.06、0.07 的中位数 0.06


def test_empty_kind_reported():
    success = np.array([True, True])
    p = pick_worlds(success, np.array([-1, -1]), np.array([-1, -1]), np.array([0.01, 0.02]), ["fall"])
    assert p["fall"][0] is None and "没有" in p["fall"][1]


def test_pelvis_first_is_not_fall_case():
    success = np.array([False])
    p = pick_worlds(success, np.array([150]), np.array([70]), np.array([0.3]), ["fall", "nearmiss"])
    assert p["fall"][0] is None and p["nearmiss"][0] is None
