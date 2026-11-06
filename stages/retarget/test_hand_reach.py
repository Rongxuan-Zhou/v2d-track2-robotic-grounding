"""hand_reach：余弦过渡与一帧的阻尼最小二乘拉手（合成的 2 关节平面臂）。

计划 agent-doc/plan-t3hand-2026-10-27.md 第 1 节 P1。契约 RETARGET-R00x（待审）。

Runs without pytest: python test_hand_reach.py
"""
import numpy as np

from hand_reach import cosine_ramp, pull_frame


def _arm(l1=0.3, l2=0.3):
    """平面 2 关节臂的末端点与到一面墙（x = X_WALL）的有向距离。"""
    def fk(q, k=0):
        x = l1 * np.cos(q[0]) + l2 * np.cos(q[0] + q[1])
        y = l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1])
        return np.array([x, y, 0.0])
    return fk


X_WALL = 0.55


def _sdist(fk):
    def f(q):
        p = fk(q)
        return X_WALL - p[0], np.array([1.0, 0.0, 0.0]), 0  # 距离 = 到墙的间隙，方向朝墙，末端点 0
    return f


def test_ramp_edges_rise_to_one():
    w = cosine_ramp(30, 10)
    assert w[0] < 0.1 and w[29] < 0.1 and abs(w[15] - 1) < 1e-12 and abs(w[9] - w[20]) < 1e-12


def test_ramp_short_window():
    w = cosine_ramp(5, 10)
    assert len(w) == 5 and w.max() <= 1 and w[2] >= w[0]


def test_pull_reaches_target_distance():
    fk = _arm()
    q0 = np.array([0.3, 0.8])  # 末端 x 约 0.37，离墙 0.18
    q, d, n = pull_frame(fk, _sdist(fk), q0, np.array([-3, -3.0]), np.array([3, 3.0]), 0.02, tol=0.003)
    assert abs(d - 0.02) <= 0.003 and n > 0 and np.degrees(np.abs(q - q0)).max() < 60


def test_pull_leaves_a_hand_already_close():
    fk = _arm()
    q0 = np.array([0.05, 0.0])  # 末端 x 约 0.6，已越过目标
    q, d, n = pull_frame(fk, _sdist(fk), q0, np.array([-3, -3.0]), np.array([3, 3.0]), 0.02)
    assert n == 0 and np.array_equal(q, q0)


def test_pull_respects_joint_limits():
    fk = _arm()
    q0 = np.array([0.3, 0.8])
    lower, upper = np.array([0.25, 0.75]), np.array([0.35, 0.85])
    q, d, n = pull_frame(fk, _sdist(fk), q0, lower, upper, 0.02, iters=20)
    assert (q >= lower - 1e-12).all() and (q <= upper + 1e-12).all()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} passed")
