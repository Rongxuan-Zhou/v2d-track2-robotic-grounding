"""fall_order.py 的测试：用造的逐步误差覆盖物体先、骨盆先、同时、摔倒世界太少、门不过时停止，以及冻结期与统一物体门槛。

守 agent-doc/plan-t1-fall-order-2026-10-07.md 第 2 版第 4 节的定义、门与结论规则。
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NAMES = ["pelvis_position_m", "pelvis_orientation_rad", "palm_position_m", "palm_orientation_rad", "object_position_m", "object_orientation_rad"]
T = 200


def make_run(root, run, ori_thr, pattern, n_fail=50, mp_shift=0.0, sha="abc"):
    """pattern：每个失败世界的 (t_obj, t_pel) 列表循环使用。"""
    w_total = 10 + n_fail
    te = np.zeros((T, w_total, 6), dtype=np.float32)
    succ = np.ones(w_total, dtype=bool)
    for i, w in enumerate(range(10, w_total)):
        succ[w] = False
        t_obj, t_pel = pattern[i % len(pattern)]
        if t_obj is not None:
            te[t_obj:, w, 4] = 0.3
        te[t_pel:, w, 0] = 1.0
    te[10:40, 0, 4] = 0.5  # 冻结期内，不计
    o = root / "runs/flash_chord" / run
    (o / "eval_dump").mkdir(parents=True)
    np.savez_compressed(o / "eval_dump/eval_1_w9.npz", tracking_error=te.astype(np.float16), error_names=np.array(NAMES),
                        criteria_error_names=np.array(NAMES), thresholds=np.array([0.5, 1.0, 0.3, 1.0, 0.1, ori_thr]),
                        active_after_step=np.array([-1, -1, 50, 50, 50, 50]), non_finite_worlds=np.array([], dtype=np.int64),
                        mp_success=succ, mp_position_m=np.where(succ, 0.01, 0.3), control_fps=np.float64(50.0))
    mp = float(succ.mean())
    new = {"step_count": T, "world_count": w_total, "checkpoint_sha256": sha,
           "metrics": {"maniptrans_sr": mp, "mppe_cm": 5.0, "object_position_error_std_m": 0.1}}
    old = {"step_count": T, "world_count": w_total, "checkpoint_sha256": "abc",
           "metrics": {"maniptrans_sr": mp + mp_shift, "mppe_cm": 5.0, "object_position_error_std_m": 0.1}}
    (o / "eval_dump/eval_1_w9.json").write_text(json.dumps(new))
    (o / "eval_1_w9.json").write_text(json.dumps(old))


def run(tmp_path, var_pattern, n_fail=50, **kw):
    spec = {"episodes": [{"ep": 1, "base42": "b42", "base43": "b43", "var42": "v42", "var43": "v43"}]}
    for name in ("b42", "b43"):
        make_run(tmp_path, name, 1.5, [(60, 120), (150, 70)], n_fail)
    for name in ("v42", "v43"):
        make_run(tmp_path, name, 0.6, var_pattern, n_fail, **kw)
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    return subprocess.run([sys.executable, str(HERE / "fall_order.py"), str(tmp_path / "spec.json"), "--episodes", "1",
                           "--steps", "1", "--worlds", "9", "--root", str(tmp_path)], capture_output=True, text=True)


def test_object_first(tmp_path):
    p = run(tmp_path, [(60, 120)] * 4 + [(150, 70)])
    assert p.returncode == 0, p.stderr
    assert "FALL_ORDER 支持甲" in p.stdout


def test_pelvis_first(tmp_path):
    p = run(tmp_path, [(150, 70)] * 4 + [(None, 90)])
    assert "FALL_ORDER 支持乙" in p.stdout


def test_simultaneous(tmp_path):
    p = run(tmp_path, [(100, 110)])
    assert "FALL_ORDER 不下结论：两类都不到 60%" in p.stdout


def test_too_few_fallen(tmp_path):
    p = run(tmp_path, [(60, 120)], n_fail=20)
    assert "只描述" in p.stdout


def test_gate_stops(tmp_path):
    p = run(tmp_path, [(60, 120)], mp_shift=0.05)
    assert p.returncode == 1 and "FALL_ORDER STOP" in p.stdout


def test_gate_sha(tmp_path):
    p = run(tmp_path, [(60, 120)], sha="other")
    assert p.returncode == 1 and "sha256" in p.stdout


def test_freeze_and_common_threshold():
    sys.path.insert(0, str(HERE))
    import fall_order
    te = np.zeros((T, 3, 6), dtype=np.float32)
    te[10:40, 0, 4] = 0.5
    te[100:, 1, 5] = 1.0  # 超过统一门槛 0.6，低于基线 checkpoint 的 1.5
    z = {"tracking_error": te, "thresholds": np.array([0.5, 1.0, 0.3, 1.0, 0.1, 1.5]), "active_after_step": np.array([-1, -1, 50, 50, 50, 50])}
    _, t_obj, t_pel, _, _ = fall_order.crossings(z)
    assert list(t_obj) == [-1, 100, -1] and list(t_pel) == [-1, -1, -1]
    _, t_obj0, _, _, _ = fall_order.crossings(z, object_from_zero=True)
    assert list(t_obj0) == [10, 100, -1]
