"""onset_terms.py 的测试：越线门槛可调，默认门槛的行为与 fall_order.crossings 相同（计划 agent-doc/plan-r3o-2026-10-15.md 第 0 步，审查第 2 条）。

造一条 run：2 个成功世界，10 个失败世界；冻结期 50 步之后，物体位置误差每步涨 1.6 mm（第 69 步过 3 cm、第 113 步过 10 cm），
朝向误差每步涨 0.0121 rad（第 93 步过 0.52 rad、第 100 步过 0.6 rad），骨盆第 180 步越线。斜率避开整步压线，免得 float32 舍入决定越线在哪一步。
1. 默认门槛（0.10 m、0.6 rad）：物体在第 100 步越线，只有朝向超，k=25 时物体先 100%；
2. MP-SR 的常数（0.03 m、0.52 rad）：物体在第 69 步越线，只有位置超；
3. 不同档的 run 一起给：退出码 2；
4. 骨盆没越线的失败世界不算进 k=25 的先后（口径同 fall_order.shares）：两个失败世界骨盆不越线时，摔倒的是 8 个、物体先仍是 100%。
要 numpy 与 pytest。
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
T, W = 200, 12


def make(root, run):
    te = np.zeros((T, W, 6), dtype=np.float32)
    steps = np.arange(T, dtype=np.float32)
    ramp = np.clip(steps - 50, 0, None)
    succ = np.ones(W, dtype=bool)
    for w in range(2, W):
        succ[w] = False
        te[:, w, 4] = 0.0016 * ramp
        te[:, w, 5] = 0.0121 * ramp
        te[180:, w, 0] = 1.0
    o = root / "runs/flash_chord" / run / "eval_dump"
    o.mkdir(parents=True)
    np.savez_compressed(o / "eval_1_w9.npz", tracking_error=te, thresholds=np.array([0.7, 1.5, 0.15, 1.5, 0.1, 0.6]),
                        active_after_step=np.array([-1, -1, 50, 50, 50, 50]), non_finite_worlds=np.array([], dtype=np.int64),
                        mp_success=succ)
    (o / "eval_1_w9.json").write_text(json.dumps({"step_count": T, "world_count": W}))


def call(root, *args):
    cmd = [sys.executable, str(HERE / "onset_terms.py"), *args, "--steps", "1", "--worlds", "9", "--root", str(root)]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_default_thresholds(tmp_path):
    make(tmp_path, "t1-ep4-w2048-cap10m-ml40")
    p = call(tmp_path, "t1-ep4-w2048-cap10m-ml40")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "物体越线 10 个：只有位置 0，只有朝向 10，两者 0" in p.stdout, p.stdout
    assert "t_obj 中位第 100 步" in p.stdout and "物体先 100.0%" in p.stdout, p.stdout


def test_mp_sr_thresholds(tmp_path):
    make(tmp_path, "t1-ep4-w2048-cap10m-ml40")
    p = call(tmp_path, "t1-ep4-w2048-cap10m-ml40", "--pos-thr", "0.03", "--ori-thr", "0.52")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "位置 > 0.03 m 或朝向 > 0.52 rad" in p.stdout
    assert "物体越线 10 个：只有位置 10，只有朝向 0，两者 0" in p.stdout, p.stdout
    assert "t_obj 中位第 69 步" in p.stdout, p.stdout


def test_mixed_tiers_refused(tmp_path):
    make(tmp_path, "t1-ep4-w2048-cap10m-ml40")
    make(tmp_path, "t2-ep4-w2048-cap10m-ml40")
    p = call(tmp_path, "t1-ep4-w2048-cap10m-ml40", "t2-ep4-w2048-cap10m-ml40")
    assert p.returncode == 2 and "一次只收同一档" in p.stderr, p.stdout + p.stderr


def test_unfallen_worlds_excluded(tmp_path):
    make(tmp_path, "t1-ep4-w2048-cap10m-ml40")
    o = tmp_path / "runs/flash_chord/t1-ep4-w2048-cap10m-ml40/eval_dump/eval_1_w9.npz"
    z = dict(np.load(o))
    te = z["tracking_error"].copy()
    te[:, 2:4, 0] = 0.0   # 世界 2、3 失败但骨盆从没越线
    z["tracking_error"] = te
    np.savez_compressed(o, **z)
    p = call(tmp_path, "t1-ep4-w2048-cap10m-ml40")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "摔倒的失败世界 8 个（失败 10 个），物体先 100.0%" in p.stdout, p.stdout

