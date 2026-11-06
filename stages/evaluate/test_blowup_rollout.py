"""blowup_rollout.py 的行为测试（暂无契约 ID：运行规范待成文）。要 numpy：Mac 上用 uv 跑，mf4090 宿主机与 FlashCHORD 容器里直接跑。

审查指出这个脚本的要害路径没人跑过：评分报“有世界被判终止”、hydra 把异常打印成 Traceback 再 sys.exit(1) 之后，仍要把抄下的轨迹存盘。
这里不起仿真：临时目录里放一份 blowup_rollout.py、假的 diag_palms.py（PalmRecorder 按场景给出物体与手掌位置）、
假的 flash_chord.evaluation.harness（rollout 调 jax_env.step 走满步数，评分按环境变量成功或抛异常），以及模仿 hydra 的假 evaluate_policy.py。
  1. 没有坏世界、评分成功：npz 在，first_bad_frame 全是 -1，score_error 为空，打印 BLOWUP_ROLLOUT_OK bad_worlds=0
  2. 评分抛终止报错（打印 Traceback、sys.exit(1)）：退出码 0，npz 在，score_error 非空；物体从第 5 帧起 NaN 的世界记 5，
     手掌第 7 帧超过 100 m 的世界记 7，原始的 NaN 没有被清零
  3. 评分抛别的 ValueError、rollout 阶段出错：非零退出，不存盘
  4. 输出已存在：退出码 2，旧文件不动
  5. first_bad_frame：全有限返回 -1；inf、NaN、超远值都认
用法：uv run --offline --no-project --with numpy python stages/evaluate/test_blowup_rollout.py（Mac）；python3 stages/evaluate/test_blowup_rollout.py（训练机）
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
T, W = 12, 6
FAKE_PALMS = '''
import os
import numpy as np

T, W = int(os.environ["FAKE_T"]), int(os.environ["FAKE_W"])


class PalmRecorder:
    def __init__(self, env, step_count):
        self.steps, self.step, self.sides = int(step_count), 0, ("left", "right")

    def wrap(self, inner):
        rec = self

        class Proxy:
            def step(self, action):
                rec.step += 1
                return inner.step(action)

            def __getattr__(self, name):
                return getattr(inner, name)

        return Proxy()

    def arrays(self):
        if self.step != self.steps:
            raise RuntimeError(f"step 被调用 {self.step} 次，应为 {self.steps} 次")
        palms = np.full((T, W, 2, 3), 0.5, np.float32)
        objects = np.full((T, W, 1, 3), 0.25, np.float32)
        if os.environ.get("FAKE_SCEN") == "bad":
            objects[5:, 3] = np.nan       # 世界 3：物体从第 5 帧起 NaN
            palms[7:, 1, 0, 0] = 1.0e6    # 世界 1：左手第 7 帧起飞到 1e6 m（有限但超远）
        return palms, objects
'''
FAKE_HARNESS = '''
import os
from types import SimpleNamespace
import numpy as np

T, W = int(os.environ["FAKE_T"]), int(os.environ["FAKE_W"])


class _Env:
    def step(self, action):
        return None


def record_object_pose_rollout(env, jax_env, *, step_count, reference):
    if os.environ.get("FAKE_ROLLOUT") == "fail":
        raise RuntimeError("rollout 阶段出错（假）")
    for _ in range(step_count):
        jax_env.step(None)
    bad = os.environ.get("FAKE_SCEN") == "bad"
    terminated = np.zeros(W, bool)
    terminated[3] = bad
    return SimpleNamespace(
        completion=SimpleNamespace(terminated=terminated),
        non_finite_worlds=(3,) if bad else (),
        reference_pose_w=np.zeros((T, 1, 7)), tracking_error=np.zeros((T, W, 2), np.float32),
        tracking_error_names=("palm_position_m", "object_position_m"), start_frame=0,
    )


def score_reference_tracking(rollout, criteria):
    kind = os.environ.get("FAKE_SCORE", "ok")
    if kind == "termination":
        raise ValueError("evaluation must disable task failure termination before scoring the full reference")
    if kind == "other":
        raise ValueError("别的错（假）")
    return SimpleNamespace(success_rate=1.0)
'''
FAKE_EVAL = '''
import os
import sys
import traceback

from flash_chord.evaluation import harness


def main():
    rollout = harness.record_object_pose_rollout(object(), harness._Env(), step_count=int(os.environ["FAKE_T"]), reference="ref")
    harness.score_reference_tracking(rollout, "criteria")


if __name__ == "__main__":
    try:
        main()
    except Exception:  # hydra 的做法：打印出来，再以 1 退出
        print("Error executing job with overrides: []", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
'''


def setup(root: Path) -> None:
    shutil.copy(REPO / "stages/evaluate/blowup_rollout.py", root / "blowup_rollout.py")
    (root / "diag_palms.py").write_text(FAKE_PALMS)
    pkg = root / "fake/flash_chord/evaluation"
    pkg.mkdir(parents=True)
    (root / "fake/flash_chord/__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "harness.py").write_text(FAKE_HARNESS)
    (root / "eval.py").write_text(FAKE_EVAL)


def run(root: Path, **kw) -> subprocess.CompletedProcess:
    e = dict(os.environ, PYTHONPATH=str(root / "fake"), V2D_BLOWUP_EVAL_SCRIPT=str(root / "eval.py"), FAKE_T=str(T), FAKE_W=str(W))
    e.update(kw)
    return subprocess.run([sys.executable, str(root / "blowup_rollout.py"), "policy.safetensors", str(root / "out"), "--worlds", str(W)],
                          cwd=root, env=e, capture_output=True, text=True, errors="replace", timeout=120)


def test_clean() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d); setup(root)
        r = run(root)
        assert r.returncode == 0, (r.stdout, r.stderr)
        assert "BLOWUP_ROLLOUT_OK bad_worlds=0 evaluator_non_finite=0 terminated=0" in r.stdout, r.stdout
        z = np.load(root / "out.npz")
        assert z["object_pos"].shape == (T, W, 1, 3) and z["palm_pos"].shape == (T, W, 2, 3)
        assert (z["first_bad_frame"] == -1).all() and str(z["score_error"]) == ""
        assert int(z["step_count"]) == T and int(z["world_count"]) == W


def test_termination_error_still_saves() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d); setup(root)
        r = run(root, FAKE_SCEN="bad", FAKE_SCORE="termination")
        assert r.returncode == 0, (r.stdout, r.stderr)
        assert "Traceback" in r.stderr and "must disable task failure termination" in r.stderr  # hydra 式的输出确实出现了
        assert "BLOWUP_ROLLOUT_OK bad_worlds=2 evaluator_non_finite=1 terminated=1" in r.stdout, r.stdout
        z = np.load(root / "out.npz")
        first = z["first_bad_frame"]
        assert first.tolist() == [-1, 7, -1, 5, -1, -1], first
        assert "must disable task failure termination" in str(z["score_error"])
        assert np.isnan(z["object_pos"][5:, 3]).all() and np.isfinite(z["object_pos"][:5, 3]).all()  # 原始值没被清零
        assert float(z["palm_pos"][7, 1, 0, 0]) == 1.0e6
        assert z["evaluator_non_finite_worlds"].tolist() == [3] and z["terminated"].tolist() == [False, False, False, True, False, False]


def test_other_errors_do_not_save() -> None:
    for kw in ({"FAKE_SCORE": "other"}, {"FAKE_ROLLOUT": "fail"}):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); setup(root)
            r = run(root, **kw)
            assert r.returncode != 0, (kw, r.stdout)
            assert not (root / "out.npz").exists() and "BLOWUP_ROLLOUT_OK" not in r.stdout, kw


def test_refuse_existing() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d); setup(root)
        (root / "out.npz").write_bytes(b"old")
        r = run(root)
        assert r.returncode == 2 and (root / "out.npz").read_bytes() == b"old", (r.returncode, r.stderr)


def test_first_bad_frame() -> None:
    sys.path.insert(0, str(REPO / "stages/evaluate"))
    from blowup_rollout import FAR_M, first_bad_frame
    a = np.zeros((6, 4, 1, 3), np.float32)
    assert first_bad_frame(a).tolist() == [-1, -1, -1, -1]
    a[2, 0, 0, 1] = np.inf
    a[4, 1, 0, 2] = np.nan
    a[3, 2, 0, 0] = FAR_M * 2
    b = np.zeros((6, 4, 2, 3), np.float32)
    b[1, 1, 1, 0] = -np.inf  # 第二个数组里更早出问题，取更早的
    assert first_bad_frame(a, b).tolist() == [2, 1, 3, -1]
    a[:] = FAR_M  # 恰好等于门限不算
    assert first_bad_frame(a).tolist() == [-1, -1, -1, -1]


if __name__ == "__main__":
    for t in (test_clean, test_termination_error_still_saves, test_other_errors_do_not_save, test_refuse_existing, test_first_bad_frame):
        t(); print(f"ok {t.__name__}")
    print("ALL_OK")
