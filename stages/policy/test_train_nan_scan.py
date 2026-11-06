"""train_nan_scan.py --log 的行为测试（暂无契约 ID：运行规范待成文）。只用标准库，Mac 与训练机都能跑。

摘要段取自 mf4090 上的真实日志（9/28 冒烟 smoke-t1-ep21-w2048-cap4m-mf4090 正常；9/29 正式 t1-ep1-w2048-cap4m-mf4090 发散），
只截了 W&B 摘要那几行。判据：没有摘要、动作均值 / 熵 / 回报里有 nan、loss_scale 为 0 或 nan，任何一条退出码 1；正常的退出码 0。
用法：python3 stages/policy/test_train_nan_scan.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "train_nan_scan.py"

HEALTHY = """\
wandb: Run summary:
wandb:          curriculum/immediate_first_frame_probability 0
wandb:                                curriculum/stage_index 0
wandb:             episode/length_mean_per_completed_episode 95.67143
wandb:             episode/return_mean_per_completed_episode 7.57051
wandb:                            mixed_precision/loss_scale 2097152.0
wandb:                 policy/action_abs_mean_per_env_action 4.87944
wandb:                  policy/entropy_mean_per_actor_update 111.65146
wandb:                                                   +34 ...
wandb:
wandb: Synced 5 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
completed 4096000 environment steps; checkpoints: /repo/runs/flash_chord/smoke-x
"""

DIVERGED = """\
wandb: Run summary:
wandb:                                curriculum/stage_index 8
wandb:             episode/length_mean_per_completed_episode 5.0786
wandb:             episode/return_mean_per_completed_episode nan
wandb:                            mixed_precision/loss_scale 0
wandb:                 policy/action_abs_mean_per_env_action nan
wandb:                  policy/entropy_mean_per_actor_update nan
wandb:                                                   +34 ...
wandb:
completed 349999104 environment steps; checkpoints: /repo/runs/flash_chord/t1-x
"""

# 摘要里三项都有限，只有 loss_scale 掉到 0：仍算发散
SCALE_ZERO = HEALTHY.replace("mixed_precision/loss_scale 2097152.0", "mixed_precision/loss_scale 0")
# 摘要被截断、没有 loss_scale 这一项：不据它判，三项有限就算正常
NO_SCALE = HEALTHY.replace("wandb:                            mixed_precision/loss_scale 2097152.0\n", "")
NO_SUMMARY = "wandb: Synced 5 W&B file(s)\ncompleted 4096000 environment steps\n"


def run_log(text: str) -> tuple:
    with tempfile.TemporaryDirectory() as t:
        p = Path(t, "smoke-x.s1.log")
        p.write_text(text, encoding="utf-8")
        r = subprocess.run([sys.executable, str(SCRIPT), "--log", str(p)], capture_output=True, text=True, timeout=30)
        return r.returncode, r.stdout


def test_healthy() -> None:
    rc, out = run_log(HEALTHY)
    assert rc == 0, out
    assert "平均回合长度 95.67143" in out and "loss_scale 2097152.0" in out and "nan 0 项" in out, out


def test_diverged() -> None:
    rc, out = run_log(DIVERGED)
    assert rc == 1, out
    assert "nan 3 项" in out and "loss_scale 0" in out, out


def test_loss_scale_zero_alone() -> None:
    rc, out = run_log(SCALE_ZERO)
    assert rc == 1, out


def test_missing_loss_scale_is_not_divergence() -> None:
    rc, out = run_log(NO_SCALE)
    assert rc == 0, out
    assert "loss_scale —" in out, out


def test_no_summary() -> None:
    rc, out = run_log(NO_SUMMARY)
    assert rc == 1 and "没有 W&B 摘要" in out, out


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("ALL_OK")
