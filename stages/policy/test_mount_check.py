"""mount_check.py 的行为测试（暂无契约 ID：运行规范待成文）。只用标准库，Mac 与训练机都能跑。

临时目录里放真的 mount_check.py、两个副本目录（训练副本经符号链接给出，核对真实路径），以及按 fc_run.sh、fc_eval.sh 的落点写的假日志。
  1. 冒烟、正式、评测三份日志都只挂了该挂的副本：MOUNT_OK；
  2. 正式训练日志里没有挂载行（挂的是 third_party 的 v0.4.0）、评测挂错成训练副本、混挂两份、评测日志不存在：MOUNT_FAIL。
用法：python3 stages/policy/test_mount_check.py
"""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = "t1-ep7-w2048-cap4m-p185-mf4090"


def line(path: Path) -> str:
    return f"in-flash-chord: /workspace ← {path}（flash_chord …）\n"


def setup(root: Path) -> tuple:
    (root / "stages/policy").mkdir(parents=True)
    shutil.copy(HERE / "mount_check.py", root / "stages/policy")
    train, ev = root / "fc_src/v040_p185", root / "fc_src/fc_8c778f9"
    train.mkdir(parents=True); ev.mkdir(parents=True)
    os.symlink(train, root / "train_link")
    logs = root / "runs/logs"; logs.mkdir(parents=True)
    (logs / f"smoke-{RUN}.log").write_text("=== start\n" + line(train.resolve()))
    (logs / f"{RUN}.log").write_text("=== s1\n" + line(train.resolve()) + "=== s2\n" + line(train.resolve()))
    (logs / f"{RUN}.eval_349999104_w4096.log").write_text("=== eval\n" + line(ev.resolve()))
    return train, ev, logs


def check(root: Path) -> tuple:
    r = subprocess.run(["python3", "stages/policy/mount_check.py", "--train", "train_link", "--eval", "fc_src/fc_8c778f9", RUN],
                       cwd=root, capture_output=True, text=True, timeout=30)
    return r.returncode, r.stdout


def test_ok() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t).resolve(); setup(root)
        rc, out = check(root)
        assert rc == 0 and out.rstrip().endswith("MOUNT_OK"), out


def test_failures() -> None:
    def no_mount(train, ev, logs): (logs / f"{RUN}.log").write_text("=== s1 挂的是 third_party\n")
    def wrong_eval(train, ev, logs): (logs / f"{RUN}.eval_349999104_w4096.log").write_text(line(train.resolve()))
    def mixed(train, ev, logs): (logs / f"{RUN}.log").write_text(line(train.resolve()) + line(ev.resolve()))
    def no_eval(train, ev, logs): (logs / f"{RUN}.eval_349999104_w4096.log").unlink()
    for mutate in (no_mount, wrong_eval, mixed, no_eval):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t).resolve(); train, ev, logs = setup(root)
            mutate(train, ev, logs)
            rc, out = check(root)
            assert rc == 1 and out.rstrip().endswith("MOUNT_FAIL 1"), (mutate.__name__, out)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("ALL_OK")
