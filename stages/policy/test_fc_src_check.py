"""fc_src_check.py 的行为测试（暂无契约 ID：运行规范待成文）。只用标准库（加 git），Mac 与训练机都能跑。

临时目录里搭最小仓库：真的 fc_src_check.py；third_party/video_to_data 是一个只有一个提交的 git 仓库，里面放一棵小的 flash_chord 树
（含子目录、目录符号链接）；副本由它复制，再按“补丁”改两个文件、加一个文件，清单列这三个文件的 sha256。HEAD 的期望值经
V2D_FC_SRC_BASE_HEAD 换成这个临时仓库的短哈希。
  1. 副本只差清单里的三个文件、有 SOURCE_V2D.txt：FC_SRC_OK；副本里另有 __pycache__/*.pyc、outputs/、logs/、assets/.cache/、根下 wandb/、.hydra/ 也照样 OK；
  2. 多改一个清单外的文件、清单里的 sha256 不对、少了 SOURCE_V2D.txt、对照的已跟踪文件被改过、HEAD 不对、
     目录符号链接指向别处：都 FC_SRC_FAIL。
用法：python3 stages/policy/test_fc_src_check.py
"""
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(cwd: Path, *a: str) -> str:
    env = dict(os.environ, GIT_AUTHOR_NAME="t", GIT_AUTHOR_EMAIL="t@t", GIT_COMMITTER_NAME="t", GIT_COMMITTER_EMAIL="t@t")
    return subprocess.run(["git", *a], cwd=cwd, env=env, capture_output=True, text=True, check=True).stdout.strip()


def setup(root: Path) -> tuple:
    (root / "stages/policy").mkdir(parents=True)
    shutil.copy(REPO / "stages/policy/fc_src_check.py", root / "stages/policy")
    up = root / "third_party/video_to_data"
    base = up / "robotic_grounding/flash_chord"
    for rel, text in (("scripts/evaluate_policy.py", "old eval\n"), ("src/flash_chord/scene/support.py", "old support\n"),
                      ("src/flash_chord/training/train.py", "train\n"), ("src/flash_chord/assets/a.txt", "asset\n")):
        (base / rel).parent.mkdir(parents=True, exist_ok=True)
        (base / rel).write_text(text)
    os.symlink("src/flash_chord/assets", base / "assets_link")
    git(up, "init", "-q"); git(up, "add", "-A"); git(up, "commit", "-q", "-m", "base")
    head = git(up, "rev-parse", "--short=7", "HEAD")
    copy = root / "copy"
    shutil.copytree(base, copy, symlinks=True)
    (copy / "scripts/evaluate_policy.py").write_text("new eval\n")
    (copy / "src/flash_chord/scene/support.py").write_text("new support\n")
    (copy / "tests").mkdir()
    (copy / "tests/test_new.py").write_text("new test\n")
    (copy / "SOURCE_V2D.txt").write_text("flash_chord 8c778f905c（测试）\n")
    man = root / "man.sha256"
    man.write_text("".join(f"{sha(copy / r)}  {r}\n" for r in ("scripts/evaluate_policy.py", "src/flash_chord/scene/support.py", "tests/test_new.py")))
    return head, copy, man, base


def check(root: Path, copy: Path, man: Path, head: str) -> tuple:
    r = subprocess.run(["python3", "stages/policy/fc_src_check.py", str(copy), str(man)], cwd=root, capture_output=True, text=True,
                       env=dict(os.environ, V2D_FC_SRC_BASE_HEAD=head), timeout=60)
    return r.returncode, r.stdout


def test_ok_and_ignored_noise() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); head, copy, man, _ = setup(root)
        rc, out = check(root, copy, man, head)
        assert rc == 0 and out.rstrip().endswith("FC_SRC_OK"), out
        for d in ("src/flash_chord/__pycache__", "outputs", "logs", "wandb", ".hydra", "src/flash_chord/logs", "src/flash_chord/assets/.cache/convex"):
            (copy / d).mkdir(parents=True, exist_ok=True)
            (copy / d / "x.pyc").write_text("noise")
        rc, out = check(root, copy, man, head)
        assert rc == 0 and "wandb" in out and out.rstrip().endswith("FC_SRC_OK"), out


def test_failures() -> None:
    def extra_file(copy, man, base, root): (copy / "src/flash_chord/training/train.py").write_text("changed\n")
    def bad_sha(copy, man, base, root): man.write_text(man.read_text().replace(man.read_text()[:8], "00000000", 1))
    def no_source(copy, man, base, root): (copy / "SOURCE_V2D.txt").unlink()
    def dirty_base(copy, man, base, root): (base / "src/flash_chord/training/train.py").write_text("dirty\n")
    def relinked(copy, man, base, root): (copy / "assets_link").unlink(); os.symlink("src", copy / "assets_link")
    for mutate in (extra_file, bad_sha, no_source, dirty_base, relinked):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); head, copy, man, base = setup(root)
            mutate(copy, man, base, root)
            rc, out = check(root, copy, man, head)
            assert rc == 1 and "FC_SRC_FAIL" in out.splitlines()[-1], (mutate.__name__, out)
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); head, copy, man, _ = setup(root)
        rc, out = check(root, copy, man, "33129dd")
        assert rc == 1 and "FC_SRC_FAIL 1" in out, out


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("ALL_OK")
