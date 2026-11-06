"""stages/recovery/t3r_run.sh 的测试（只在 Linux 上跑，要能 unshare 用户命名空间；Mac 上跳过）。守计划 agent-doc/plan-t3takeover-v3-2026-10-21.md 判据 K0：
用一个临时目录扮“她的目录”，命名空间里写不进去、命名空间外写得进去（对照）；参数落在她目录下就拒绝；白名单里的解释器放行；我们的根可写；根在她目录下就拒绝。
用法：python3 stages/recovery/test_t3r_run.py（也可用 pytest 跑）。最后一行 T3R_RUN_TESTS_OK。
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN = HERE / "t3r_run.sh"


def _skip() -> bool:
    return sys.platform != "linux" or shutil.which("unshare") is None or shutil.which("setpriv") is None


def _run(her: Path, root: Path, *cmd: str, allow: str = "", mask: str = "") -> subprocess.CompletedProcess:
    env = dict(os.environ, T3R_HER=str(her), T3R_ROOT=str(root), T3R_ALLOW=allow, T3R_MASK=mask)
    return subprocess.run(["bash", str(RUN), *cmd], env=env, capture_output=True, text=True, check=False)


def _dirs(d: str) -> tuple[Path, Path]:
    her, root = Path(d) / "her", Path(d) / "root"
    her.mkdir()
    root.mkdir()
    (her / "data.txt").write_text("hers\n")
    return her, root


def test_write_into_her_fails_inside_and_works_outside():
    """K0：命名空间里改她的文件失败（只读），同一个脚本在外面跑能改（对照，证明测试能看出差别）；她的文件可读。"""
    if _skip():
        return
    with tempfile.TemporaryDirectory() as d:
        her, root = _dirs(d)
        script = root / "w.sh"   # 写入的目标不出现在命令行里，只有挂载能挡住
        script.write_text(f"cat '{her}/data.txt' > read.txt && echo mine >> '{her}/data.txt'\n")
        p = _run(her, root, "bash", "w.sh")
        assert p.returncode != 0, (p.stdout, p.stderr)
        assert (her / "data.txt").read_text() == "hers\n"
        assert (root / "read.txt").read_text() == "hers\n"   # 她的文件读得到，我们的根写得进
        q = subprocess.run(["bash", str(script)], cwd=root, capture_output=True, text=True, check=False)
        assert q.returncode == 0 and (her / "data.txt").read_text() == "hers\nmine\n", (q.stdout, q.stderr)


def test_arguments_into_her_are_refused():
    """K0：参数（含 key=value 的值、相对路径）的 realpath 落在她目录下就拒绝，退出码 2，不打印她的路径。"""
    if _skip():
        return
    with tempfile.TemporaryDirectory() as d:
        her, root = _dirs(d)
        for args in ([ "cat", str(her / "data.txt")], ["env", f"X={her}/data.txt", "true"], ["cat", "../her/data.txt"]):
            p = _run(her, root, *args)
            assert p.returncode == 2 and p.stdout.startswith("T3R_REFUSED"), (args, p.stdout, p.stderr)
            assert str(her) not in p.stdout


def test_allowlisted_interpreter_runs():
    """K0：白名单里的程序（按 realpath 比）即使在她目录下也放行，并且在只读命名空间里跑。"""
    if _skip():
        return
    with tempfile.TemporaryDirectory() as d:
        her, root = _dirs(d)
        exe = her / "mytrue"
        shutil.copy2(shutil.which("true"), exe)
        p = _run(her, root, str(exe), allow=str(exe))
        assert p.returncode == 0, (p.stdout, p.stderr)
        p = _run(her, root, str(exe))
        assert p.returncode == 2, (p.stdout, p.stderr)


def test_root_under_her_is_refused():
    """K0：我们的根不能在她的目录下。"""
    if _skip():
        return
    with tempfile.TemporaryDirectory() as d:
        her, _ = _dirs(d)
        inner = her / "inner"
        inner.mkdir()
        p = _run(her, inner, "true")
        assert p.returncode == 2 and "根在她的目录下" in p.stdout, (p.stdout, p.stderr)



def test_mask_hides_her_dirs():
    """1b 复核改动第 1 条：T3R_MASK 盖住的她的子目录在命名空间里是空的（里面的解释器跑不起来）；不盖时同一个脚本看得见（对照）；
    盖她目录以外的东西要拒绝。"""
    if _skip():
        return
    with tempfile.TemporaryDirectory() as d:
        her, root = _dirs(d)
        sub = her / "venv" / "bin"
        sub.mkdir(parents=True)
        shutil.copy2(shutil.which("true"), sub / "python")
        script = root / "c.sh"   # 路径写在脚本里，不出现在命令行上
        script.write_text(f"test -x '{sub}/python' && '{sub}/python'\n")
        p = _run(her, root, "bash", "c.sh", mask=str(her / "venv"))
        assert p.returncode != 0, (p.stdout, p.stderr)
        p = _run(her, root, "bash", "c.sh")
        assert p.returncode == 0, (p.stdout, p.stderr)
        p = _run(her, root, "true", mask=str(root))
        assert p.returncode == 2 and "T3R_MASK" in p.stdout, (p.stdout, p.stderr)


def test_allowlisted_exe_is_logged():
    """1b 复核改动第 1 条：白名单里的程序在第 2 级打出 realpath 与 sha256，记录里看得到实际用的是谁。"""
    if _skip():
        return
    with tempfile.TemporaryDirectory() as d:
        her, root = _dirs(d)
        exe = root / "mytrue"
        shutil.copy2(shutil.which("true"), exe)
        p = _run(her, root, str(exe), allow=str(exe))
        assert p.returncode == 0 and f"T3R_EXE {exe.resolve()} sha256 " in p.stderr, (p.stdout, p.stderr)


if __name__ == "__main__":
    if _skip():
        print("T3R_RUN_TESTS_SKIPPED（不是 Linux 或没有 unshare / setpriv）")
        sys.exit(0)
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name} 过")
    print("T3R_RUN_TESTS_OK")
