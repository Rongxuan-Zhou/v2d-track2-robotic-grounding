"""stages/evaluate/t3g5_r2_gate.sh 的测试：守计划 agent-doc/plan-t3g5-2026-10-21.md 判据 R2（A 段 7 集里 ≥ 5 集没跟上就停 B 段的等待，否则放行）。
在临时目录里扮仓库根：假的 R1 记录、一个命令行像 runs/tools/arm_after.sh <B 段> 的假等待进程。只在 Linux 上跑（要 pgrep -f）；最后一行 T3G5_R2_GATE_TESTS_OK。
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

GATE = Path(__file__).resolve().parent / "t3g5_r2_gate.sh"


def _skip() -> bool:
    return sys.platform != "linux" or shutil.which("pgrep") is None


def _setup(d: Path, nxt: str) -> subprocess.Popen:
    (d / "runs/tools").mkdir(parents=True)
    (d / "results/t3g5").mkdir(parents=True)
    w = d / "runs/tools/arm_after.sh"
    w.write_text("sleep 300\n")
    return subprocess.Popen(["bash", str(w.relative_to(d)), nxt, "x"], cwd=d)


def _gate(d: Path, *args: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, V2D_GATE_ROOT=str(d), V2D_POLL_S="1", V2D_GATE_MAX_S="20")
    return subprocess.run(["bash", str(GATE), *args], env=env, capture_output=True, text=True, timeout=60, check=False)


def _write_r1_later(d: Path, seg: str, mach: str, k: int, n: int) -> None:
    subprocess.Popen(["bash", "-c", f"sleep 2; echo 'T3G5_R1_DONE 跟上 {k} / 共 {n}' > results/t3g5/eval_{seg}_{mach}.txt"], cwd=d)


def test_below_threshold_releases_hold_and_keeps_waiter():
    """R2 不成立（没跟上 2 + 1 = 3 < 5）：闸文件建过又撤掉，B 段的等待进程还在。"""
    if _skip():
        return
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        waiter = _setup(d, "segB")
        try:
            _write_r1_later(d, "segA", "m1", 3, 4)
            p = _gate(d, "segA", "m1", "2", "5", "segB")
            assert p.returncode == 0 and "R2 不成立" in p.stdout, (p.stdout, p.stderr)
            assert not (d / "runs/.chain-active-t3g5r2-segB").exists()
            assert waiter.poll() is None
        finally:
            waiter.kill()


def test_threshold_stops_only_the_named_waiter():
    """R2 成立（没跟上 2 + 3 = 5 ≥ 5）：只停 B 段那个等待进程，别的段的同名脚本不碰，闸撤掉。"""
    if _skip():
        return
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        waiter = _setup(d, "segB")
        other = subprocess.Popen(["bash", "runs/tools/arm_after.sh", "segC", "x"], cwd=d)
        try:
            time.sleep(0.5)
            _write_r1_later(d, "segA", "m1", 1, 4)
            p = _gate(d, "segA", "m1", "2", "5", "segB")
            assert p.returncode == 0 and "R2 成立" in p.stdout, (p.stdout, p.stderr)
            assert waiter.wait(timeout=10) is not None
            assert other.poll() is None
            assert not (d / "runs/.chain-active-t3g5r2-segB").exists()
        finally:
            for x in (waiter, other):
                if x.poll() is None:
                    x.kill()


def test_readout_without_r1_keeps_hold():
    """读数做完却没有 R1 结论：闸留着、报 FAILED，不猜。"""
    if _skip():
        return
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        waiter = _setup(d, "segB")
        try:
            subprocess.Popen(["bash", "-c", "sleep 2; echo x > results/t3g5/readout_segA_m1.txt.failed"], cwd=d)
            p = _gate(d, "segA", "m1", "2", "5", "segB")
            assert p.returncode == 1 and "T3G5_R2_GATE_FAILED" in p.stdout, (p.stdout, p.stderr)
            assert (d / "runs/.chain-active-t3g5r2-segB").exists()
            assert waiter.poll() is None
        finally:
            waiter.kill()


if __name__ == "__main__":
    if _skip():
        print("T3G5_R2_GATE_TESTS_SKIPPED（不是 Linux 或没有 pgrep）")
        sys.exit(0)
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name} 过")
    print("T3G5_R2_GATE_TESTS_OK")
