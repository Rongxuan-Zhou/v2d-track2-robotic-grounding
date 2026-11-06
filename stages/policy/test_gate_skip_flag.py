"""gate_skip_flag.sh 的测试（暂无契约 ID；计划 agent-doc/plan-c10-extend-2026-10-16.md 第 3 版）。
  1. 一开始就有标记：立刻不放行；
  2. 一直没有标记：等满时间放行（0 分钟立刻放行）；
  3. 等的过程中标记出现：不放行；
  4. 分钟写法非法：退出码 2。
用法：python3 stages/policy/test_gate_skip_flag.py；最后一行 ALL_OK。
"""
import subprocess
import tempfile
import threading
import time
from pathlib import Path

GATE = Path(__file__).resolve().parent / "gate_skip_flag.sh"


def run(flag, minutes, poll="0.2"):
    return subprocess.run(["bash", str(GATE), str(flag), str(minutes)], capture_output=True, text=True, check=False,
                          env={"PATH": "/usr/bin:/bin", "V2D_GATE_POLL_S": poll})


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / ".t3ps-skip"
        f.write_text("")
        t0 = time.time(); r = run(f, 25)
        assert r.returncode == 1 and r.stdout.strip().startswith("GATE_SKIP_STOP") and time.time() - t0 < 3, r.stdout
        print("  1. 一开始就有标记：立刻不放行")
        f.unlink()
        r = run(f, 0)
        assert r.returncode == 0 and r.stdout.strip().startswith("GATE_SKIP_PASS"), r.stdout
        print("  2. 没有标记：放行")
        threading.Timer(1.0, lambda: f.write_text("")).start()
        t0 = time.time(); r = run(f, 1)
        assert r.returncode == 1 and "GATE_SKIP_STOP" in r.stdout and time.time() - t0 < 10, r.stdout
        print("  3. 等的过程中出现标记：不放行")
        assert run(f, "x").returncode == 2 and run(f, "-1").returncode == 2
        print("  4. 分钟写法非法：退出码 2")
    print("ALL_OK")
