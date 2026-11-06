"""gate_verdict.sh 的测试（暂无契约 ID；计划 agent-doc/plan-c10-extend-2026-10-16.md 路 A，独立复核 MINOR 3）。
  1. 最后一行是允许的总判就放行（有用、看不清）；
  2. 最后一行是“没用”，即使正文里写着“有用”也不放行；
  3. 只有 .failed 时立刻不放行；文件一直不出现时等满次数后不放行；文件晚到时等到再判；
  4. 允许的总判写法非法时退出码 2。
用法：python3 stages/policy/test_gate_verdict.py；最后一行 ALL_OK。
"""
import subprocess
import tempfile
import threading
import time
from pathlib import Path

GATE = Path(__file__).resolve().parent / "gate_verdict.sh"


def run(f, ok="有用|看不清", polls="3", poll_s="0"):
    return subprocess.run(["bash", str(GATE), str(f), ok], capture_output=True, text=True, check=False,
                          env={"PATH": "/usr/bin:/bin", "V2D_GATE_POLLS": polls, "V2D_GATE_POLL_S": poll_s})


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "verdict_C1_ml40.txt"
        body = "# command  : python3 stages/policy/r3o_verdict.py C1\n总判：修好 ≥ 1 集记有用；有起色 ≥ 2 集记看不清\n"
        for v in ("有用", "看不清"):
            f.write_text(body + f"R3O_VERDICT {v}\n")
            r = run(f)
            assert r.returncode == 0 and r.stdout.strip().splitlines()[-1] == f"GATE_VERDICT_PASS {v}", r.stdout + r.stderr
        print("  1. 有用、看不清放行")
        f.write_text(body + "R3O_VERDICT 没用\n")
        r = run(f)
        assert r.returncode == 1 and "GATE_VERDICT_STOP 总判是：R3O_VERDICT 没用" in r.stdout, r.stdout + r.stderr
        f.write_text(body + "R3O_VERDICT 有用 \n")
        assert run(f).returncode == 1   # 整行匹配，多一个空格也不放行
        print("  2. 没用不放行，正文里的“有用”不算")
        f.unlink()
        Path(str(f) + ".failed").write_text("x\n")
        t0 = time.time(); r = run(f, polls="1000", poll_s="1")
        assert r.returncode == 1 and "判读失败" in r.stdout and time.time() - t0 < 5, r.stdout
        Path(str(f) + ".failed").unlink()
        r = run(f, polls="2", poll_s="0")
        assert r.returncode == 1 and "等了 2 次还没有" in r.stdout, r.stdout
        threading.Timer(1.0, lambda: f.write_text(body + "R3O_VERDICT 有用\n")).start()
        r = run(f, polls="20", poll_s="0.5")
        assert r.returncode == 0 and "GATE_VERDICT_PASS 有用" in r.stdout, r.stdout + r.stderr
        print("  3. .failed 立刻停；一直没有就等满次数；晚到的等到再判")
        for bad in ("有用;rm", "有用 看不清", "$(x)"):
            assert run(f, ok=bad).returncode == 2, bad
        print("  4. 允许的总判写法非法退出码 2")
    print("ALL_OK")
