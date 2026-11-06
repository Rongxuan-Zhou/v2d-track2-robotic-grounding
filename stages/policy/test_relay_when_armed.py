"""relay_when_armed.sh（计划 agent-doc/plan-t1r3o-2026-10-31.md 第 3 版）：前一条接力日志末行 ARMED 才跑命令，STOPPED 退出 1、
"arm_after:" 报错行也退出 1、超时退出 3、命令照原样 exec（退出码传出）。只用标准库，python3 直接跑。"""
import subprocess
import tempfile
import threading
import time
from pathlib import Path

SH = Path(__file__).resolve().parent / "relay_when_armed.sh"


def run(log, *cmd, polls="5"):
    env = {"PATH": "/usr/bin:/bin", "V2D_POLL_S": "0.2", "V2D_RELAY_MAX_POLLS": polls}
    return subprocess.run(["bash", str(SH), str(log), *cmd], capture_output=True, text=True, env=env, timeout=30)


def test_armed_later_runs_command():
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "a.log"
        log.write_text("=== x 等\n", encoding="utf-8")
        out = Path(d) / "ran"
        t = threading.Timer(0.5, lambda: log.write_text("=== x 等\n=== 2026-11-04 04:30:00 CST ARM_AFTER_ARMED\n", encoding="utf-8"))
        t.start()
        r = run(log, "bash", "-c", f"echo ok > {out}; exit 7", polls="50")
        t.join()
        assert r.returncode == 7 and out.read_text().strip() == "ok", (r.returncode, r.stdout, r.stderr)


def test_stopped_and_timeout():
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "a.log"
        out = Path(d) / "ran"
        log.write_text("=== y ARM_AFTER_STOPPED 预演没过\n", encoding="utf-8")
        r = run(log, "touch", str(out))
        assert r.returncode == 1 and not out.exists(), (r.returncode, r.stdout)
        log.write_text("arm_after: 暂存目录不可读：runs/stage/x\n", encoding="utf-8")
        r = run(log, "touch", str(out))
        assert r.returncode == 1 and not out.exists(), (r.returncode, r.stdout)
        log.write_text("=== x 等\n", encoding="utf-8")
        r = run(log, "touch", str(out), polls="3")
        assert r.returncode == 3 and not out.exists(), (r.returncode, r.stdout)
        r = run(Path(d) / "missing.log", "touch", str(out), polls="2")
        assert r.returncode == 3 and not out.exists(), (r.returncode, r.stdout)


def test_relay_on_relay_waits():
    """前一条也是 relay_when_armed.sh：它的等待行、“已起链”行里都有 ARM_AFTER_ARMED 这个词，不能当成已起链（11/03 final1）。"""
    with tempfile.TemporaryDirectory() as d:
        log = Path(d) / "a.log"
        out = Path(d) / "ran"
        log.write_text("=== 2026-11-04 04:25:55 CST 等 b.log 的最后一行变成 ARM_AFTER_ARMED，再跑：bash arm_after.sh x\n", encoding="utf-8")
        r = run(log, "touch", str(out), polls="3")
        assert r.returncode == 3 and not out.exists(), (r.returncode, r.stdout)
        log.write_text("=== 2026-11-04 04:25:55 CST 前一条接力已起链：=== 2026-11-04 04:25:50 CST ARM_AFTER_ARMED\n", encoding="utf-8")
        r = run(log, "touch", str(out), polls="3")
        assert r.returncode == 3 and not out.exists(), (r.returncode, r.stdout)
        log.write_text("=== 2026-11-04 04:30:00 CST ARM_AFTER_ARMED\n", encoding="utf-8")
        r = run(log, "touch", str(out), polls="3")
        assert r.returncode == 0 and out.exists(), (r.returncode, r.stdout)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("ALL_OK")
