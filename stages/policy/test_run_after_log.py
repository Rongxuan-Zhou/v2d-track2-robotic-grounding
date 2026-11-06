"""run_after_log.sh 的行为测试（暂无契约 ID：运行规范待成文）。只用标准库，Mac 与训练机都能跑。
临时目录里搭最小仓库：真的 run_after_log.sh，假的 scripts/dev/record（记下收到的记录名与命令，退出码按 FAKE_RC）。
  1. 日志一开始不是结束行，过一会儿追加结束行：脚本等到匹配后跑 record，最后一行 RUN_AFTER_DONE 0；
  2. 一直不匹配：等满 V2D_MAX_POLLS 次后 RUN_AFTER_TIMEOUT，退出码 1，没有调 record；
  3. record 失败（退出码 3）：最后一行 RUN_AFTER_DONE 3，脚本退出码 3；日志不存在时先打警告；
  4. 记录名带 .. 或以 / 开头：退出码 2，不等。
用法：python3 stages/policy/test_run_after_log.py；最后一行 ALL_OK。
"""
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAKE_RECORD = '#!/usr/bin/env bash\necho "$*" >> "$FAKE_CALLS"\nexit "${FAKE_RC:-0}"\n'


def setup(root: Path) -> None:
    (root / "stages/policy").mkdir(parents=True)
    (root / "scripts/dev").mkdir(parents=True)
    (root / "runs/logs").mkdir(parents=True)
    shutil.copy(HERE / "run_after_log.sh", root / "stages/policy")
    (root / "scripts/dev/record").write_text(FAKE_RECORD)
    (root / "scripts/dev/record").chmod(0o755)


def run(root: Path, *args, **env) -> tuple:
    calls = root / "calls.txt"
    e = dict(os.environ, FAKE_CALLS=str(calls), V2D_POLL_S="1")
    e.update(env)
    r = subprocess.run(["bash", str(root / "stages/policy/run_after_log.sh"), *args], cwd=root, env=e, capture_output=True,
                       text=True, check=False, timeout=60)
    return r.returncode, r.stdout + r.stderr, calls.read_text() if calls.exists() else ""


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root)
        log = root / "runs/logs/x_eval.log"
        log.write_text("=== start\n")

        def finish() -> None:
            time.sleep(2.5)
            with log.open("a") as f:
                f.write("=== 2026-10-17 03:40:00 CST DAY2_EVAL_DONE（成功 5、失败 0）\n")

        threading.Thread(target=finish).start()
        rc, text, calls = run(root, "runs/logs/x_eval.log", "DAY2_EVAL_(DONE|FAILED)", "r3o/verdict_P2_ml40",
                              "python3", "stages/policy/r3o_verdict.py", "P2", V2D_MAX_POLLS="20")
        assert rc == 0 and text.strip().splitlines()[-1] == "RUN_AFTER_DONE 0", text
        assert calls.strip() == "r3o/verdict_P2_ml40 python3 stages/policy/r3o_verdict.py P2", calls
    print("  1. 等到结束行再跑 record，RUN_AFTER_DONE 0")
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root)
        (root / "runs/logs/x_eval.log").write_text("=== still running\n")
        rc, text, calls = run(root, "runs/logs/x_eval.log", "DAY2_EVAL_DONE", "r3o/x", "true", V2D_MAX_POLLS="2")
        assert rc == 1 and text.strip().splitlines()[-1] == "RUN_AFTER_TIMEOUT" and not calls, (rc, text, calls)
    print("  2. 一直不匹配：RUN_AFTER_TIMEOUT，不调 record")
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root)
        (root / "runs/logs/x_eval.log").write_text("DAY2_EVAL_DONE\n")
        rc, text, calls = run(root, "runs/logs/x_eval.log", "DAY2_EVAL_DONE", "r3o/x", "true", FAKE_RC="3", V2D_MAX_POLLS="2")
        assert rc == 3 and text.strip().splitlines()[-1] == "RUN_AFTER_DONE 3", text
    print("  3. record 失败：RUN_AFTER_DONE 3，退出码 3")
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root)
        rc, text, calls = run(root, "runs/logs/nope.log", "DONE", "r3o/x", "true", V2D_MAX_POLLS="1")
        assert rc == 1 and "现在还不存在" in text and not calls, (rc, text)
    print("  3b. 日志不存在：先打警告，超时退出码 1")
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root)
        for bad in ("../x", "/abs/x"):
            rc, text, calls = run(root, "runs/logs/x_eval.log", "DONE", bad, "true", V2D_MAX_POLLS="1")
            assert rc == 2 and not calls, (bad, rc, text)
    print("  4. 记录名非法：退出码 2")
    print("ALL_OK")
