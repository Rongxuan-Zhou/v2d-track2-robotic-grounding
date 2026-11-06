"""chain_after.sh 的行为测试（暂无契约 ID：运行规范待成文）。只用标准库，Mac 与训练机都能跑。

临时目录里放真的 chain_after.sh 与一个假的 day2_train_chain.sh（只把收到的参数和 V2D_PRUNE_SMOKE_STATE 记进文件）。轮询间隔 1 秒。
  1. 上一段评完、训练链 DAY2_TRAIN_DONE 且完成 5 条（门槛 3）：开下一段，参数与环境变量原样传过去；
  2. 完成 2 条：不开，下一段日志最后一行是 DAY2_TRAIN_FAILED，退出码 1；
  3. 上一段训练链本身 DAY2_TRAIN_FAILED：不开；
  4. 评测链还没结束时一直等，结束后才判；
  5. 设了 V2D_GATE_CMD：判定命令退出码 0 才开，输出记进下一段日志；非 0 不开，最后一行 DAY2_TRAIN_FAILED，退出码 1。
用法：python3 stages/policy/test_chain_after.py
"""
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAKE_CHAIN = '#!/usr/bin/env bash\necho "CHAIN $* PRUNE=${V2D_PRUNE_SMOKE_STATE:-}" >> chain_calls.txt\n'
ARGS = ["e.tsv", "s.txt", "f.txt", "24564", "t2b_train_x"]


def run(train_last: str, eval_last, delay_eval: float = 0.0, gate=None) -> tuple:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        (root / "stages/policy").mkdir(parents=True)
        (root / "runs/logs").mkdir(parents=True)
        shutil.copy(HERE / "chain_after.sh", root / "stages/policy")
        (root / "stages/policy/day2_train_chain.sh").write_text(FAKE_CHAIN)
        (root / "stages/policy/day2_train_chain.sh").chmod(0o755)
        (root / "runs/logs/a_train.log").write_text("=== x 正式 t2-ep0\n" + train_last + "\n", encoding="utf-8")
        ev = root / "runs/logs/a_eval.log"
        ev.write_text("=== x 等训练链\n", encoding="utf-8")
        if delay_eval:
            threading.Timer(delay_eval, lambda: ev.write_text(f"=== x {eval_last}\n", encoding="utf-8")).start()
        else:
            ev.write_text(f"=== x {eval_last}\n", encoding="utf-8")
        t0 = time.time()
        env = dict(os.environ, V2D_POLL_S="1", V2D_PRUNE_SMOKE_STATE="1")
        env.pop("V2D_GATE_CMD", None)
        if gate is not None:
            env["V2D_GATE_CMD"] = gate
        r = subprocess.run(["bash", str(root / "stages/policy/chain_after.sh"), "runs/logs/a_train.log", "runs/logs/a_eval.log", "3", *ARGS],
                           cwd=root, env=env, capture_output=True, text=True, errors="replace", timeout=60)
        took = time.time() - t0
        calls = (root / "chain_calls.txt").read_text() if (root / "chain_calls.txt").exists() else ""
        blog = (root / "runs/logs/t2b_train_x.log").read_text(encoding="utf-8") if (root / "runs/logs/t2b_train_x.log").exists() else ""
        return r.returncode, calls, blog, took


def test_opens_next_segment() -> None:
    rc, calls, blog, _ = run("=== 2026-10-01 12:00:00 CST DAY2_TRAIN_DONE（完成 5、失败 0、跳过 0、发散 0）", "DAY2_EVAL_DONE（成功 5、失败 0）")
    assert rc == 0 and calls.strip() == "CHAIN e.tsv s.txt f.txt 24564 t2b_train_x PRUNE=1", (rc, calls)
    assert "上一段完成 5 条" in blog, blog


def test_too_few_done() -> None:
    rc, calls, blog, _ = run("=== x DAY2_TRAIN_DONE（完成 2、失败 1、跳过 0、发散 2）", "DAY2_EVAL_DONE（成功 2、失败 0）")
    assert rc == 1 and not calls, (rc, calls)
    assert blog.rstrip().splitlines()[-1].endswith("DAY2_TRAIN_FAILED") and "完成 ≥ 3" in blog, blog


def test_previous_chain_failed() -> None:
    rc, calls, blog, _ = run("=== x 磁盘只剩 900 MB（要求至少 2048 MB），停下\n=== x DAY2_TRAIN_FAILED", "DAY2_EVAL_DONE（成功 3、失败 0）")
    assert rc == 1 and not calls and blog.rstrip().endswith("DAY2_TRAIN_FAILED"), (rc, calls, blog)


def test_waits_for_eval() -> None:
    rc, calls, blog, took = run("=== x DAY2_TRAIN_DONE（完成 4、失败 1、跳过 0、发散 0）", "DAY2_EVAL_FAILED", delay_eval=3)
    assert rc == 0 and "CHAIN" in calls and took >= 3, (rc, calls, took)


def test_gate_continue() -> None:
    rc, calls, blog, _ = run("=== x DAY2_TRAIN_DONE（完成 6、失败 0、跳过 0、发散 0）", "DAY2_EVAL_DONE（成功 6、失败 0）",
                             gate='echo "MIDCHECK CONTINUE 平均 +0.12"; exit 0')
    assert rc == 0 and "CHAIN" in calls, (rc, calls)
    assert "退出码 0" in blog and "MIDCHECK CONTINUE" in blog and "上一段完成 6 条" in blog, blog


def test_gate_stop() -> None:
    rc, calls, blog, _ = run("=== x DAY2_TRAIN_DONE（完成 6、失败 0、跳过 0、发散 0）", "DAY2_EVAL_DONE（成功 6、失败 0）",
                             gate='echo "MIDCHECK STOP 平均 -0.03"; exit 1')
    assert rc == 1 and not calls, (rc, calls)
    assert "MIDCHECK STOP" in blog and "判定不过" in blog and blog.rstrip().splitlines()[-1].endswith("DAY2_TRAIN_FAILED"), blog


def test_gate_not_reached_when_too_few_done() -> None:
    rc, calls, blog, _ = run("=== x DAY2_TRAIN_DONE（完成 2、失败 0、跳过 0、发散 4）", "DAY2_EVAL_DONE（成功 2、失败 0）",
                             gate='echo GATE_RAN; exit 0')
    assert rc == 1 and not calls and "GATE_RAN" not in blog, (rc, calls, blog)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("ALL_OK")
