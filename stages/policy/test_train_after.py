"""train_after.sh 的行为测试：守 agent-doc/plan-t1-r3x-2026-10-08.md 第 7 节（上一条训练链正常结束才开下一条；暂无契约 ID）。只用标准库，Mac 与训练机都能跑。

临时目录里放真的 train_after.sh 与一个假的 day2_train_chain.sh（只把收到的参数、V2D_FC_SRC 和当时的长链标记记进文件）。轮询间隔 1 秒。
  1. 上一条链 DAY2_TRAIN_DONE：开训练链，5 个参数与环境变量原样传过去；开链时等待标记已删，结束后也不留
  2. DAY2_TRAIN_FAILED：不开，新日志最后一行 DAY2_TRAIN_FAILED，其余行不出现结束标记原文，退出码 1，标记删掉
  3. 上一条链还没结束时一直等（期间有等待标记），结束后才判
  4. 新日志已存在：拒绝（退出码 2），不开、不写标记
  5. 参数个数不对：拒绝（退出码 2）
用法：python3 stages/policy/test_train_after.py
"""
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAKE = ('#!/usr/bin/env bash\n'
        'echo "CHAIN $* FSRC=${V2D_FC_SRC:-} MARK=$(ls runs/.chain-active-* 2>/dev/null | tr \'\\n\' \' \')" >> chain_calls.txt\n')
ARGS = ["e.tsv", "s.txt", "q.txt", "24564", "x2_train"]
DONE = "=== 2026-10-09 02:20:00 CST DAY2_TRAIN_DONE（完成 5、失败 1、跳过 0、发散 0）"
FAILED = "=== 2026-10-09 02:20:00 CST DAY2_TRAIN_FAILED"


def run(prev_last: str | None, delay: float = 0.0, args=None, pre_log: bool = False):
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        (root / "stages/policy").mkdir(parents=True)
        (root / "runs/logs").mkdir(parents=True)
        shutil.copy(HERE / "train_after.sh", root / "stages/policy")
        (root / "stages/policy/day2_train_chain.sh").write_text(FAKE)
        (root / "stages/policy/day2_train_chain.sh").chmod(0o755)
        prev = root / "runs/logs/x_train.log"
        prev.write_text("=== x 正式 t1-ep5\n", encoding="utf-8")
        if pre_log:
            (root / "runs/logs/x2_train.log").write_text("old\n", encoding="utf-8")
        seen = {}
        if prev_last is not None:
            if delay:
                def finish():
                    seen["mark_while_waiting"] = (root / "runs/.chain-active-x2_train-wait").exists()
                    with prev.open("a", encoding="utf-8") as f:
                        f.write(prev_last + "\n")
                threading.Timer(delay, finish).start()
            else:
                with prev.open("a", encoding="utf-8") as f:
                    f.write(prev_last + "\n")
        t0 = time.time()
        env = dict(os.environ, V2D_POLL_S="1", V2D_FC_SRC="/src/nfg")
        r = subprocess.run(["bash", str(root / "stages/policy/train_after.sh"), "runs/logs/x_train.log", *(ARGS if args is None else args)],
                           cwd=root, env=env, capture_output=True, text=True, errors="replace", timeout=60)
        calls = (root / "chain_calls.txt").read_text() if (root / "chain_calls.txt").exists() else ""
        log = root / "runs/logs/x2_train.log"
        newlog = log.read_text(encoding="utf-8") if log.exists() else ""
        marks = sorted(p.name for p in (root / "runs").glob(".chain-active-*"))
        return r.returncode, calls, newlog, marks, time.time() - t0, seen


def test_starts_chain_after_done():
    rc, calls, newlog, marks, _, _ = run(DONE)
    assert rc == 0 and calls.strip() == "CHAIN e.tsv s.txt q.txt 24564 x2_train FSRC=/src/nfg MARK=", (rc, calls)
    assert not marks and not newlog, (marks, newlog)


def test_failed_prev_chain_stops():
    rc, calls, newlog, marks, _, _ = run(FAILED)
    lines = newlog.strip().splitlines()
    assert rc == 1 and not calls and not marks, (rc, calls, marks)
    assert lines[-1].endswith("DAY2_TRAIN_FAILED") and all("DAY2_TRAIN_" not in s for s in lines[:-1]), newlog
    assert "训练链结束" in newlog, newlog


def test_waits_for_prev_end():
    rc, calls, _, marks, took, seen = run(DONE, delay=3)
    assert rc == 0 and calls and took >= 3 and seen.get("mark_while_waiting") is True and not marks, (rc, calls, took, seen, marks)


def test_existing_log_refuses():
    rc, calls, newlog, marks, _, _ = run(DONE, pre_log=True)
    assert rc == 2 and not calls and not marks and newlog == "old\n", (rc, calls, marks, newlog)


def test_wrong_arg_count_refuses():
    rc, calls, _, marks, _, _ = run(DONE, args=ARGS[:4])
    assert rc == 2 and not calls and not marks, (rc, calls, marks)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("ALL_OK")
