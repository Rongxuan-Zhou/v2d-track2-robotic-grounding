"""export_after_eval.sh 的行为测试：守 agent-doc/plan-t1-r3x-2026-10-08.md 第 3 节（评测链成功结束才导出；暂无契约 ID）。只用标准库，Mac 与训练机都能跑。

临时目录里放真的 export_after_eval.sh 与一个假的 export_chain.sh（只把收到的参数和 V2D_FALLBACK_MAX 记进文件，并看长链标记在不在）。轮询间隔 1 秒。
  1. 评测链 DAY2_EVAL_DONE（成功 6、失败 0）：开导出链，参数与环境变量原样传过去，等待期间有长链标记，结束后标记删掉
  2. DAY2_EVAL_DONE（成功 5、失败 1）：不导出，门日志最后一行 EXPORT_GATE_STOPPED，退出码 1
  3. DAY2_EVAL_FAILED：不导出，同上
  4. 评测链还没结束时一直等，结束后才判
  5. 门日志里不出现评测链结束标记的原文
用法：python3 stages/evaluate/test_export_after_eval.py
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
        'echo "EXPORT $* FMAX=${V2D_FALLBACK_MAX:-} MARK=$(ls runs/.chain-active-* 2>/dev/null | tr \'\\n\' \' \')" >> export_calls.txt\n')


def run(eval_last: str, delay: float = 0.0):
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        (root / "stages/evaluate").mkdir(parents=True)
        (root / "runs/logs").mkdir(parents=True)
        shutil.copy(HERE / "export_after_eval.sh", root / "stages/evaluate")
        (root / "stages/evaluate/export_chain.sh").write_text(FAKE)
        (root / "stages/evaluate/export_chain.sh").chmod(0o755)
        ev = root / "runs/logs/x_eval.log"
        ev.write_text("=== x 等训练链\n", encoding="utf-8")
        seen = {}
        if delay:
            def finish():
                seen["mark_while_waiting"] = (root / "runs/.chain-active-cand_x-gate").exists()
                ev.write_text(f"=== x 等训练链\n=== y {eval_last}\n", encoding="utf-8")
            threading.Timer(delay, finish).start()
        else:
            ev.write_text(f"=== y {eval_last}\n", encoding="utf-8")
        t0 = time.time()
        env = dict(os.environ, V2D_POLL_S="1", V2D_FALLBACK_MAX="0")
        r = subprocess.run(["bash", str(root / "stages/evaluate/export_after_eval.sh"), "runs/logs/x_eval.log", "list.tsv", "cand_x"],
                           cwd=root, env=env, capture_output=True, text=True, errors="replace", timeout=60)
        calls = (root / "export_calls.txt").read_text() if (root / "export_calls.txt").exists() else ""
        glog = (root / "runs/logs/cand_x_gate.log").read_text(encoding="utf-8")
        mark_after = (root / "runs/.chain-active-cand_x-gate").exists()
        return r.returncode, calls, glog, mark_after, time.time() - t0, seen


def test_exports_after_clean_eval():
    rc, calls, glog, mark_after, _, _ = run("DAY2_EVAL_DONE（成功 6、失败 0）")
    assert rc == 0 and calls.strip() == "EXPORT list.tsv cand_x FMAX=0 MARK=runs/.chain-active-cand_x-gate", (rc, calls)
    assert not mark_after and "评测链成功 6 条、失败 0" in glog, glog


def test_partial_eval_failure_stops():
    rc, calls, glog, mark_after, _, _ = run("DAY2_EVAL_DONE（成功 5、失败 1）")
    assert rc == 1 and not calls and glog.strip().splitlines()[-1].endswith("EXPORT_GATE_STOPPED") and not mark_after, (rc, calls, glog)


def test_failed_eval_chain_stops():
    rc, calls, glog, _, _, _ = run("DAY2_EVAL_FAILED")
    assert rc == 1 and not calls and "评测链结束" in glog and glog.strip().splitlines()[-1].endswith("EXPORT_GATE_STOPPED"), (rc, calls, glog)


def test_waits_for_eval_end():
    rc, calls, glog, _, took, seen = run("DAY2_EVAL_DONE（成功 6、失败 0）", delay=3)
    assert rc == 0 and calls and took >= 3 and seen.get("mark_while_waiting") is True, (rc, calls, took, seen)


def test_no_raw_end_marker_in_gate_log():
    for last in ("DAY2_EVAL_DONE（成功 5、失败 1）", "DAY2_EVAL_FAILED", "DAY2_EVAL_DONE（成功 6、失败 0）"):
        _, _, glog, _, _, _ = run(last)
        assert "DAY2_EVAL_" not in glog, glog


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("ALL_OK")
