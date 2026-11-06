"""fc_run_watched.sh 的行为测试（暂无契约 ID：运行规范待成文）。只用标准库，Mac 与训练机都能跑。

在临时目录里搭一个最小仓库：真的 fc_run_watched.sh 与 ckpt_nonfinite.py，假的 fc_run.sh（按 FAKE_MODE 写 policy、DONE 或等 STOP）
与假的 fc_stop.sh（只放 STOP 标记）。看守间隔 1 秒、写入余量 0 秒。四种情形：
  1. 正常训完：退出码 0，最新 policy 有限，不写 NONFINITE；
  2. 中途连续两份非有限：看守止损，fc_run 按 STOP 退出 3，包装改成 9；
  3. 训完但最后一份非有限（看守来不及看到）：退出码 10，写 NONFINITE；
  4. fc_run 自己失败：原样返回它的退出码 1；
  5. 真的 fc_stop 要等 docker stop（最多 60 秒）：假 fc_stop 放了 STOP 之后再睡几秒，fc_run 先按 STOP 退出 3、包装收掉看守，
     看守日志里只有“：fc_stop <run>”那一行、没有 NAN_WATCH_STOPPED，包装仍要改成 9；
  6. 看守起不来（ckpt_nonfinite.py 一启动就退出）：退出码 11，不开训练。
每种情形结束后，这个沙箱里不能留下看守进程。
用法：python3 stages/policy/test_fc_run_watched.py
"""
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

MK = r'''
import json, os, struct, sys, time
from array import array
path, val = sys.argv[1], float(sys.argv[2])
data = array("f", [val, 1.0]).tobytes()
head = json.dumps({"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, len(data)]}}).encode()
open(path, "wb").write(struct.pack("<Q", len(head)) + head + data)
old = time.time() - 60
os.utime(path, (old, old))
'''

FAKE_FC_RUN = r'''#!/usr/bin/env bash
RUN=$1; D=runs/flash_chord/$RUN; mkdir -p "$D"
mk() { python3 stages/policy/mk.py "$D/policy_$1.safetensors" "$2"; }
case "$FAKE_MODE" in
  ok)        mk 10 1; mk 20 1; mk 30 1; date > "$D/DONE"; exit 0 ;;
  nan)       mk 10 1; mk 20 nan; mk 30 nan
             for _ in $(seq 60); do [ -f "$D/STOP" ] && { rm -f "$D/STOP"; exit 3; }; sleep 0.5; done; exit 1 ;;
  nan_final) mk 10 1; mk 20 1; mk 30 nan; date > "$D/DONE"; exit 0 ;;
  fail)      mk 10 1; exit 1 ;;
esac
'''

FAKE_FC_STOP = '#!/usr/bin/env bash\ntouch "runs/flash_chord/$1/STOP"\necho "$1" >> fc_stop_calls.txt\nsleep "${FAKE_STOP_SLEEP:-0}"\n'


def sandbox(root: Path) -> None:
    pol = root / "stages/policy"
    pol.mkdir(parents=True)
    shutil.copy(HERE / "fc_run_watched.sh", pol)
    shutil.copy(HERE / "ckpt_nonfinite.py", pol)
    (pol / "mk.py").write_text(MK)
    (pol / "fc_run.sh").write_text(FAKE_FC_RUN)
    (pol / "fc_stop.sh").write_text(FAKE_FC_STOP)
    (pol / "fc_stop.sh").chmod(0o755)
    (root / ".v2d-source-commit").write_text("abc123def456\n")
    (root / "runs/logs").mkdir(parents=True)


def watchers_alive(root: Path) -> list:
    out = subprocess.run(["ps", "-ax", "-o", "pid=,command="], capture_output=True, text=True).stdout
    return [l for l in out.splitlines() if "ckpt_nonfinite.py --watch" in l and str(root) in l]


def case(mode: str, stop_sleep: str = "0", broken_watch: bool = False) -> tuple:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t).resolve()
        sandbox(root)
        if broken_watch:
            (root / "stages/policy/ckpt_nonfinite.py").write_text("import sys\nprint('boom')\nsys.exit(1)\n")
        env = dict(os.environ, FAKE_MODE=mode, V2D_WATCH_INTERVAL="1", V2D_SETTLE_S="0", FAKE_STOP_SLEEP=stop_sleep)
        r = subprocess.run(["bash", str(root / "stages/policy/fc_run_watched.sh"), "rx1", "/repo/p.parquet", "2048", "4096"],
                           cwd=root, env=env, capture_output=True, text=True, errors="replace", timeout=120)
        time.sleep(0.5)
        left = watchers_alive(root)
        time.sleep(float(stop_sleep))  # 等假 fc_stop 睡完再查（它不受包装管）
        nonfinite = (root / "runs/flash_chord/rx1/NONFINITE").exists()
        calls = (root / "fc_stop_calls.txt").read_text().split() if (root / "fc_stop_calls.txt").exists() else []
        wlog = (root / "runs/logs/rx1.nanwatch.log").read_text() if (root / "runs/logs/rx1.nanwatch.log").exists() else ""
        return r.returncode, r.stdout + r.stderr, nonfinite, calls, left, wlog


def test_ok() -> None:
    rc, out, nonfinite, calls, left, _ = case("ok")
    assert rc == 0 and not nonfinite and not calls and not left, (rc, out, calls, left)
    assert "非有限 0 / 2" in out, out


def test_nan_mid_run_is_stopped() -> None:
    rc, out, nonfinite, calls, left, wlog = case("nan")
    assert rc == 9 and calls == ["rx1"] and not left, (rc, out, calls, left)
    assert "NAN_WATCH_STOPPED" in wlog and "src=abc123def456" in wlog, wlog


def test_nan_in_final_policy() -> None:
    rc, out, nonfinite, calls, left, _ = case("nan_final")
    assert rc == 10 and nonfinite and not left, (rc, out, left)


def test_fc_run_failure_passes_through() -> None:
    rc, out, nonfinite, calls, left, _ = case("fail")
    assert rc == 1 and not nonfinite and not calls and not left, (rc, out, calls, left)


def test_stop_label_survives_slow_fc_stop() -> None:
    rc, out, nonfinite, calls, left, wlog = case("nan", stop_sleep="4")
    assert rc == 9 and calls == ["rx1"] and not left, (rc, out, calls, left, wlog)
    assert "：fc_stop rx1" in wlog and "NAN_WATCH_STOPPED" not in wlog, wlog  # 确实走的是“只有 fc_stop 那一行”的分支


def test_watch_that_cannot_start() -> None:
    rc, out, nonfinite, calls, left, _ = case("ok", broken_watch=True)
    assert rc == 11 and "看守没起来" in out and not left, (rc, out)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("ALL_OK")
