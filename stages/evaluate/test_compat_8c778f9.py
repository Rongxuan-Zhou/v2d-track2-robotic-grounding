"""compat_8c778f9.sh（连同 compat_expect.sh、scripts/dev/record、chain_after.sh）的行为测试（暂无契约 ID：运行规范待成文）。只用标准库。

临时目录里搭最小仓库：真的 compat_8c778f9.sh、compat_expect.sh、check_export.sh（经 V2D_COMPAT_PY_IMPORT 走宿主机）、scripts/dev/record、chain_after.sh；假的 fc_export.sh（按 run 名的环境变量
决定退出码，预检模式另有一套）、check_export.py、eval_compare.py、fc_src_check.py、day2_train_chain.sh，PATH 前面放假的 nvidia-smi 与 flock。
chain_after.sh 按计划第 4.4 节的用法接门槛文件，门槛 1。
  1. 全部符合：门槛“完成 1”，COMPAT_PASS，chain_after 开方盒链；
  2. 一条 export 导出失败：门槛“完成 0”，COMPAT_FAIL，不开链；
  3. 导出都成功但一条核对不过、guard 不灵、negctl 的坏 policy 核对通过、negctl 时核对脚本崩溃：都记 COMPAT_FAIL，但门槛仍是“完成 1”，开链；
  4. 副本核不过、GPU 被占、没设 V2D_FC_SRC：DAY2_EVAL_FAILED，门槛“完成 0”，不开链；
  5. 旧日志或旧门槛文件已存在、清单里没有 export 行：退出码 2，不写日志；
  6. 设了 V2D_WAIT_FOR_EVAL_LOG：等那份日志结束才开始；compat 等待期间 chain_after.sh 同时盯着 compat 的日志，不许提前开链
     （10/1 起链时真出过：等待提示里带了结束标记的原文）；compat 日志里只有最后一行带结束标记；
  7. 预检：全对时 COMPAT_DRYRUN_OK、不写日志、门槛与标记；某行 fc_export 预检退出码不对、要等的日志不存在时 COMPAT_DRYRUN_FAIL。
用法：python3 stages/evaluate/test_compat_8c778f9.py（审查 Fable 5.1 留下的原型改写而来）
"""
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
EP1 = "t1-ep1-w2048-cap4m-mf4090"
FAKE_EXPORT = r'''#!/usr/bin/env bash
RUN=$1; STEP=$2; W=${V2D_EXPORT_WORLDS:-1}; K="${RUN//-/_}"
if [ "${V2D_EXPORT_DRYRUN:-0}" = 1 ]; then
  key="FAKE_DRY_RC_$K"; rc="${!key:-0}"
  [ "${V2D_EXPORT_UNGUARDED:-0}" = 0 ] && [ "$RUN" = t1-ep1-w2048-cap4m-mf4090 ] && rc=12
  echo "fake dry $RUN rc $rc"; exit "$rc"
fi
key="FAKE_RC_$K"; rc="${!key:-0}"
if [ "${V2D_EXPORT_UNGUARDED:-0}" = 1 ]; then key2="FAKE_RC_UNGUARDED_$K"; rc="${!key2:-$rc}"; fi
d="runs/flash_chord/$RUN/export_8c778f9/s${STEP}_w${W}"
if [ "$rc" = 0 ]; then mkdir -p "$d"; echo '{"world_count": '"$W"'}' > "$d/episode_000006.json"; echo "fake ok $RUN"; fi
[ "$rc" = 0 ] || echo "fake fail rc $rc" >&2
exit "$rc"
'''
FAKE_CHECK = r'''import os, sys
mode = os.environ.get("FAKE_CHECK_" + sys.argv[1].replace("-", "_"), "ok")
if mode == "crash":
    print("Traceback (most recent call last):"); sys.exit(1)
print("EXPORT_OK" if mode == "ok" else "EXPORT_FAIL 1"); sys.exit(0 if mode == "ok" else 1)
'''
FAKE_COMPARE = 'import os, sys\nrc = int(os.environ.get("FAKE_COMPARE_RC", "0")); print("RESULT CONSISTENT" if rc == 0 else "RESULT INCONSISTENT"); sys.exit(rc)\n'
FAKE_SRC = 'import os, sys\nrc = int(os.environ.get("FAKE_SRC_RC", "0")); print("FC_SRC_OK" if rc == 0 else "FC_SRC_FAIL 1"); sys.exit(rc)\n'
FAKE_CHAIN = '#!/usr/bin/env bash\necho "FAKE_CHAIN_STARTED $*" >> "runs/logs/$5.log"\n'
FAKE_GPU = '#!/usr/bin/env bash\n[ "${FAKE_GPU:-idle}" = busy ] && echo 4242\nexit 0\n'
TSV = ("# 注释行\n"
       "export\tt1-ep8-w2048-cap4m-mf4090\t349999104\t1\n"
       "export\tt1-ep6-w2048-cap4m-mf4090\t349999104\t1\n"
       "export\tt2-ep2-w2048-cap4m-mf4090\t349999104\t1\n"
       "export\tt1-ep8-w2048-cap4m-mf4090\t349999104\t2\n"
       "eval\tt1-ep6-w2048-cap4m-mf4090\t349999104\t4096\n"
       f"guard\t{EP1}\t349999104\t1\n"
       f"negctl\t{EP1}\t349999104\t1\n")
GOOD = {f"FAKE_RC_{EP1.replace('-', '_')}": "12", f"FAKE_RC_UNGUARDED_{EP1.replace('-', '_')}": "13"}


def setup(root: Path, tsv: str = TSV) -> None:
    for d in ("stages/evaluate/compat", "stages/policy", "scripts/dev", "bin", "runs/logs", "third_party/patches"):
        (root / d).mkdir(parents=True, exist_ok=True)
    for f in ("stages/evaluate/compat_8c778f9.sh", "stages/evaluate/compat_expect.sh", "stages/evaluate/check_export.sh", "stages/policy/chain_after.sh",
              "scripts/dev/record"):
        shutil.copy(REPO / f, root / f)
    files = {"stages/evaluate/fc_export.sh": FAKE_EXPORT, "stages/evaluate/check_export.py": FAKE_CHECK, "stages/policy/eval_compare.py": FAKE_COMPARE,
             "stages/policy/fc_src_check.py": FAKE_SRC, "stages/policy/day2_train_chain.sh": FAKE_CHAIN, "bin/nvidia-smi": FAKE_GPU,
             "bin/flock": "#!/usr/bin/env bash\nexit 0\n"}
    for f, text in files.items():
        (root / f).write_text(text)
    for f in ("stages/evaluate/fc_export.sh", "stages/policy/day2_train_chain.sh", "bin/nvidia-smi", "bin/flock",
              "stages/evaluate/compat_8c778f9.sh", "stages/evaluate/compat_expect.sh", "stages/evaluate/check_export.sh", "stages/policy/chain_after.sh",
              "scripts/dev/record"):
        (root / f).chmod(0o755)
    (root / "stages/evaluate/compat/mf.tsv").write_text(tsv)
    (root / ".v2d-source-commit").write_text("deadbeef\n")
    (root / "third_party/patches/flash_chord-8c778f9.sha256").write_text("x  y\n")
    (root / "third_party/patches/flash_chord-8c778f9.eval.json").write_text("{}\n")
    (root / "runs/flash_chord/t1-ep6-w2048-cap4m-mf4090").mkdir(parents=True, exist_ok=True)
    (root / "runs/flash_chord/t1-ep6-w2048-cap4m-mf4090/eval_349999104_w4096.json").write_text("{}")


def env(root: Path, **kw) -> dict:
    e = dict(os.environ, PATH=f"{root / 'bin'}:{os.environ['PATH']}", V2D_FC_SRC="/x/fcsrc", V2D_POLL_S="1", V2D_COMPAT_PY_IMPORT="import json")
    e.update(kw)
    return {k: v for k, v in e.items() if v is not None}


def compat(root: Path, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "stages/evaluate/compat_8c778f9.sh", "stages/evaluate/compat/mf.tsv", "compat_mf"], cwd=root, env=env(root, **kw),
                          capture_output=True, text=True, errors="replace", timeout=120)


def chain_after(root: Path) -> str:
    subprocess.run(["bash", "stages/policy/chain_after.sh", "runs/logs/compat_mf.gate", "runs/logs/compat_mf.log", "1",
                    "e.tsv", "s.txt", "f.txt", "24564", "box_train"], cwd=root, env=env(root), capture_output=True, text=True, timeout=30)
    log = root / "runs/logs/box_train.log"
    return log.read_text() if log.exists() else ""


def outcome(root: Path, **kw) -> tuple:
    r = compat(root, **kw)
    log = (root / "runs/logs/compat_mf.log").read_text()
    gate = (root / "runs/logs/compat_mf.gate").read_text()
    return r.returncode, log, gate, chain_after(root)


def test_all_pass() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root)
        rc, log, gate, box = outcome(root, **GOOD)
        assert rc == 0 and "COMPAT_PASS（7 行全部符合）" in log and log.rstrip().endswith("DAY2_EVAL_DONE（成功 7、失败 0）"), log
        assert sum("DAY2_EVAL_" in line for line in log.splitlines()) == 1, log
        assert "完成 1、失败 0" in gate and "FAKE_CHAIN_STARTED" in box, (gate, box)
        assert not (root / "runs/.chain-active-compat_mf").exists()
        assert (root / "results/submission/compat/mf_negctl_t1-ep1-w2048-cap4m-mf4090_w1.txt").exists()


def test_export_failure_blocks() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root)
        rc, log, gate, box = outcome(root, FAKE_RC_t1_ep6_w2048_cap4m_mf4090="3", **GOOD)
        assert "导出失败（进门槛）" in log and "COMPAT_FAIL" in log and "完成 0、失败 1" in gate, (log, gate)
        assert "FAKE_CHAIN_STARTED" not in box and box.rstrip().endswith("DAY2_TRAIN_FAILED"), box


def test_non_gating_failures_still_open() -> None:
    k = EP1.replace("-", "_")
    for extra in ({"FAKE_CHECK_t1_ep6_w2048_cap4m_mf4090": "fail"}, {f"FAKE_RC_{k}": "0", f"FAKE_RC_UNGUARDED_{k}": "13"},
                  {f"FAKE_RC_{k}": "12", f"FAKE_RC_UNGUARDED_{k}": "0", f"FAKE_CHECK_{k}": "ok"},
                  {f"FAKE_RC_{k}": "12", f"FAKE_RC_UNGUARDED_{k}": "0", f"FAKE_CHECK_{k}": "crash"}, {"FAKE_COMPARE_RC": "1"}):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); setup(root)
            rc, log, gate, box = outcome(root, **{**GOOD, **extra})
            assert "COMPAT_FAIL" in log and "完成 1、失败 0" in gate and "FAKE_CHAIN_STARTED" in box, (extra, log[-800:], gate)


def test_stops() -> None:
    for extra in ({"FAKE_SRC_RC": "1"}, {"FAKE_GPU": "busy"}, {"V2D_FC_SRC": None}):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); setup(root)
            rc, log, gate, box = outcome(root, **{**GOOD, **extra})
            assert rc == 1 and log.rstrip().endswith("DAY2_EVAL_FAILED") and "完成 0、失败 1" in gate, (extra, log)
            assert "FAKE_CHAIN_STARTED" not in box, box


def test_refusals() -> None:
    for stale in ("runs/logs/compat_mf.log", "runs/logs/compat_mf.gate"):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); setup(root)
            (root / stale).write_text("=== x DAY2_EVAL_DONE（成功 1、失败 0）\n")
            r = compat(root, **GOOD)
            assert r.returncode == 2 and "已存在" in r.stderr, (stale, r.stderr)
            assert (root / stale).read_text() == "=== x DAY2_EVAL_DONE（成功 1、失败 0）\n"
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root, tsv="# 只有注释\neval\tt1-ep6-w2048-cap4m-mf4090\t349999104\t4096\n")
        r = compat(root, **GOOD)
        assert r.returncode == 2 and "没有 export 行" in r.stderr and not (root / "runs/logs/compat_mf.log").exists(), r.stderr


def test_waits_for_previous_eval() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root)
        prev = root / "runs/logs/t2b_eval_x.log"
        prev.write_text("=== x 等训练链\n")
        threading.Timer(3, lambda: prev.write_text("=== x DAY2_EVAL_DONE（成功 8、失败 0）\n")).start()
        t0 = time.time()
        rc, log, gate, _ = outcome(root, V2D_WAIT_FOR_EVAL_LOG="runs/logs/t2b_eval_x.log", **GOOD)
        assert rc == 0 and time.time() - t0 >= 3 and "上一段评测链已结束" in log and "完成 1" in gate, log


def test_chain_after_waits_while_compat_waits() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root)
        prev = root / "runs/logs/t2b_eval_x.log"
        prev.write_text("=== x 等 runs/logs/t2b_train_x.log 的最后一行变成 DAY2_TRAIN_DONE / DAY2_TRAIN_FAILED\n")
        c = subprocess.Popen(["bash", "stages/evaluate/compat_8c778f9.sh", "stages/evaluate/compat/mf.tsv", "compat_mf"], cwd=root,
                             env=env(root, V2D_WAIT_FOR_EVAL_LOG="runs/logs/t2b_eval_x.log", **GOOD),
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.5)
        a = subprocess.Popen(["bash", "stages/policy/chain_after.sh", "runs/logs/compat_mf.gate", "runs/logs/compat_mf.log", "1",
                              "e.tsv", "s.txt", "f.txt", "24564", "box_train"], cwd=root, env=env(root),
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(3.5)
        early = (root / "runs/logs/box_train.log").exists()
        prev.write_text("=== x DAY2_EVAL_DONE（成功 8、失败 0）\n")
        c.wait(timeout=90); a.wait(timeout=90)
        assert not early, "compat 还在等，chain_after 就做了决定"
        log = (root / "runs/logs/compat_mf.log").read_text()
        assert sum("DAY2_EVAL_" in line for line in log.splitlines()) == 1, log
        assert "FAKE_CHAIN_STARTED" in (root / "runs/logs/box_train.log").read_text()


def test_dryrun() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root)
        r = compat(root, V2D_COMPAT_DRYRUN="1")
        assert r.returncode == 0 and r.stdout.rstrip().endswith("COMPAT_DRYRUN_OK"), r.stdout + r.stderr
        assert "guard t1-ep1-w2048-cap4m-mf4090（1 世界）：fc_export 预检退出码 12" in r.stdout, r.stdout
        assert not any((root / p).exists() for p in ("runs/logs/compat_mf.log", "runs/logs/compat_mf.gate", "runs/.chain-active-compat_mf"))
        r = compat(root, V2D_COMPAT_DRYRUN="1", FAKE_DRY_RC_t2_ep2_w2048_cap4m_mf4090="2")
        assert r.returncode == 1 and "COMPAT_DRYRUN_FAIL 1" in r.stdout, r.stdout
        r = compat(root, V2D_COMPAT_DRYRUN="1", V2D_WAIT_FOR_EVAL_LOG="runs/logs/missing.log")
        assert r.returncode == 1 and "COMPAT_DRYRUN_FAIL 1" in r.stdout, r.stdout


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("ALL_OK")
