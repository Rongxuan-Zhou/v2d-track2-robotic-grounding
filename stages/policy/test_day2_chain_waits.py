"""day2_train_chain.sh 的停链标记与等卡、day2_eval_chain.sh 的等卡（守 agent-doc/plan-t2-scaleout-2026-10-09.md 第 5 节与 Fable 5.1 审查 M1、M3；
暂无契约 ID）。只用标准库。链脚本要 bash 4 以上（mapfile、declare -A），在训练机上跑；Mac 自带的 bash 3.2 跑不了，开头就报错退出。
临时目录里放真的两份链脚本，其余全是桩：precheck_expect.sh 只打 PRECHECK_OK；fc_run.sh / fc_run_watched.sh 建 DONE 与 policy 文件并记调用；
smoke_check.sh、in-flash-chord 什么也不做；record 给冒烟写帧数 100 的 summary_json，给评测就执行 fc_eval.sh 桩；
nvidia-smi 桩打印 gpu_pids.txt（有这个文件就算 GPU 被别人占着）。轮询 1 秒。期望表 3 条 run。
训练链：
  1. 起链前就有 runs/.chain-stop-<日志名>：什么也不开，最后一行 DAY2_TRAIN_DONE（完成 0、…），日志里有“停链”
  2. 第一条正式训练期间建停链标记：3 条冒烟都跑了，正式只开了第一条，完成 1
  3. 起链时 GPU 被占、V2D_GPU_WAIT_S 没设：照旧立刻停（DAY2_TRAIN_FAILED），什么也不开
  4. 第一条正式训练期间 GPU 被别人占上、几秒后空出来：V2D_GPU_WAIT_S=20 时等到空了接着跑完，完成 3
  5. GPU 一直被占、V2D_GPU_WAIT_S=2：等 2 秒后停，DAY2_TRAIN_FAILED
评测链：
  6. 第一份评测期间 GPU 被占、几秒后空出来：V2D_GPU_WAIT_S=20 时等到空了评完 3 份，DAY2_EVAL_DONE（成功 3、失败 0）
  7. 同样情形、V2D_GPU_WAIT_S 没设：第二份评测前停，DAY2_EVAL_FAILED
用法：python3 stages/policy/test_day2_chain_waits.py
"""
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
HEADER = "run\tsmoke\tsequence_id\tplayback\tsrc_frames\tsupport\tpos_tol\tori_tol\tcapacity\tseed\txeval"
RUNS = [f"t1-ep{e}-wtest" for e in (1, 2, 3)]
STUB_RUN = """#!/usr/bin/env bash
run=$1; steps=$4
if [ "$(basename "$0")" = fc_run_watched.sh ]; then
  if ! grep -q '^formal' calls.txt 2>/dev/null; then
    [ -n "${STUB_STOP_FIRST:-}" ] && touch "runs/.chain-stop-$STUB_STOP_FIRST"
    [ -n "${STUB_GPU_FIRST:-}" ] && echo 4242 > gpu_pids.txt
  fi
  echo "formal $run" >> calls.txt
else
  echo "smoke $run" >> calls.txt
fi
mkdir -p "runs/flash_chord/$run"; : > "runs/flash_chord/$run/policy_$steps.safetensors"; date > "runs/flash_chord/$run/DONE"
printf '{"event": "start"}\\n' > "runs/flash_chord/$run/run_record.jsonl"
"""
STUB_RECORD = """#!/usr/bin/env bash
name=$1; shift; mkdir -p "results/$(dirname "$name")"
case "$*" in
  *fc_eval.sh*) "$@" > "results/$name.txt" 2>&1 ;;
  *) echo ' summary_json {"frames": 100}' > "results/$name.txt" ;;
esac
"""
STUB_EVAL = """#!/usr/bin/env bash
run=$1; step=$2
echo '{"step_count": 100}' > "runs/flash_chord/$run/eval_${step}_w4096.json"
if [ -n "${STUB_GPU_ON_EVAL:-}" ] && ! grep -q '^eval' calls.txt 2>/dev/null; then echo 4343 > gpu_pids.txt; fi
echo "eval $run" >> calls.txt
"""


def make_root(t: Path) -> Path:
    root = t / "repo"
    for d in ("stages/policy", "stages/evaluate", "scripts/dev", "runs/logs", "bin"):
        (root / d).mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "stages/policy/day2_train_chain.sh", root / "stages/policy")
    shutil.copy(REPO / "stages/evaluate/day2_eval_chain.sh", root / "stages/evaluate")
    stubs = {
        "stages/policy/precheck_expect.sh": "#!/usr/bin/env bash\necho PRECHECK_OK\n",
        "stages/policy/fc_run.sh": STUB_RUN,
        "stages/policy/fc_run_watched.sh": STUB_RUN,
        "stages/policy/smoke_check.sh": "#!/usr/bin/env bash\nexit 0\n",
        "stages/policy/fc_eval.sh": STUB_EVAL,
        "scripts/dev/in-flash-chord": "#!/usr/bin/env bash\nexit 0\n",
        "scripts/dev/record": STUB_RECORD,
        "bin/nvidia-smi": "#!/usr/bin/env bash\n[ -f gpu_pids.txt ] && cat gpu_pids.txt\nexit 0\n",
    }
    for rel, body in stubs.items():
        (root / rel).write_text(body)
        (root / rel).chmod(0o755)
    (root / ".v2d-source-commit").write_text("abc123def456\n")
    rows = [HEADER] + ["\t".join([r, f"smoke-{r}", f"seq{i}", "frame", "100", "false", "0.015", "0.086", "4000000", "42", "no"])
                       for i, r in enumerate(RUNS)]
    (root / "e.tsv").write_text("\n".join(rows) + "\n")
    (root / "s.txt").write_text("".join(f"V2D_X=1 smoke-{r} /repo/p.parquet 2048 4096000\n" for r in RUNS))
    (root / "q.txt").write_text("".join(f"V2D_X=1 {r} /repo/p.parquet 2048 349999104\n" for r in RUNS))
    return root


def env(root: Path, **kw) -> dict:
    e = dict(os.environ, PATH=f"{root / 'bin'}:{os.environ['PATH']}", V2D_POLL_S="1", V2D_MIN_FREE_MB="1")
    e.pop("V2D_GPU_WAIT_S", None)
    e.update({k: str(v) for k, v in kw.items()})
    return e


def free_gpu_later(root: Path, after: float):
    """gpu_pids.txt 出现以后再过 after 秒删掉它（模拟别人的任务结束）。"""
    def watch():
        t0 = time.time()
        while time.time() - t0 < 60:
            if (root / "gpu_pids.txt").exists():
                time.sleep(after)
                (root / "gpu_pids.txt").unlink(missing_ok=True)
                return
            time.sleep(0.1)
    th = threading.Thread(target=watch, daemon=True)
    th.start()
    return th


def train(root: Path, **kw):
    r = subprocess.run(["bash", "stages/policy/day2_train_chain.sh", "e.tsv", "s.txt", "q.txt", "24564", "tchain"],
                       cwd=root, env=env(root, **kw), capture_output=True, text=True, timeout=120)
    log = (root / "runs/logs/tchain.log").read_text(encoding="utf-8")
    calls = (root / "calls.txt").read_text().split("\n") if (root / "calls.txt").exists() else []
    return r.returncode, log, [c for c in calls if c]


def last(log: str) -> str:
    return log.strip().splitlines()[-1]


def test_stop_mark_before_start():
    with tempfile.TemporaryDirectory() as t:
        root = make_root(Path(t))
        (root / "runs/.chain-stop-tchain").touch()
        rc, log, calls = train(root)
        assert rc == 0 and calls == [], (rc, calls, log)
        assert last(log).endswith("DAY2_TRAIN_DONE（完成 0、失败 0、跳过 0、发散 0）") and "见到 runs/.chain-stop-tchain" in log and "停链" in log, log


def test_stop_mark_during_formal():
    with tempfile.TemporaryDirectory() as t:
        root = make_root(Path(t))
        rc, log, calls = train(root, STUB_STOP_FIRST="tchain")
        assert rc == 0 and calls == [f"smoke smoke-{r}" for r in RUNS] + [f"formal {RUNS[0]}"], (calls, log)
        assert last(log).endswith("DAY2_TRAIN_DONE（完成 1、失败 0、跳过 0、发散 0）") and "停链" in log, log


def test_gpu_busy_without_wait_dies():
    with tempfile.TemporaryDirectory() as t:
        root = make_root(Path(t))
        (root / "gpu_pids.txt").write_text("4242\n")
        rc, log, calls = train(root)
        assert rc == 1 and calls == [] and last(log).endswith("DAY2_TRAIN_FAILED") and "GPU 上有别的进程（4242），停下" in log, (rc, calls, log)


def test_gpu_wait_then_continue():
    with tempfile.TemporaryDirectory() as t:
        root = make_root(Path(t))
        free_gpu_later(root, 3)
        rc, log, calls = train(root, STUB_GPU_FIRST=1, V2D_GPU_WAIT_S=20)
        assert rc == 0 and [c for c in calls if c.startswith("formal")] == [f"formal {r}" for r in RUNS], (calls, log)
        assert "最多等 20 秒" in log and "GPU 空了，接着跑" in log, log
        assert last(log).endswith("DAY2_TRAIN_DONE（完成 3、失败 0、跳过 0、发散 0）"), log


def test_gpu_wait_timeout_dies():
    with tempfile.TemporaryDirectory() as t:
        root = make_root(Path(t))
        (root / "gpu_pids.txt").write_text("4242\n")
        t0 = time.time()
        rc, log, calls = train(root, V2D_GPU_WAIT_S=2)
        assert rc == 1 and calls == [] and time.time() - t0 >= 2, (rc, calls)
        assert "GPU 被别的进程占了 2 秒，停下" in log and last(log).endswith("DAY2_TRAIN_FAILED"), log


def evaluate(root: Path, **kw):
    for r in RUNS:
        d = root / "runs/flash_chord" / r
        d.mkdir(parents=True, exist_ok=True)
        (d / "DONE").write_text("x\n")
    r = subprocess.run(["bash", "stages/evaluate/day2_eval_chain.sh", "e.tsv", "-", "echain"], cwd=root,
                       env=env(root, V2D_EVAL_STEPS="349999104", **kw), capture_output=True, text=True, timeout=120)
    log = (root / "runs/logs/echain.log").read_text(encoding="utf-8")
    calls = (root / "calls.txt").read_text().split("\n") if (root / "calls.txt").exists() else []
    return r.returncode, log, [c for c in calls if c]


def test_eval_waits_for_gpu():
    with tempfile.TemporaryDirectory() as t:
        root = make_root(Path(t))
        free_gpu_later(root, 3)
        rc, log, calls = evaluate(root, STUB_GPU_ON_EVAL=1, V2D_GPU_WAIT_S=20)
        assert rc == 0 and calls == [f"eval {r}" for r in RUNS], (calls, log)
        assert "GPU 空了，接着评" in log and last(log).endswith("DAY2_EVAL_DONE（成功 3、失败 0）"), log


def test_eval_without_wait_stops():
    with tempfile.TemporaryDirectory() as t:
        root = make_root(Path(t))
        rc, log, calls = evaluate(root, STUB_GPU_ON_EVAL=1)
        assert rc == 1 and calls == [f"eval {RUNS[0]}"], (calls, log)
        assert "GPU 上有别的进程（4343），停下" in log and last(log).endswith("DAY2_EVAL_FAILED"), log


if __name__ == "__main__":
    ver = subprocess.run(["bash", "-c", "echo ${BASH_VERSINFO[0]}"], capture_output=True, text=True).stdout.strip()
    if not ver.isdigit() or int(ver) < 4:
        raise SystemExit(f"test_day2_chain_waits: 要 bash 4 以上，这里是 {ver}；在训练机上跑")
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("ALL_OK")
