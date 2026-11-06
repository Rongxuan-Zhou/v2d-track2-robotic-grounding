"""export_chain.sh（连同 check_export.sh、scripts/dev/record）的行为测试（暂无契约 ID：运行规范待成文）。只用标准库。

临时目录里搭最小仓库：真的 export_chain.sh、check_export.sh（经 V2D_COMPAT_PY_IMPORT 走宿主机）、scripts/dev/record；假的 fc_export.sh
（按 run 与步数的环境变量决定退出码，预检模式另有一套；已有完整结果就直接返回、不查 GPU，GPU 被占时退出码 6，都同真的 fc_export.sh）、
check_export.py、fc_src_check.py，
PATH 前面放假的 nvidia-smi（runs/gpu_busy 存在且未过期时报一个进程）。计划第 3.2 节、第 4 节（第 2 版）。
  1. 正常：导出、核对、候选条件都过记 ok；没有 DONE 记 no_done；有 NONFINITE 记 nonfinite、不调 fc_export；最终一步退出码 12 记 nonfinite；
     结果表的 sha256 与文件一致；
  2. 候选条件：没有 4096 世界评测、有非有限世界、checkpoint 不同 → no_eval；导出文件的 episode 号与清单不符 → ep_mismatch；
  3. 规矩二：最终一步报 13，往前退时跳过 12、用第一份导出与核对都过的，补评 4096 世界 → fallback；补评失败或补评有非有限世界 → fallback_noeval；
     一路 13 到上限 → fallback_failed，只试上限那么多份；往前退时遇到别的失败、核对不过 → 停在这一条；
  4. 导出其他失败 → export_failed；核对不过 → check_failed，接着做下一行；
  5. GPU：开头被占、过一会儿空了 → 等到再做；一直被占 → EXPORT_CHAIN_FAILED；做到一半有进程占了 GPU（fc_export 退出码 6）→ 等空了重跑这一行；
     V2D_GPU_WAIT_START=0：GPU 一直被占，导出已在的行照样核对、记 ok；要真导出的行在 V2D_GPU_WAIT_S=0 下立刻停；这个开关只许 0 或 1；
  6. compat 门槛不是“完成 1”：记警告照样导出，日志里没有结束标记的原文；没设 V2D_FC_SRC、副本核不过：EXPORT_CHAIN_FAILED，一行都不做；
  7. 旧日志或旧结果表已存在、清单没有行：退出码 2，不写日志；
  8. 设了 V2D_WAIT_FOR_EVAL_LOG：等那份日志结束才开始；本日志里不出现评测链结束标记的原文；
  9. 预检：不写日志、结果表与标记；没训完、NONFINITE、预检退出码 12 的行只提示；别的预检退出码 → EXPORT_DRYRUN_FAIL。
用法：python3 stages/evaluate/test_export_chain.py
"""
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STEP = "349999104"
EPS = {3: "t1-ep3-w2048-cap4m-mf4090", 6: "t1-ep6-w2048-cap4m-mf4090", 8: "t1-ep8-w2048-cap4m-mf4090", 11: "t1-ep11-w2048-cap4m-mf4090"}
A, B, C, D = (EPS[e] for e in (3, 6, 8, 11))
FAKE_EXPORT = r'''#!/usr/bin/env bash
RUN=$1; STEP=$2; W=${V2D_EXPORT_WORLDS:-1}; TRAJ=${V2D_EXPORT_TRAJ:-1}; K="${RUN//-/_}"
if [ "${V2D_EXPORT_DRYRUN:-0}" = 1 ]; then key="FAKE_DRY_RC_$K"; rc="${!key:-0}"; echo "fc_export: DRYRUN $RUN policy_$STEP"; exit "$rc"; fi
echo "$RUN $STEP $W" >> runs/fake_calls.txt
EP=$(cat "runs/flash_chord/$RUN/EP" 2>/dev/null || echo 7)
d="runs/flash_chord/$RUN/export_8c778f9/s${STEP}_w${W}"; e="$d/episode_$(printf %06d "$EP")"
if [ -s "$e.json" ] && { [ "$TRAJ" = 0 ] || [ -s "$e.parquet" ]; }; then echo "fc_export: 已有结果 $e.json"; exit 0; fi   # 同真的：不查 GPU
if [ "${FAKE_BUSY_RUN:-}" = "$RUN" ] && [ ! -f runs/busy_done ]; then   # 第一次调用时有别的进程占了 GPU，占 2 秒
  date +%s > runs/gpu_busy; touch runs/busy_done; echo "fc_export: GPU 上有别的进程" >&2; exit 6
fi
[ -n "$(bin/nvidia-smi)" ] && { echo "fc_export: GPU 上有别的进程" >&2; exit 6; }
key="FAKE_RC_${K}_${STEP}_w${W}"; rc="${!key:-}"
[ -n "$rc" ] || { key="FAKE_RC_${K}_${STEP}"; rc="${!key:-0}"; }
if [ "$rc" = 0 ]; then
  key="FAKE_NF_${K}_${STEP}_w${W}"; nf="${!key:-[]}"
  mkdir -p "$d"; echo "{\"world_count\": $W, \"step_count\": 100, \"checkpoint_sha256\": \"ck-$RUN-$STEP\", \"non_finite_worlds\": $nf}" > "$e.json"
  [ "$TRAJ" = 1 ] && echo "parquet $RUN $STEP" > "$e.parquet"
  echo "fake ok $RUN $STEP"
fi
[ "$rc" = 0 ] || echo "fake fail rc $rc" >&2
exit "$rc"
'''
FAKE_CHECK = r'''import os, sys
mode = os.environ.get("FAKE_CHECK_" + sys.argv[1].replace("-", "_") + "_" + sys.argv[2], "ok")
print("EXPORT_OK" if mode == "ok" else "EXPORT_FAIL 1"); sys.exit(0 if mode == "ok" else 1)
'''
FAKE_SRC = 'import os, sys\nrc = int(os.environ.get("FAKE_SRC_RC", "0")); print("FC_SRC_OK" if rc == 0 else "FC_SRC_FAIL 1"); sys.exit(rc)\n'
FAKE_GPU = r'''#!/usr/bin/env bash
# runs/gpu_busy 里写着开始占用的时刻；FAKE_GPU_HOLD 秒（默认 2）之内报一个进程。FAKE_GPU=busy：一直被占
[ "${FAKE_GPU:-}" = busy ] && { printf "4242\n4343\n"; exit 0; }
[ -f runs/gpu_busy ] && [ $(( $(date +%s) - $(cat runs/gpu_busy) )) -lt "${FAKE_GPU_HOLD:-2}" ] && echo 4242
exit 0
'''
GATE_OK = "=== x DAY2_TRAIN_DONE（完成 1、失败 0、跳过 0、发散 0）\n"
TSV = "# 注释行\n" + "".join(f"1\t{ep}\t{run}\t{STEP}\n" for ep, run in EPS.items())


def evaljson(root: Path, run: str, step: str = STEP, **kw) -> None:
    e = {"world_count": 4096, "step_count": 100, "checkpoint_sha256": f"ck-{run}-{step}", "non_finite_worlds": []}
    e.update(kw)
    (root / f"runs/flash_chord/{run}/eval_{step}_w4096.json").write_text(json.dumps(e))


def setup(root: Path, tsv: str = TSV, done=(A, B), nonfinite=(), policies: dict = None, evals=None) -> None:
    for d in ("stages/evaluate/exports", "stages/policy", "scripts/dev", "bin", "runs/logs", "third_party/patches"):
        (root / d).mkdir(parents=True, exist_ok=True)
    for f in ("stages/evaluate/export_chain.sh", "stages/evaluate/check_export.sh", "scripts/dev/record"):
        shutil.copy(REPO / f, root / f)
    files = {"stages/evaluate/fc_export.sh": FAKE_EXPORT, "stages/evaluate/check_export.py": FAKE_CHECK,
             "stages/policy/fc_src_check.py": FAKE_SRC, "bin/nvidia-smi": FAKE_GPU}
    for f, text in files.items():
        (root / f).write_text(text)
    for f in ("stages/evaluate/fc_export.sh", "bin/nvidia-smi", "stages/evaluate/export_chain.sh", "stages/evaluate/check_export.sh", "scripts/dev/record"):
        (root / f).chmod(0o755)
    (root / "stages/evaluate/exports/mf.tsv").write_text(tsv)
    (root / ".v2d-source-commit").write_text("deadbeef\n")
    (root / "third_party/patches/flash_chord-8c778f9.sha256").write_text("x  y\n")
    (root / "runs/logs/compat_mf.gate").write_text(GATE_OK)
    ep_of = {run: ep for ep, run in EPS.items()}
    for run in done:
        (root / f"runs/flash_chord/{run}").mkdir(parents=True, exist_ok=True)
        (root / f"runs/flash_chord/{run}/DONE").write_text("")
        (root / f"runs/flash_chord/{run}/EP").write_text(str(ep_of[run]))
        if evals is None or run in evals:
            evaljson(root, run)
    for run in nonfinite:
        (root / f"runs/flash_chord/{run}/NONFINITE").write_text("")
    for run, steps in (policies or {}).items():
        for s in steps:
            (root / f"runs/flash_chord/{run}/policy_{s}.safetensors").write_text("")


def env(root: Path, **kw) -> dict:
    e = dict(os.environ, PATH=f"{root / 'bin'}:{os.environ['PATH']}", V2D_FC_SRC="/x/fcsrc", V2D_POLL_S="1", V2D_COMPAT_PY_IMPORT="import json",
             V2D_COMPAT_GATE="runs/logs/compat_mf.gate")
    e.update(kw)
    return {k: v for k, v in e.items() if v is not None}


def k(run: str, step: str = STEP, w: str = "", kind: str = "RC") -> str:
    return f"FAKE_{kind}_{run.replace('-', '_')}_{step}" + (f"_w{w}" if w else "")


def chain(root: Path, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "stages/evaluate/export_chain.sh", "stages/evaluate/exports/mf.tsv", "export_mf"], cwd=root, env=env(root, **kw),
                          capture_output=True, text=True, errors="replace", timeout=120)


def log(root: Path) -> str:
    return (root / "runs/logs/export_mf.log").read_text()


def table(root: Path) -> dict:
    lines = (root / "runs/exports/export_mf.tsv").read_text().splitlines()
    assert lines[0].split("\t")[:5] == ["tier", "episode", "run", "step", "status"], lines[0]
    return {f[2]: f for f in (line.split("\t") for line in lines[1:])}


def calls(root: Path) -> list:
    f = root / "runs/fake_calls.txt"
    return f.read_text().splitlines() if f.exists() else []


def mine(root: Path, run: str) -> list:
    return [c for c in calls(root) if c.startswith(run + " ")]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def no_markers(text: str) -> bool:
    return not any(m in text for m in ("DAY2_EVAL_DONE", "DAY2_EVAL_FAILED", "DAY2_TRAIN_DONE", "DAY2_TRAIN_FAILED"))


def test_normal() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root, done=(A, B, D), nonfinite=(D,))
        r = chain(root)
        lg = log(root)
        assert r.returncode == 0 and lg.rstrip().endswith("EXPORT_CHAIN_DONE（成功 2、失败 2）"), lg
        tb = table(root)
        assert [tb[x][4] for x in (A, B, C, D)] == ["ok", "ok", "no_done", "nonfinite"], tb
        j = root / tb[A][5]; q = root / tb[A][6]
        assert tb[A][5].endswith(f"s{STEP}_w1/episode_000003.json") and tb[A][7] == sha(j) and tb[A][8] == sha(q), tb[A]
        assert tb[C][5:] == ["", "", "", ""], tb[C]
        assert mine(root, D) == [], calls(root)
        assert (root / f"results/submission/exports/export_mf_{A}_s{STEP}.txt").exists()
        assert (root / f"results/submission/exports/export_mf_{A}_s{STEP}_check.txt").exists()
        assert not (root / "runs/.chain-active-export_mf").exists()
        assert sum("EXPORT_CHAIN_" in line for line in lg.splitlines()) == 1 and "compat 门槛是“完成 1”" in lg, lg
    with tempfile.TemporaryDirectory() as t:   # 最终一步 12（非有限、读不了，但没有 NONFINITE）：规矩一，记 nonfinite，不回退
        root = Path(t); setup(root, policies={A: ("300", STEP)})
        chain(root, **{k(A): "12"})
        assert table(root)[A][4] == "nonfinite" and mine(root, A) == [f"{A} {STEP} 1"], (table(root), calls(root))


def test_candidate_condition() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root, done=(A, B, C), evals=(B, C))
        evaljson(root, B, non_finite_worlds=[17, 99])
        evaljson(root, C, checkpoint_sha256="other")
        chain(root)
        tb = table(root); lg = log(root)
        assert [tb[x][4] for x in (A, B, C)] == ["no_eval", "no_eval", "no_eval"], tb
        assert tb[A][7] and tb[A][8], tb[A]   # 轨迹照样记 sha256，拉回时进 held/
        assert "没有可用的 4096 世界评测" in lg and "non_finite_worlds 2 个" in lg and "checkpoint 不同" in lg, lg
        assert lg.rstrip().endswith("EXPORT_CHAIN_DONE（成功 0、失败 4）"), lg
    with tempfile.TemporaryDirectory() as t:   # 导出文件的 episode 号与清单不符
        root = Path(t); setup(root)
        (root / f"runs/flash_chord/{A}/EP").write_text("9")
        chain(root)
        tb = table(root)
        assert tb[A][4] == "ep_mismatch" and tb[A][5] == "" and tb[B][4] == "ok", tb


def test_rule2_fallback() -> None:
    pol = {A: ("100", "200", "300", STEP)}
    with tempfile.TemporaryDirectory() as t:   # 13 → 300 非有限跳过 → 200 能用 → 补评成功
        root = Path(t); setup(root, policies=pol)
        r = chain(root, **{k(A): "13", k(A, "300"): "12"})
        tb = table(root)
        assert r.returncode == 0 and tb[A][3:5] == ["200", "fallback"] and tb[B][4] == "ok", tb
        assert mine(root, A) == [f"{A} {STEP} 1", f"{A} 300 1", f"{A} 200 1", f"{A} 200 4096"], calls(root)
        assert tb[A][5].endswith("s200_w1/episode_000003.json") and tb[A][8], tb[A]
        assert (root / f"results/submission/exports/export_mf_{A}_s200_w4096.txt").exists()
    for extra in ({k(A, "300", "4096"): "3"}, {k(A, "300", "4096", "NF"): "[5]"}):   # 补评失败、补评有非有限世界 → fallback_noeval
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); setup(root, policies=pol)
            chain(root, **{k(A): "13", **extra})
            tb = table(root); lg = log(root)
            assert tb[A][3:5] == ["300", "fallback_noeval"] and "成功 1、失败 3" in lg, (extra, tb, lg)
    with tempfile.TemporaryDirectory() as t:   # 一路 13，上限 2 份
        root = Path(t); setup(root, policies=pol)
        chain(root, V2D_FALLBACK_MAX="2", **{k(A): "13", k(A, "300"): "13", k(A, "200"): "13", k(A, "100"): "0"})
        tb = table(root)
        assert tb[A][3:5] == ["200", "fallback_failed"], tb
        assert mine(root, A) == [f"{A} {STEP} 1", f"{A} 300 1", f"{A} 200 1"], calls(root)
    with tempfile.TemporaryDirectory() as t:   # 往前退时别的失败 → 停在这一条
        root = Path(t); setup(root, policies=pol)
        chain(root, **{k(A): "13", k(A, "300"): "3"})
        tb = table(root)
        assert tb[A][3:5] == ["300", "fallback_failed"] and tb[B][4] == "ok", tb
        assert mine(root, A) == [f"{A} {STEP} 1", f"{A} 300 1"], calls(root)
    with tempfile.TemporaryDirectory() as t:   # 往前退时核对不过 → 停在这一条
        root = Path(t); setup(root, policies=pol)
        chain(root, **{k(A): "13", f"FAKE_CHECK_{A.replace('-', '_')}_300": "fail"})
        assert table(root)[A][3:5] == ["300", "fallback_failed"], table(root)


def test_row_failures() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root)
        r = chain(root, **{k(A): "3", f"FAKE_CHECK_{B.replace('-', '_')}_{STEP}": "fail"})
        tb = table(root); lg = log(root)
        assert r.returncode == 0 and tb[A][4] == "export_failed" and tb[B][4] == "check_failed", tb
        assert tb[B][7] and tb[B][8], tb[B]   # 核对不过的那份文件照样记 sha256，方便事后对照
        assert lg.rstrip().endswith("EXPORT_CHAIN_DONE（成功 0、失败 4）"), lg


def test_gpu_waits() -> None:
    with tempfile.TemporaryDirectory() as t:   # 开头被占，2 秒后空了
        root = Path(t); setup(root)
        (root / "runs/gpu_busy").write_text(str(int(time.time())))
        t0 = time.time(); r = chain(root, V2D_GPU_WAIT_S="30")
        lg = log(root)
        assert r.returncode == 0 and time.time() - t0 >= 1 and "GPU 空了，接着做" in lg and table(root)[A][4] == "ok", lg
    with tempfile.TemporaryDirectory() as t:   # 一直被占
        root = Path(t); setup(root)
        r = chain(root, FAKE_GPU="busy", V2D_GPU_WAIT_S="2")
        lg = log(root)
        assert r.returncode == 1 and "4242,4343" in lg and lg.rstrip().endswith("EXPORT_CHAIN_FAILED") and calls(root) == [], lg
    with tempfile.TemporaryDirectory() as t:   # 做到一半有进程占了 GPU：等空了重跑这一行
        root = Path(t); setup(root)
        r = chain(root, FAKE_BUSY_RUN=B, V2D_GPU_WAIT_S="30")
        tb = table(root); lg = log(root)
        assert r.returncode == 0 and tb[A][4] == "ok" and tb[B][4] == "ok", (tb, lg)
        assert mine(root, B) == [f"{B} {STEP} 1", f"{B} {STEP} 1"] and "GPU 空了，接着做" in lg, (calls(root), lg)
        assert (root / f"results/submission/exports/export_mf_{B}_s{STEP}.txt").exists()
        assert not (root / f"results/submission/exports/export_mf_{B}_s{STEP}.txt.failed").exists()


def preexport(root: Path, run: str, step: str = STEP) -> None:   # 先放好一份完整的 1 世界导出，内容同假的 fc_export 写的
    d = root / f"runs/flash_chord/{run}/export_8c778f9/s{step}_w1"; d.mkdir(parents=True)
    e = d / f"episode_{int((root / f'runs/flash_chord/{run}/EP').read_text()):06d}"
    e.with_suffix(".json").write_text(json.dumps({"world_count": 1, "step_count": 100, "checkpoint_sha256": f"ck-{run}-{step}", "non_finite_worlds": []}))
    e.with_suffix(".parquet").write_text(f"parquet {run} {step}\n")


def test_no_start_wait() -> None:
    with tempfile.TemporaryDirectory() as t:   # 两行导出都已在：GPU 一直被占也做完，不等 GPU
        root = Path(t); setup(root)
        preexport(root, A); preexport(root, B)
        r = chain(root, FAKE_GPU="busy", V2D_GPU_WAIT_START="0", V2D_GPU_WAIT_S="0")
        tb = table(root); lg = log(root)
        assert r.returncode == 0 and tb[A][4] == "ok" and tb[B][4] == "ok" and lg.rstrip().endswith("EXPORT_CHAIN_DONE（成功 2、失败 2）"), (tb, lg)
        assert "开头不等 GPU" in lg and "GPU 上有别的进程" not in lg and mine(root, A) == [f"{A} {STEP} 1"], (calls(root), lg)
        assert tb[A][7] == sha(root / tb[A][5]) and tb[A][8] == sha(root / tb[A][6]), tb[A]
    with tempfile.TemporaryDirectory() as t:   # B 的导出不在：A 记 ok；B 要真导出，GPU 被占、V2D_GPU_WAIT_S=0 → 立刻停
        root = Path(t); setup(root)
        preexport(root, A)
        t0 = time.time(); r = chain(root, FAKE_GPU="busy", V2D_GPU_WAIT_START="0", V2D_GPU_WAIT_S="0")
        tb = table(root); lg = log(root)
        assert r.returncode == 1 and time.time() - t0 < 30 and tb[A][4] == "ok" and B not in tb, (tb, lg)
        assert f"停在 {B} 第 {STEP} 步" in lg and lg.rstrip().endswith("EXPORT_CHAIN_FAILED"), lg
    with tempfile.TemporaryDirectory() as t:   # 只许 0 或 1：拒绝，不写日志
        root = Path(t); setup(root)
        r = chain(root, V2D_GPU_WAIT_START="no")
        assert r.returncode == 2 and "V2D_GPU_WAIT_START" in r.stderr and not (root / "runs/logs/export_mf.log").exists(), r.stderr
    with tempfile.TemporaryDirectory() as t:   # 预检里说一句
        root = Path(t); setup(root)
        r = chain(root, V2D_EXPORT_CHAIN_DRYRUN="1", V2D_GPU_WAIT_START="0")
        assert r.returncode == 0 and "开头不等 GPU（V2D_GPU_WAIT_START=0）" in r.stdout, r.stdout


def test_gate_and_stops() -> None:
    with tempfile.TemporaryDirectory() as t:   # 门槛“完成 0”：警告，照样导出
        root = Path(t); setup(root)
        (root / "runs/logs/compat_bad.gate").write_text("=== x DAY2_TRAIN_DONE（完成 0、失败 1、跳过 0、发散 0）\n")
        r = chain(root, V2D_COMPAT_GATE="runs/logs/compat_bad.gate")
        lg = log(root)
        assert r.returncode == 0 and "警告：compat 门槛" in lg and "训练链完成（完成 0、失败 1" in lg and table(root)[A][4] == "ok", lg
        assert no_markers(lg), lg
    for kw, word in (({"V2D_FC_SRC": None}, "没设 V2D_FC_SRC"), ({"FAKE_SRC_RC": "1"}, "不是核对过的 8c778f9 副本")):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); setup(root)
            r = chain(root, **kw)
            lg = log(root)
            assert r.returncode == 1 and word in lg and lg.rstrip().endswith("EXPORT_CHAIN_FAILED"), (kw, lg)
            assert calls(root) == [] and table(root) == {}, kw
            assert not (root / "runs/.chain-active-export_mf").exists()


def test_refusals() -> None:
    for pre in ("runs/logs/export_mf.log", "runs/exports/export_mf.tsv"):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); setup(root)
            (root / pre).parent.mkdir(parents=True, exist_ok=True); (root / pre).write_text("old\n")
            r = chain(root)
            assert r.returncode == 2 and (root / pre).read_text() == "old\n" and calls(root) == [], (pre, r.stderr)
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root, tsv="# 只有注释\n\n")
        r = chain(root)
        assert r.returncode == 2 and not (root / "runs/logs/export_mf.log").exists(), r.stderr


def test_waits_for_previous_eval() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root)
        prev = root / "runs/logs/box_eval_mf.log"
        prev.write_text("=== x 等 runs/logs/box_train_mf.log 的最后一行变成 DAY2_TRAIN_DONE / DAY2_TRAIN_FAILED\n")
        threading.Timer(3, lambda: prev.write_text("=== x DAY2_EVAL_DONE（成功 4、失败 0）\n")).start()
        t0 = time.time()
        r = chain(root, V2D_WAIT_FOR_EVAL_LOG="runs/logs/box_eval_mf.log")
        lg = log(root)
        assert r.returncode == 0 and time.time() - t0 >= 3 and "上一段评测链已结束：=== x 评测链完成（成功 4、失败 0）" in lg, lg
        assert no_markers(lg), lg


def test_dryrun() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root, done=(A, B, D), nonfinite=(D,), evals=(A,))
        (root / "runs/logs/compat_mf.gate").unlink()
        r = chain(root, V2D_EXPORT_CHAIN_DRYRUN="1", **{f"FAKE_DRY_RC_{B.replace('-', '_')}": "12"})
        out = r.stdout
        assert r.returncode == 0 and out.rstrip().endswith("EXPORT_DRYRUN_OK"), out
        assert "还没写，开跑时再看" in out and f"{C}：还没有 DONE" in out and f"{D}：有 NONFINITE" in out and f"{B}：fc_export 预检退出码 12" in out, out
        assert f"有干净的 4096 世界评测 runs/flash_chord/{A}/eval_{STEP}_w4096.json" in out, out
        assert not (root / "runs/logs/export_mf.log").exists() and not (root / "runs/exports").exists() and calls(root) == []
        assert not (root / "runs/.chain-active-export_mf").exists()
    with tempfile.TemporaryDirectory() as t:   # 已训完但还没有 4096 世界评测：只提示
        root = Path(t); setup(root, evals=())
        r = chain(root, V2D_EXPORT_CHAIN_DRYRUN="1")
        assert r.returncode == 0 and f"{A}：还没有干净的 4096 世界评测" in r.stdout, r.stdout
        r = chain(root, V2D_EXPORT_CHAIN_DRYRUN="1", V2D_WAIT_FOR_EVAL_LOG="runs/logs/missing.log", **{f"FAKE_DRY_RC_{A.replace('-', '_')}": "3"})
        assert r.returncode == 1 and r.stdout.rstrip().endswith("EXPORT_DRYRUN_FAIL 2"), r.stdout


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"ok  {name}")
    print("ALL_OK")
