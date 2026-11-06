"""dump_chain.sh（连同 scripts/dev/record）的行为测试（暂无契约 ID：运行规范待成文）。只用标准库；要 bash 4，在 mf4090 上跑。

临时目录里搭最小仓库：真的 dump_chain.sh 与 scripts/dev/record；假的 fc_eval_dump.sh（按 run 的环境变量决定退出码；GPU 被占时退出码 6，
同真的 fc_eval_dump.sh；被调用时记下长链标记在不在）、fc_src_check.py，PATH 前面放假的 nvidia-smi（runs/gpu_busy 存在且未过期时报一个进程）
与假的 df（设了 FAKE_DF_MB 时报这么多可用 MB）。计划 agent-doc/plan-frame-onset-2026-10-14.md 第 3 节第 0 步。
  1. 正常：评完记 ok；已有 dump 记 exists、不调用；NONFINITE、没有 policy 各记一类、不调用；结果表的 sha256 与 npz 一致；
     调用时长链标记在，结束后删掉；每次调用落 results/frames/<日志名>/<run>_s<步数>.txt；结尾一行计数对；
  2. 某行失败（退出码 3）记 failed 与退出码，接着做下一行；
  3. GPU：开头被占、过一会儿空了 → 等到再做；一直被占 → DUMP_CHAIN_FAILED、一行都不做；做到一半退出码 6 → 等空了重跑这一行；
  4. 没设 V2D_FC_SRC、副本核不过：DUMP_CHAIN_FAILED，一行都不做；
  5. 拒绝（退出码 2、不写日志）：旧日志已存在、清单没有行、行格式不对、run 重复、日志名非法；
  6. 日志里链结束标记的原文只出现在最后一行；
  7. 预检：不写日志、结果表与标记；policy 缺、NONFINITE、原评测不是 4096 世界或有非有限世界、副本核不过、磁盘不够、别的长链标记、
     残留的 fc-evaldump-* 容器各报 FAIL；已有 dump 的行只提示；全过打 DUMP_DRYRUN_OK。
  另：副本核不过时不等 GPU 就停；调用时长链标记非空；环境里残留的 V2D_EVAL_WORLDS 不影响（固定 4096）；nvidia-smi 出错按被占处理；
  一直报 6 时重跑之间停顿、累计等满就停；清单最后一行没有换行也读。
用法：python3 stages/evaluate/test_dump_chain.py
"""
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
STEP = "349999104"
A, B, C, D = "t3-ep7-w2048-cap10m-r3a-hm-ml40", "t3-ep8-w2048-cap10m-r3a-hm-ml40", "t3-ep4-w2048-cap10m-r3a-hm-ml40", "t1-ep25-w2048-cap10m-ml40"
NAME = "dump_mf"
FAKE_DUMP = r'''#!/usr/bin/env bash
RUN=$1; STEP=$2; K="${RUN//-/_}"
echo "$RUN $STEP $([ -s runs/.chain-active-dump_mf ] && echo marked || echo unmarked) W=${V2D_EVAL_WORLDS:-unset}" >> runs/fake_calls.txt
[ "${FAKE_ALWAYS6_RUN:-}" = "$RUN" ] && { echo "fc_eval_dump: GPU 上有别的进程" >&2; exit 6; }   # 一直报 6，GPU 却看着是空的
if [ "${FAKE_BUSY_RUN:-}" = "$RUN" ] && [ ! -f runs/busy_done ]; then   # 第一次调用时有别的进程占了 GPU，占 2 秒
  date +%s > runs/gpu_busy; touch runs/busy_done; echo "fc_eval_dump: GPU 上有别的进程" >&2; exit 6
fi
[ -n "$(bin/nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] && { echo "fc_eval_dump: GPU 上有别的进程" >&2; exit 6; }
key="FAKE_RC_${K}"; rc="${!key:-0}"
if [ "$rc" = 0 ]; then
  d="runs/flash_chord/$RUN/eval_dump"; mkdir -p "$d"
  echo '{"world_count": 4096}' > "$d/eval_${STEP}_w4096.json"; echo "npz $RUN" > "$d/eval_${STEP}_w4096.npz"; echo "fc_eval_dump: ok $RUN"
fi
[ "$rc" = 0 ] || echo "fake fail rc $rc" >&2
exit "$rc"
'''
FAKE_SRC = 'import os, sys\nrc = int(os.environ.get("FAKE_SRC_RC", "0")); print("FC_SRC_OK" if rc == 0 else "FC_SRC_FAIL 1"); sys.exit(rc)\n'
FAKE_GPU = r'''#!/usr/bin/env bash
[ "${FAKE_GPU:-}" = error ] && { echo "NVIDIA-SMI has failed" >&2; exit 9; }
[ "${FAKE_GPU:-}" = busy ] && { printf "4242\n"; exit 0; }
[ -f runs/gpu_busy ] && [ $(( $(date +%s) - $(cat runs/gpu_busy) )) -lt "${FAKE_GPU_HOLD:-2}" ] && echo 4242
exit 0
'''
FAKE_DOCKER = r'''#!/usr/bin/env bash
[ -n "${FAKE_DOCKER_LEFT:-}" ] && echo "$FAKE_DOCKER_LEFT"
exit 0
'''
FAKE_DF = r'''#!/usr/bin/env bash
if [ -n "${FAKE_DF_MB:-}" ]; then printf 'Filesystem 1048576-blocks Used Available Capacity Mounted\n/dev/x 100000 1 %s 1%% /\n' "$FAKE_DF_MB"; exit 0; fi
exec /bin/df "$@"
'''


def evaljson(root: Path, run: str, **kw) -> None:
    e = {"world_count": 4096, "step_count": 100, "checkpoint_sha256": f"ck-{run}", "non_finite_worlds": []}
    e.update(kw)
    (root / f"runs/flash_chord/{run}/eval_{STEP}_w4096.json").write_text(json.dumps(e))


def setup(root: Path, runs=(A, B), dumped=(), nonfinite=(), no_policy=(), lst=None) -> None:
    for d in ("stages/evaluate", "stages/policy", "scripts/dev", "bin", "runs/logs", "third_party/patches"):
        (root / d).mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "stages/evaluate/dump_chain.sh", root / "stages/evaluate/dump_chain.sh")
    shutil.copy(REPO / "scripts/dev/record", root / "scripts/dev/record")
    for f, text in {"stages/evaluate/fc_eval_dump.sh": FAKE_DUMP, "stages/policy/fc_src_check.py": FAKE_SRC,
                    "bin/nvidia-smi": FAKE_GPU, "bin/df": FAKE_DF, "bin/docker": FAKE_DOCKER}.items():
        (root / f).write_text(text)
    for f in ("stages/evaluate/fc_eval_dump.sh", "stages/evaluate/dump_chain.sh", "scripts/dev/record", "bin/nvidia-smi", "bin/df", "bin/docker"):
        (root / f).chmod(0o755)
    (root / ".v2d-source-commit").write_text("deadbeef\n")
    (root / "third_party/patches/flash_chord-8c778f9.sha256").write_text("x  y\n")
    for run in runs:
        d = root / f"runs/flash_chord/{run}"
        d.mkdir(parents=True, exist_ok=True)
        if run not in no_policy:
            (d / f"policy_{STEP}.safetensors").write_text("p")
        evaljson(root, run)
    for run in dumped:
        d = root / f"runs/flash_chord/{run}/eval_dump"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"eval_{STEP}_w4096.json").write_text("{}")
        (d / f"eval_{STEP}_w4096.npz").write_text("old")
    for run in nonfinite:
        (root / f"runs/flash_chord/{run}/NONFINITE").write_text("")
    (root / "list.txt").write_text(lst if lst is not None else "# 注释\n\n" + "".join(f"{r} {STEP}\n" for r in runs))


def env(root: Path, **kw) -> dict:
    e = dict(os.environ, PATH=f"{root / 'bin'}:{os.environ['PATH']}", V2D_FC_SRC="/x/fcsrc", V2D_POLL_S="1")
    e.update(kw)
    return {k: v for k, v in e.items() if v is not None}


def chain(root: Path, name: str = NAME, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "stages/evaluate/dump_chain.sh", "list.txt", name], cwd=root, env=env(root, **kw),
                          capture_output=True, text=True, errors="replace", timeout=120)


def log(root: Path) -> str:
    return (root / f"runs/logs/{NAME}.log").read_text()


def table(root: Path) -> dict:
    lines = (root / f"runs/frames/{NAME}.tsv").read_text().splitlines()
    assert lines[0].split("\t") == ["run", "step", "status", "rc", "json", "npz", "npz_sha256"], lines[0]
    return {f[0]: f for f in (line.split("\t") for line in lines[1:])}


def calls(root: Path) -> list:
    f = root / "runs/fake_calls.txt"
    return f.read_text().splitlines() if f.exists() else []


def k(run: str) -> str:
    return "FAKE_RC_" + run.replace("-", "_")


def test_normal(tmp: Path) -> None:
    setup(tmp, runs=(A, B, C, D), dumped=(B,), nonfinite=(C,), no_policy=(D,))
    p = chain(tmp, V2D_EVAL_WORLDS="64")   # 环境里残留的世界数不影响（审查第 12 条）
    assert p.returncode == 0, p.stderr
    t = table(tmp)
    assert [t[r][2] for r in (A, B, C, D)] == ["ok", "exists", "nonfinite", "no_policy"], t
    npz = tmp / f"runs/flash_chord/{A}/eval_dump/eval_{STEP}_w4096.npz"
    assert t[A][6] == hashlib.sha256(npz.read_bytes()).hexdigest()
    assert calls(tmp) == [f"{A} {STEP} marked W=4096"], calls(tmp)   # 调用时标记非空（scripts/dev/remote 只认非空标记），世界数固定 4096
    assert not (tmp / f"runs/.chain-active-{NAME}").exists()
    assert (tmp / f"results/frames/{NAME}/{A}_s{STEP}.txt").exists()
    assert (tmp / f"results/frames/{NAME}/fc_src.txt").exists()
    assert log(tmp).splitlines()[-1].endswith("DUMP_CHAIN_DONE（成功 2、失败 2）"), log(tmp).splitlines()[-1]


def test_fail_continue(tmp: Path) -> None:
    setup(tmp, runs=(A, B))
    p = chain(tmp, **{k(A): "3"})
    assert p.returncode == 0, p.stderr
    t = table(tmp)
    assert (t[A][2], t[A][3], t[B][2]) == ("failed", "3", "ok"), t
    assert (tmp / f"results/frames/{NAME}/{A}_s{STEP}.txt.failed").exists()
    assert log(tmp).splitlines()[-1].endswith("DUMP_CHAIN_DONE（成功 1、失败 1）")


def test_gpu_wait_then_go(tmp: Path) -> None:
    setup(tmp, runs=(A,))
    (tmp / "runs/gpu_busy").write_text(str(int(time.time())))
    p = chain(tmp, FAKE_GPU_HOLD="3")
    assert p.returncode == 0, p.stderr
    assert "GPU 上有别的进程" in log(tmp) and "GPU 空了，接着做" in log(tmp)
    assert table(tmp)[A][2] == "ok"


def test_gpu_always_busy(tmp: Path) -> None:
    setup(tmp, runs=(A,))
    p = chain(tmp, FAKE_GPU="busy", V2D_GPU_WAIT_S="2")
    assert p.returncode == 1, p.stdout
    assert log(tmp).splitlines()[-1].endswith("DUMP_CHAIN_FAILED")
    assert calls(tmp) == [] and not (tmp / f"runs/.chain-active-{NAME}").exists()


def test_gpu_error_and_retry_limit(tmp: Path) -> None:
    setup(tmp, runs=(A,))
    p = chain(tmp, FAKE_GPU="error", V2D_GPU_WAIT_S="2")   # nvidia-smi 出错按被占处理
    assert p.returncode == 1 and calls(tmp) == [] and "nvidia-smi 出错" in log(tmp), log(tmp)
    d = tmp / "always6"
    d.mkdir()
    setup(d, runs=(A, B))
    p = chain(d, FAKE_ALWAYS6_RUN=A, V2D_GPU_WAIT_S="3")
    text = (d / f"runs/logs/{NAME}.log").read_text()
    assert p.returncode == 1 and "累计等 GPU 超过 3 秒" in text and text.splitlines()[-1].endswith("DUMP_CHAIN_FAILED"), text
    n = len([c for c in calls(d) if c.startswith(A + " ")])
    assert 2 <= n <= 5 and not any(c.startswith(B + " ") for c in calls(d)), calls(d)   # 重跑之间停 1 秒，不连发
    d = tmp / "nonl"
    d.mkdir()
    setup(d, runs=(A, B), lst=f"{A} {STEP}\n{B} {STEP}")   # 最后一行没有换行
    p = chain(d)
    t = {f[0]: f for f in (x.split("\t") for x in (d / f"runs/frames/{NAME}.tsv").read_text().splitlines()[1:])}
    assert p.returncode == 0 and t[A][2] == "ok" and t[B][2] == "ok", t


def test_busy_midway(tmp: Path) -> None:
    setup(tmp, runs=(A, B))
    p = chain(tmp, FAKE_BUSY_RUN=A)
    assert p.returncode == 0, p.stderr
    assert [c.split()[0] for c in calls(tmp)] == [A, A, B], calls(tmp)
    t = table(tmp)
    assert t[A][2] == "ok" and t[B][2] == "ok"


def test_fc_src(tmp: Path) -> None:
    setup(tmp, runs=(A,))
    p = chain(tmp, V2D_FC_SRC=None, FAKE_GPU="busy", V2D_GPU_WAIT_S="30")   # 先核副本：不用等满 GPU 就停
    assert p.returncode == 1 and calls(tmp) == [] and log(tmp).splitlines()[-1].endswith("DUMP_CHAIN_FAILED")
    assert "GPU 上有别的进程" not in log(tmp), log(tmp)
    shutil.rmtree(tmp)
    tmp.mkdir()
    setup(tmp, runs=(A,))
    p = chain(tmp, FAKE_SRC_RC="1")
    assert p.returncode == 1 and calls(tmp) == [] and log(tmp).splitlines()[-1].endswith("DUMP_CHAIN_FAILED")


def test_refuse(tmp: Path) -> None:
    cases = [("old_log", None), ("empty", "# 只有注释\n"), ("format", f"{A} {STEP} extra\n"), ("dup", f"{A} {STEP}\n{A} {STEP}\n"),
             ("badstep", f"{A} abc\n")]
    for label, lst in cases:
        d = tmp / label
        d.mkdir()
        setup(d, runs=(A,), lst=lst)
        if label == "old_log":
            (d / f"runs/logs/{NAME}.log").write_text("old\n")
        p = chain(d)
        assert p.returncode == 2, (label, p.stdout, p.stderr)
        assert calls(d) == [], label
        assert label == "old_log" or not (d / f"runs/logs/{NAME}.log").exists(), label
    d = tmp / "badname"
    d.mkdir()
    setup(d, runs=(A,))
    assert chain(d, name="../x").returncode == 2


def test_markers_once(tmp: Path) -> None:
    setup(tmp, runs=(A, B))
    p = chain(tmp, **{k(B): "5"})
    assert p.returncode == 0
    text = log(tmp)
    assert text.count("DUMP_CHAIN_DONE") == 1 and "DUMP_CHAIN_FAILED" not in text, text


def test_dryrun(tmp: Path) -> None:
    setup(tmp, runs=(A, B, C, D), dumped=(B,), nonfinite=(C,), no_policy=(D,))
    evaljson(tmp, A, world_count=64)
    p = chain(tmp, V2D_DUMP_CHAIN_DRYRUN="1", FAKE_SRC_RC="1", FAKE_DF_MB="10")
    assert p.returncode == 1, p.stdout
    out = p.stdout
    for frag in ("FAIL V2D_FC_SRC", "FAIL runs 所在分区", f"FAIL {A}：原评测", f"FAIL {C}：没有 NONFINITE", f"FAIL {D}：有 policy_{STEP}",
                 f"ok   {B}：已有 dump", "DUMP_DRYRUN_FAIL 5"):
        assert frag in out, (frag, out)
    assert not (tmp / f"runs/logs/{NAME}.log").exists() and not (tmp / f"runs/frames/{NAME}.tsv").exists()
    assert not (tmp / f"runs/.chain-active-{NAME}").exists() and calls(tmp) == []
    d = tmp / "good"
    d.mkdir()
    setup(d, runs=(A, B))
    evaljson(d, B, non_finite_worlds=[3])
    p = chain(d, V2D_DUMP_CHAIN_DRYRUN="1")
    assert p.returncode == 1 and f"FAIL {B}：原评测" in p.stdout and "DUMP_DRYRUN_FAIL 1" in p.stdout, p.stdout
    d = tmp / "allok"
    d.mkdir()
    setup(d, runs=(A, B))
    p = chain(d, V2D_DUMP_CHAIN_DRYRUN="1")
    assert p.returncode == 0 and p.stdout.rstrip().endswith("DUMP_DRYRUN_OK"), p.stdout
    d = tmp / "others"
    d.mkdir()
    setup(d, runs=(A,))
    (d / "runs/.chain-active-t1x_train_mf4090").write_text("x\n")
    p = chain(d, V2D_DUMP_CHAIN_DRYRUN="1", FAKE_DOCKER_LEFT="fc-evaldump-old")
    assert p.returncode == 1 and "FAIL 没有别的长链标记" in p.stdout and "FAIL 没有残留的 fc-evaldump-* 容器（fc-evaldump-old）" in p.stdout, p.stdout
    assert "DUMP_DRYRUN_FAIL 2" in p.stdout, p.stdout
    d = tmp / "smierr"
    d.mkdir()
    setup(d, runs=(A,))
    p = chain(d, V2D_DUMP_CHAIN_DRYRUN="1", FAKE_GPU="error")
    assert p.returncode == 1 and "FAIL nvidia-smi 能跑" in p.stdout, p.stdout


def main() -> None:
    tests = [test_normal, test_fail_continue, test_gpu_wait_then_go, test_gpu_always_busy, test_gpu_error_and_retry_limit, test_busy_midway,
             test_fc_src, test_refuse,
             test_markers_once, test_dryrun]
    for t in tests:
        with tempfile.TemporaryDirectory() as d:
            t(Path(d))
        print(f"ok {t.__name__}")
    print("ALL_OK")


if __name__ == "__main__":
    main()
