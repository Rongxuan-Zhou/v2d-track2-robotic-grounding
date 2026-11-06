"""diag2_chain.sh（连同 scripts/dev/record）的行为测试（暂无契约 ID：运行规范待成文）。只用标准库。

临时目录里搭最小仓库：真的 diag2_chain.sh 与 scripts/dev/record；假的 scripts/dev/in-flash-chord（按命令里的脚本名模拟诊断 rollout、
同步检查与分析，退出码由环境变量定；设了 V2D_FC_SRC 就报错），PATH 前面放假的 nvidia-smi。计划第 5 节第 4 步。
  1. 正常：第一行先试跑 16 个世界、再同步检查，三行都诊断、分析，DIAG2_DONE（完成 3…）；rollout 走 GPU、同步与分析不走；
     即使外面设了 V2D_FC_SRC，传给容器的环境里也没有它；hydra 目录删掉、npz 留着
  2. 试跑的同步闸不过：DIAG2_FAILED，不跑 4096 世界
  3. 分析退出码 3、4、其他，rollout 失败，缺本机评测日志：分别记闸不过、对照不过、出错，接着做下一行
  4. 拒绝：日志已存在、运行表的行不对
  5. 等导出链日志结束才开始，本日志里没有结束标记的原文；GPU 先被占、后空出来；一直被占则 DIAG2_FAILED；
     做完一行后 GPU 被占，下一行用 GPU 前会等它空出来（第 2 版）
  6. 本机评测日志里有 in-flash-chord 的挂载说明行（那份评测挂的是源码副本）：这一行跳过、记出错（第 2 版）
  7. 设了 V2D_DIAG_FC_SRC（plan-t2dg）：挂载行指向同一份副本的行照做，容器拿到的 V2D_FC_SRC 就是它；没有挂载行、指向别的副本（含前缀相同的路径）的行跳过；
     给的路径末尾带斜杠或经软链接时照样匹配（按 pwd -P 解析）；相对路径、不存在的目录拒绝；Tier 2 的 run 名可以进运行表
用法：python3 stages/evaluate/test_diag2_chain.py
"""
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS = ["t1-ep8-w2048-cap4m-mf4090", "t1-ep21-w2048-cap4m-mf4090", "t1-ep16-w2048-cap4m-mf4090"]
ROLES = ["control", "main", "main"]
FAKE_FC = r'''#!/usr/bin/env bash
cmd="$*"
echo "${V2D_GPU:-}|${V2D_CONTAINER_NAME:-}|${V2D_FC_SRC-unset}|$cmd" >> runs/fake_calls.txt
[ "${V2D_FC_SRC:-}" = "${FAKE_EXPECT_SRC:-}" ] || { echo "fake: V2D_FC_SRC 不对" >&2; exit 9; }
case "$cmd" in
  *diag_rollout.py*)
    run=$(sed -E 's|.*/runs/flash_chord/([^/]+)/policy.*|\1|' <<< "$cmd"); key="FAKE_ROLLOUT_RC_${run//-/_}"; rc="${!key:-0}"
    out=$(sed -E 's|.* /repo/(runs/diag2/[^ ]+) --worlds.*|\1|' <<< "$cmd"); mkdir -p "${out}_hydra"
    [ "$rc" = 0 ] && touch "$out.npz" "$out.json"
    if [ -n "${FAKE_BUSY_AFTER:-}" ] && [[ $cmd == *"--worlds 4096"* ]] && [ ! -f runs/busy_done ]; then date +%s > runs/gpu_busy; touch runs/busy_done; fi
    echo "fake rollout $run rc $rc"; exit "$rc" ;;
  *diag2_sync.py*)
    if [ "${FAKE_SYNC:-ok}" = ok ]; then echo "robot_palm_pos (708, 16, 2, 3)，同步最大差 0 m"; echo SYNC_OK; exit 0; fi
    echo "同步最大差 0.01 m"; echo SYNC_FAIL; exit 1 ;;
  *diag_phase.py*)
    run=$(sed -E 's|.*/runs/diag2/([^/]+)/rollout.*|\1|' <<< "$cmd"); key="FAKE_PHASE_RC_${run//-/_}"; rc="${!key:-0}"
    echo "  主要失败环节（第 4 节的规则）：抓取"; exit "$rc" ;;
esac
exit 7
'''
FAKE_GPU = r'''#!/usr/bin/env bash
[ "${FAKE_GPU:-}" = busy ] && { echo 4242; exit 0; }
[ -f runs/gpu_busy ] && [ $(( $(date +%s) - $(cat runs/gpu_busy) )) -lt 2 ] && echo 4242
exit 0
'''


def setup(root: Path, table=None, missing_log=(), mounted_log=(), mount_src="/x/fc_src/v040_p185", runs=RUNS) -> None:
    for d in ("stages/evaluate/diag2", "scripts/dev", "bin", "runs/logs"):
        (root / d).mkdir(parents=True, exist_ok=True)
    for f in ("stages/evaluate/diag2_chain.sh", "scripts/dev/record"):
        shutil.copy(REPO / f, root / f); (root / f).chmod(0o755)
    (root / "scripts/dev/in-flash-chord").write_text(FAKE_FC); (root / "scripts/dev/in-flash-chord").chmod(0o755)
    (root / "bin/nvidia-smi").write_text(FAKE_GPU); (root / "bin/nvidia-smi").chmod(0o755)
    (root / ".v2d-source-commit").write_text("deadbeef\n")
    rows = table if table is not None else "".join(f"{r}\t349999104\t{role}\n" for r, role in zip(runs, ROLES))
    (root / "stages/evaluate/diag2/mf.tsv").write_text("# 注释\n" + rows)
    for r in runs:
        (root / f"runs/flash_chord/{r}").mkdir(parents=True, exist_ok=True)
        (root / f"runs/flash_chord/{r}/policy_349999104.safetensors").write_text("")
        (root / f"runs/flash_chord/{r}/eval_349999104_w4096.json").write_text("{}")
        if r not in missing_log:
            src = mount_src.get(r) if isinstance(mount_src, dict) else mount_src
            body = f"in-flash-chord: /workspace ← {src}（flash_chord x）\n" if r in mounted_log else "x\n"
            (root / f"runs/logs/{r}.eval_349999104_w4096.log").write_text(body)


def env(root: Path, **kw) -> dict:
    e = dict(os.environ, PATH=f"{root / 'bin'}:{os.environ['PATH']}", V2D_POLL_S="1", V2D_FC_SRC="/should/not/pass")
    e.update(kw)
    return {k: v for k, v in e.items() if v is not None}


def chain(root: Path, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "stages/evaluate/diag2_chain.sh", "stages/evaluate/diag2/mf.tsv", "diag2_mf"], cwd=root, env=env(root, **kw),
                          capture_output=True, text=True, errors="replace", timeout=120)


def log(root: Path) -> str:
    return (root / "runs/logs/diag2_mf.log").read_text()


def calls(root: Path) -> list:
    f = root / "runs/fake_calls.txt"
    return f.read_text().splitlines() if f.exists() else []


def k(prefix: str, run: str) -> str:
    return f"{prefix}{run.replace('-', '_')}"


def test_normal() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root)
        r = chain(root)
        lg = log(root); cs = calls(root)
        assert r.returncode == 0 and lg.rstrip().endswith("DIAG2_DONE（完成 3、闸不过 0、对照不过 0、出错 0）"), lg
        smoke = [c for c in cs if "--worlds 16" in c]
        assert len(smoke) == 1 and RUNS[0] in smoke[0] and smoke[0].startswith("1|"), cs
        assert sum("--worlds 4096 --palms" in c for c in cs) == 3 and all(c.startswith("1|") for c in cs if "diag_rollout" in c), cs
        assert all(c.startswith("0|") for c in cs if "diag_phase" in c or "diag2_sync" in c), cs
        assert all(c.split("|")[2] == "unset" for c in cs), cs
        assert any("--role control" in c and RUNS[0] in c for c in cs) and any("--role main" in c and RUNS[1] in c for c in cs), cs
        for run in RUNS:
            assert (root / f"runs/diag2/{run}/rollout_w4096.npz").exists() and not (root / f"runs/diag2/{run}/rollout_w4096_hydra").exists()
            assert (root / f"results/diag2/phase_{run}.txt").exists() and (root / f"results/diag2/rollout_{run}.txt").exists()
        assert not (root / f"runs/diag2/{RUNS[0]}/smoke_w16_hydra").exists()
        assert "试跑通过（robot_palm_pos (708, 16, 2, 3)，同步最大差 0 m）" in lg and "主要失败环节（第 4 节的规则）：抓取" in lg, lg
        assert not (root / "runs/.chain-active-diag2_mf").exists()


def test_smoke_sync_fail() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root)
        r = chain(root, FAKE_SYNC="fail")
        lg = log(root)
        assert r.returncode == 1 and lg.rstrip().endswith("DIAG2_FAILED") and "同步闸不过" in lg, lg
        assert not any("--worlds 4096" in c for c in calls(root)), calls(root)


def test_row_outcomes() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root)
        r = chain(root, **{k("FAKE_PHASE_RC_", RUNS[0]): "4", k("FAKE_PHASE_RC_", RUNS[1]): "3", k("FAKE_PHASE_RC_", RUNS[2]): "5"})
        lg = log(root)
        assert r.returncode == 0 and lg.rstrip().endswith("DIAG2_DONE（完成 0、闸不过 1、对照不过 1、出错 1）"), lg
        assert (root / f"results/diag2/phase_{RUNS[0]}.txt.failed").exists()
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root, missing_log=(RUNS[2],))
        r = chain(root, **{k("FAKE_ROLLOUT_RC_", RUNS[1]): "1"})
        lg = log(root)
        assert r.returncode == 0 and lg.rstrip().endswith("DIAG2_DONE（完成 1、闸不过 0、对照不过 0、出错 2）"), lg
        assert not any(RUNS[2] in c for c in calls(root)) and not any("diag_phase" in c and RUNS[1] in c for c in calls(root)), calls(root)


def test_refusals() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root)
        (root / "runs/logs/diag2_mf.log").write_text("old\n")
        r = chain(root)
        assert r.returncode == 2 and (root / "runs/logs/diag2_mf.log").read_text() == "old\n" and calls(root) == [], r.stderr
    for bad in (f"{RUNS[0]}\t349999104\tboss\n", f"t3-ep1-x\t349999104\tmain\n", "# 只有注释\n"):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); setup(root, table=bad)
            r = chain(root)
            assert r.returncode == 2 and not (root / "runs/logs/diag2_mf.log").exists(), (bad, r.stderr)


def test_waits_and_gpu() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root)
        prev = root / "runs/logs/export_mf.log"
        prev.write_text("=== x 等 runs/logs/box_eval_mf.log 结束（它的最后一行变成评测链的结束行）\n")
        threading.Timer(3, lambda: prev.write_text("=== x EXPORT_CHAIN_DONE（成功 30、失败 0）\n")).start()
        t0 = time.time()
        r = chain(root, V2D_WAIT_FOR_LOG="runs/logs/export_mf.log")
        lg = log(root)
        assert r.returncode == 0 and time.time() - t0 >= 3 and "上一段已结束：=== x 导出链完成（成功 30、失败 0）" in lg, lg
        assert "EXPORT_CHAIN_" not in lg and sum("DIAG2_" in line for line in lg.splitlines()) == 1, lg
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root)
        (root / "runs/gpu_busy").write_text(str(int(time.time())))
        r = chain(root, V2D_GPU_WAIT_S="30")
        assert r.returncode == 0 and "GPU 空了" in log(root), log(root)
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root)
        r = chain(root, FAKE_GPU="busy", V2D_GPU_WAIT_S="2")
        assert r.returncode == 1 and log(root).rstrip().endswith("DIAG2_FAILED") and calls(root) == [], log(root)


def test_mid_chain_gpu_and_mounted_log() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root)
        r = chain(root, FAKE_BUSY_AFTER="1", V2D_GPU_WAIT_S="30")
        lg = log(root)
        assert r.returncode == 0 and "GPU 空了" in lg and lg.rstrip().endswith("DIAG2_DONE（完成 3、闸不过 0、对照不过 0、出错 0）"), lg
        i_busy = lg.index("GPU 上有别的进程")
        assert lg.index(f"{RUNS[0]}：分析") < i_busy < lg.index(f"{RUNS[1]}：诊断 rollout"), lg
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root, mounted_log=(RUNS[1],))
        r = chain(root)
        lg = log(root)
        assert r.returncode == 0 and "挂的是源码副本" in lg and lg.rstrip().endswith("DIAG2_DONE（完成 2、闸不过 0、对照不过 0、出错 1）"), lg
        assert not any(RUNS[1] in c for c in calls(root)), calls(root)


def test_diag_src() -> None:
    t2 = ["t2-ep0-w2048-cap10m-r3a-ml40", "t2-ep25-w2048-cap10m-r3a-ml40", "t2-ep4-w2048-cap10m-r3a-ml40"]
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); (root / "fc_src/fc_8c778f9").mkdir(parents=True); (root / "link").symlink_to(root / "fc_src")
        src = str((root / "fc_src/fc_8c778f9").resolve())          # in-flash-chord 打印的是 pwd -P
        setup(root, runs=t2, mounted_log=tuple(t2), mount_src=src)
        for given in (src, src + "/", str(root / "link/fc_8c778f9")):   # 原样、末尾斜杠、经软链接
            for f in (root / "runs/logs").glob("diag2_mf.log"):
                f.unlink()
            (root / "runs/fake_calls.txt").unlink(missing_ok=True)
            r = chain(root, V2D_DIAG_FC_SRC=given, FAKE_EXPECT_SRC=src)
            lg = log(root); cs = calls(root)
            assert r.returncode == 0 and lg.rstrip().endswith("DIAG2_DONE（完成 3、闸不过 0、对照不过 0、出错 0）"), (given, lg)
            assert cs and all(c.split("|")[2] == src for c in cs) and f"运行时为源码副本 {src}" in lg, (given, cs, lg)
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); (root / "fc_src/fc_8c778f9").mkdir(parents=True)
        src = str((root / "fc_src/fc_8c778f9").resolve())
        setup(root, runs=t2, mounted_log=(t2[0], t2[2]), mount_src={t2[0]: src, t2[2]: src + "_x"})
        r = chain(root, V2D_DIAG_FC_SRC=src, FAKE_EXPECT_SRC=src)
        lg = log(root); cs = calls(root)
        assert r.returncode == 0 and lg.rstrip().endswith("DIAG2_DONE（完成 1、闸不过 0、对照不过 0、出错 2）"), lg
        assert "挂载行：没有" in lg and f"挂载行：in-flash-chord: /workspace ← {src}_x" in lg, lg
        assert not any(t2[1] in c or t2[2] in c for c in cs), cs
    for bad in ("lab/fc_src", "/no/such/dir"):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); setup(root, runs=t2)
            r = chain(root, V2D_DIAG_FC_SRC=bad)
            assert r.returncode == 2 and not (root / "runs/logs/diag2_mf.log").exists(), (bad, r.stderr)

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"ok  {name}")
    print("ALL_OK")
