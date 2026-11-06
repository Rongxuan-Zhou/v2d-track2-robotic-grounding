"""arm_after.sh 的行为测试（暂无契约 ID：运行规范待成文）。只用标准库，Mac 与训练机都能跑。

临时目录当仓库根，runs/tools/ 下放真的 arm_after.sh，stages/policy/ 下放假的 arm_var_chain.sh（把 V2D_ARM_DRYRUN 与几个环境变量是否还在
记进 calls.txt，最后一行按环境变量给 DRYRUN_OK / ARMED 或别的）；PATH 前面放假的 nvidia-smi、pgrep、timeout
（gpu_busy 文件在时报一个 GPU 进程，smi_error 在时出错，cpu_busy 在时 pgrep 报一个导出进程；timeout 只是照跑）。轮询间隔 1 秒。
  1. 长链标记 2 秒后才消失：一直等，消失后先跑前置命令、再预演、再起链；起链时凭据、代理、前置命令都已不在环境里；最后一行 ARM_AFTER_ARMED；
  2. 上一段训练链 DAY2_TRAIN_FAILED、完成不够、评测有失败、导出缺或有失败：前置命令照跑（删除不看结局），但不预演、不起；
  3. 前置命令退出码非 0：不预演、不起；
  4. 卡一直被占、有导出进程、nvidia-smi 出错：停，不跑前置命令；卡占 2 秒后空出：照常往下；
  5. 预演最后一行不是 DRYRUN_OK：不真起；起链最后一行不是 ARMED：停；
  6. 标记一直不消失：等满次数后停；日志最后一行 2 秒后才变成结束行：一直等；
  7. 前置命令期间来历变了（有人同步）：不预演、不起；
  8. V2D_AFTER_SYNC_FROM（计划 agent-doc/plan-t12night-2026-10-30.md 第 2 节）：链结束后从 runs/stage/ 下的暂存目录同步进仓库副本
     （改了的队列换成暂存的、仓库副本独有的源码文件删掉，.env、data、runs、results 留着，来历写成暂存的），再预演、起链；
     暂存目录不完整、或等待期间仓库副本的来历变了（有人同步过）时同步不做、不起。
  9. V2D_POST_SYNC_CMD（计划 agent-doc/plan-t1hard-2026-10-31.md）：同步之后、预演之前跑；退出码非 0 不预演、不起；不同步时在前置命令之后跑；起链时已不在环境里。
用法：python3 stages/policy/test_arm_after.py
"""
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAKE_ARM = '''#!/usr/bin/env bash
echo "arm $1 DRYRUN=${V2D_ARM_DRYRUN:-0} TOKEN=${HF_TOKEN:+set} PROXY=${https_proxy:+set}${HTTP_PROXY:+set} EP=${HF_ENDPOINT:+set} PRE=${V2D_PRE_CMD:+set}" >> calls.txt
if [ "${V2D_ARM_DRYRUN:-0}" = 1 ]; then echo "预演，不起会话"; echo "${FAKE_DRY_LAST:-DRYRUN_OK}"; else echo "${FAKE_ARM_LAST:-ARMED}"; fi
'''
FAKE_SMI = '#!/usr/bin/env bash\n[ -e smi_error ] && { echo "NVIDIA-SMI has failed"; exit 9; }\n[ -e gpu_busy ] && echo "4242, 9000 MiB"\nexit 0\n'
FAKE_PGREP = '#!/usr/bin/env bash\n[ -e cpu_busy ] && { echo "777 bash stages/evaluate/fc_export.sh x"; exit 0; }\nexit 1\n'
FAKE_TIMEOUT = '#!/usr/bin/env bash\nshift\nexec "$@"\n'
TRAIN_DONE = "=== x DAY2_TRAIN_DONE（完成 5、失败 0、跳过 0、发散 0）"
EVAL_DONE = "=== x DAY2_EVAL_DONE（成功 15、失败 0）"
EXPORT_DONE = "=== x EXPORT_CHAIN_DONE（成功 5、失败 0）"
PRE_OK = 'echo "PRE ran" >> calls.txt; exit 0'
ARMED = ["PRE ran", "arm t3p4b DRYRUN=1 TOKEN= PROXY= EP= PRE=", "arm t3p4b DRYRUN=0 TOKEN= PROXY= EP= PRE="]


def run(train_last=TRAIN_DONE, eval_last=EVAL_DONE, export_last=EXPORT_DONE, marker_s=None, files=(), clear_after=None,
        late_train=None, pre=PRE_OK, setup=None, post=None, **env_extra) -> tuple:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        for d in ("runs/tools", "runs/logs", "stages/policy/queues", "stages/evaluate/exports", "bin"):
            (root / d).mkdir(parents=True)
        shutil.copy(HERE / "arm_after.sh", root / "runs/tools")
        shutil.copy(HERE / "stage_sync.sh", root / "runs/tools")
        for path, text in (("stages/policy/arm_var_chain.sh", FAKE_ARM), ("bin/nvidia-smi", FAKE_SMI), ("bin/pgrep", FAKE_PGREP),
                           ("bin/timeout", FAKE_TIMEOUT)):
            (root / path).write_text(text, encoding="utf-8")
            (root / path).chmod(0o755)
        (root / ".v2d-source-commit").write_text("a629fbdbf3e5\n")
        for q in ("mf4090_t3p4b.expect.tsv", "mf4090_t3p4b_smoke.txt", "mf4090_t3p4b.txt"):
            (root / "stages/policy/queues" / q).write_text(q + "\n")
        (root / "stages/evaluate/exports/cand_t3_t3p4b_mf4090.tsv").write_text("x\n")
        tl = root / "runs/logs/a_train.log"
        tl.write_text("=== x 正式 t3-ep3\n" + (train_last if late_train is None else "=== x 正式 t3-ep3 训练中") + "\n", encoding="utf-8")
        if late_train is not None:
            threading.Timer(late_train, lambda: tl.write_text(f"=== x 正式 t3-ep3\n{train_last}\n", encoding="utf-8")).start()
        (root / "runs/logs/a_eval.log").write_text("=== x 等训练链\n" + eval_last + "\n", encoding="utf-8")
        if export_last is not None:
            (root / "runs/logs/a_export.log").write_text("=== x 导出\n" + export_last + "\n", encoding="utf-8")
        for f in files:
            (root / f).write_text("")
        if clear_after:
            threading.Timer(clear_after, lambda: [(root / f).unlink() for f in files]).start()
        if marker_s is not None:
            mk = root / "runs/.chain-active-a_train_mf4090"
            mk.write_text("a")
            if marker_s:
                threading.Timer(marker_s, mk.unlink).start()
        if setup:
            setup(root)
        env = dict(os.environ, PATH=f"{root / 'bin'}:{os.environ['PATH']}", V2D_POLL_S="1", V2D_AFTER_MAX_POLLS="8",
                   V2D_GPU_MAX_POLLS="3", HF_TOKEN="hf_fake", https_proxy="http://p", HTTP_PROXY="http://p",
                   HF_ENDPOINT="https://huggingface.co", V2D_PRE_CMD=pre, **env_extra)
        t0 = time.time()
        r = subprocess.run(["bash", str(root / "runs/tools/arm_after.sh"), "t3p4b", "runs/logs/a_train.log", "runs/logs/a_eval.log",
                            "runs/logs/a_export.log", "5"], cwd=root, env=env, capture_output=True, text=True, errors="replace", timeout=90)
        calls = (root / "calls.txt").read_text().split("\n")[:-1] if (root / "calls.txt").exists() else []
        extra = post(root) if post else None
        return (r.returncode, calls, r.stdout + r.stderr, time.time() - t0) + ((extra,) if post else ())


def last(out: str) -> str:
    return out.rstrip().splitlines()[-1]


def test_waits_then_arms() -> None:
    rc, calls, out, took = run(marker_s=2)
    assert rc == 0 and took >= 2 and calls == ARMED, (rc, took, calls, out)
    assert last(out).endswith("ARM_AFTER_ARMED") and "训练链结束（DONE）" in out and "DAY2_TRAIN_DONE" not in out, out


def test_previous_outcome_blocks_arm_not_delete() -> None:
    cases = ((dict(train_last="=== x DAY2_TRAIN_FAILED"), "上一段训练没有全部完成"),
             (dict(train_last="=== x DAY2_TRAIN_DONE（完成 3、失败 2、跳过 0、发散 0）"), "上一段训练没有全部完成"),
             (dict(train_last="=== x DAY2_TRAIN_DONE（完成 4、失败 0、跳过 1、发散 0）"), "上一段训练没有全部完成"),
             (dict(eval_last="=== x DAY2_EVAL_DONE（成功 12、失败 3）"), "上一段评测有失败"),
             (dict(eval_last="=== x DAY2_EVAL_FAILED"), "上一段评测有失败"),
             (dict(export_last=None), "上一段导出没有全部成功"),
             (dict(export_last="=== x EXPORT_CHAIN_DONE（成功 4、失败 1）"), "上一段导出没有全部成功"),
             (dict(export_last="=== x EXPORT_CHAIN_FAILED"), "上一段导出没有全部成功"))
    for kw, want in cases:
        rc, calls, out, _ = run(**kw)
        assert rc == 1 and calls == ["PRE ran"] and want in last(out), (kw, rc, calls, out)


def test_independent_ignores_previous_outcome() -> None:
    # 计划 agent-doc/plan-t3-gb3-switch-2026-10-17.md 做法 1d：V2D_AFTER_INDEPENDENT=1 时上一段的失败只记录、照起；不设时照旧拦（上一个测试）
    for kw in (dict(train_last="=== x DAY2_TRAIN_DONE（完成 2、失败 1、跳过 0、发散 2）"), dict(eval_last="=== x DAY2_EVAL_FAILED"),
               dict(export_last=None), dict(export_last="=== x EXPORT_CHAIN_DONE（成功 4、失败 1）")):
        rc, calls, out, _ = run(V2D_AFTER_INDEPENDENT="1", **kw)
        assert rc == 0 and calls == ARMED and last(out).endswith("ARM_AFTER_ARMED") and "只记录、不拦" in out, (kw, rc, calls, out)
    rc, calls, out, _ = run(V2D_AFTER_INDEPENDENT="1", pre='echo "PRE ran" >> calls.txt; echo 1234567890ab > .v2d-source-commit')
    assert rc == 1 and calls == ["PRE ran"] and "来历或 t3p4b 的队列在等待期间变了" in last(out), (rc, calls, out)
    rc, calls, out, _ = run(V2D_AFTER_INDEPENDENT="1", marker_s=0)
    assert rc == 1 and calls == [] and "等了 8 次还没结束" in out, (rc, calls, out)


def test_pre_cmd_fails() -> None:
    rc, calls, out, _ = run(pre='echo "PRE ran" >> calls.txt; exit 3')
    assert rc == 1 and calls == ["PRE ran"] and last(out).endswith("ARM_AFTER_STOPPED 前置命令没过（退出码 3）"), (rc, calls, out)


def test_machine_busy() -> None:
    for files, want in ((("gpu_busy",), "卡上或机器上还有进程：4242"), (("cpu_busy",), "fc_export.sh"), (("smi_error",), "nvidia-smi 查不了 GPU")):
        rc, calls, out, _ = run(files=files)
        assert rc == 1 and calls == [] and want in last(out), (files, rc, calls, out)


def test_gpu_frees_up() -> None:
    rc, calls, out, took = run(files=("gpu_busy",), clear_after=2)
    assert rc == 0 and calls == ARMED and took >= 2, (rc, calls, out)


def test_dryrun_or_arm_not_ok() -> None:
    rc, calls, out, _ = run(FAKE_DRY_LAST="arm_var_chain: 可用空间只有 1200 MB（要求 2500 MB），停")
    assert rc == 1 and calls == ARMED[:2] and last(out).endswith("ARM_AFTER_STOPPED 预演没过（退出码 0）"), (rc, calls, out)
    rc, calls, out, _ = run(FAKE_ARM_LAST="没起全：会话 2 / 3")
    assert rc == 1 and calls == ARMED and last(out).endswith("ARM_AFTER_STOPPED 起链没起全（退出码 0）"), (rc, calls, out)


def test_waits_for_markers_and_logs() -> None:
    rc, calls, out, _ = run(marker_s=0)
    assert rc == 1 and calls == [] and "等了 8 次还没结束" in out and "chain-active-a_train_mf4090" in out, (rc, calls, out)
    rc, calls, out, took = run(late_train=2)
    assert rc == 0 and calls == ARMED and took >= 2, (rc, calls, out)


def test_source_changed() -> None:
    rc, calls, out, _ = run(pre='echo "PRE ran" >> calls.txt; echo 1234567890ab > .v2d-source-commit')
    assert rc == 1 and calls == ["PRE ran"] and "来历或 t3p4b 的队列在等待期间变了" in last(out), (rc, calls, out)


def _stage(root: Path, complete: bool = True) -> None:
    (root / ".v2d-remote-root").write_text("")
    for f, text in ((".env", "SECRET=1\n"), ("data/d.bin", "d"), ("results/r.txt", "r"), ("stages/old_only_here.sh", "x")):
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_text(text)
    st = root / "runs/stage/b0c0ffee1234"
    for d in ("stages/policy/queues", "stages/evaluate/exports"):
        (st / d).mkdir(parents=True)
    (st / ".v2d-source-commit").write_text("b0c0ffee1234\n")
    if complete:
        shutil.copy(root / "stages/policy/arm_var_chain.sh", st / "stages/policy/arm_var_chain.sh")
    for q in ("mf4090_t3p4b.expect.tsv", "mf4090_t3p4b_smoke.txt", "mf4090_t3p4b.txt"):
        (st / "stages/policy/queues" / q).write_text("new " + q + "\n")
    (st / "stages/evaluate/exports/cand_t3_t3p4b_mf4090.tsv").write_text("y\n")


def _after(root: Path) -> dict:
    return {"commit": (root / ".v2d-source-commit").read_text().strip(),
            "queue": (root / "stages/policy/queues/mf4090_t3p4b.txt").read_text().strip(),
            "kept": all((root / f).exists() for f in (".env", "data/d.bin", "results/r.txt", "runs/logs/a_train.log")),
            "old_gone": not (root / "stages/old_only_here.sh").exists()}


def test_sync_from_stage() -> None:
    rc, calls, out, _, st = run(setup=_stage, post=_after, V2D_AFTER_SYNC_FROM="runs/stage/b0c0ffee1234")
    # 前置命令的 calls.txt 在仓库根、暂存目录里没有，同步时被 --delete 删掉（与 scripts/dev/remote 的行为相同），所以只剩预演与起链两行
    assert rc == 0 and calls == ARMED[1:] and "STAGE_SYNC_DONE b0c0ffee1234" in out, (rc, calls, out)
    assert st == {"commit": "b0c0ffee1234", "queue": "new mf4090_t3p4b.txt", "kept": True, "old_gone": True}, st
    rc, calls, out, _, st = run(setup=_stage, post=_after, V2D_AFTER_SYNC_FROM="runs/stage/b0c0ffee1234",
                                pre='echo "PRE ran" >> calls.txt; echo 1234567890ab > .v2d-source-commit')
    assert rc == 1 and calls == ["PRE ran"] and "仓库副本的来历在等待期间变了" in last(out) and st["commit"] == "1234567890ab" and not st["old_gone"], (rc, calls, out, st)
    rc, calls, out, _, st = run(setup=lambda r: _stage(r, complete=False), post=_after, V2D_AFTER_SYNC_FROM="runs/stage/b0c0ffee1234")
    assert rc == 1 and calls == ["PRE ran"] and "ARM_AFTER_STOPPED 暂存目录同步没过" in out and "暂存目录不完整" in out and st["commit"] == "a629fbdbf3e5" and not st["old_gone"], (rc, calls, out, st)



def test_post_sync_cmd() -> None:
    post_ok = 'echo "POST ran PRE=${V2D_PRE_CMD:+set}" >> calls.txt; [ -f stages/policy/queues/mf4090_t3p4b.txt ] && grep -q new stages/policy/queues/mf4090_t3p4b.txt'
    arm_post = ARMED[1:]

    def stage_bin(root: Path) -> None:   # 假的 timeout 等在仓库根的 bin/ 里，同步的 --delete 会删掉；暂存目录里放一份（训练机上用系统的 timeout）
        _stage(root)
        shutil.copytree(root / "bin", root / "runs/stage/b0c0ffee1234/bin")

    # 同步之后跑：看得到暂存的新队列；calls.txt 被同步删掉，所以只剩后置命令与预演、起链
    rc, calls, out, _, st = run(setup=stage_bin, post=_after, V2D_AFTER_SYNC_FROM="runs/stage/b0c0ffee1234", V2D_POST_SYNC_CMD=post_ok)
    assert rc == 0 and calls == ["POST ran PRE=set"] + arm_post and "后置命令退出码 0" in out and st["commit"] == "b0c0ffee1234", (rc, calls, out)
    rc, calls, out, _, st = run(setup=stage_bin, post=_after, V2D_AFTER_SYNC_FROM="runs/stage/b0c0ffee1234",
                                V2D_POST_SYNC_CMD='echo "POST ran" >> calls.txt; exit 3')
    assert rc == 1 and calls == ["POST ran"] and "ARM_AFTER_STOPPED 后置命令没过（退出码 3）" in last(out), (rc, calls, out)
    # 不同步：前置命令之后跑；起链时环境里已经没有后置命令
    rc, calls, out, _ = run(V2D_POST_SYNC_CMD='echo "POST ran" >> calls.txt')
    assert rc == 0 and calls == ["PRE ran", "POST ran"] + ARMED[1:] and last(out).endswith("ARM_AFTER_ARMED"), (rc, calls, out)
    rc, calls, out, _ = run(V2D_POST_SYNC_CMD='env | grep -q "^V2D_POST_SYNC_CMD=" && echo "POST env" >> calls.txt; exit 0')
    assert calls[:2] == ["PRE ran", "POST env"] and all("POST" not in c for c in calls[2:]), (rc, calls, out)


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_") and callable(f):
            f()
            print("ok", name)
    print("ALL_OK")
