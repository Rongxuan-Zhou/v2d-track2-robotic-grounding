"""hf_archive_delete_checked.sh 的行为测试（暂无契约 ID：运行规范待成文）。只用标准库，Mac 与训练机都能跑。

临时目录当仓库根，runs/tools/ 下放真的 hf_archive_delete_checked.sh 与假的 hf_archive.py、hf_archive_run.sh、hf_archive_report.py
（把收到的子命令记进 calls.txt，按环境变量给出成功或失败的输出与退出码）。清单 3 个文件、各 1 MiB，删除前可用 1000 MB。
  1. 都过：依次 verify、演习、删除（V2D_HF_ARCHIVE_STEPS=delete）、报告，最后一行 HF_DELETE_CHECKED_DONE；三份日志都落盘；
  2. 对齐核对没过（HF_VERIFY_FAIL，退出码 1）：不演习、不删；
  3. 演习“将删”2 个（清单 3 个）、4 个（多了）、有“跳过”、演习自己退出码 1（例如有长链标记）：都停在演习，不删；
  4. 删除一步退出码非 0、最后一行 HF_ARCHIVE_FILES_CHECK_FAILED、有没删的文件、可用空间对不上：都停在删除，不报告；
  5. 缺第一段的核对或抽查结果、删除日志已存在：什么都不跑；
  6. 带 token 样式的行不进日志。
用法：python3 tools/test_hf_archive_delete_checked.py
"""
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAKE_ARCHIVE = r'''import os, sys
open("calls.txt", "a").write(sys.argv[1] + "\n")
if sys.argv[1] == "verify":
    print("  ok  a  hf_" + "A" * 24)
    if os.environ.get("FAKE_VERIFY") == "fail":
        print("2 / 3 个文件核对一致；HF_VERIFY_FAIL 1"); sys.exit(1)
    print("3 / 3 个文件核对一致；HF_VERIFY_OK")
elif sys.argv[1] == "delete":
    if os.environ.get("FAKE_DRY_RC") == "1":
        sys.exit("有训练链在跑（runs/.chain-active-*），policy 一个都不删")
    for i in range(int(os.environ.get("FAKE_WILL", "3"))):
        print(f"将删 r/policy_{i}.safetensors（1 MB，在 x:y）")
    for i in range(int(os.environ.get("FAKE_SKIP", "0"))):
        print(f"跳过 r/policy_{i}.safetensors：和清单里的不一样了")
    print("预计释放约 0.00 GB（演习，没删）")
'''
FAKE_RUN = '''#!/usr/bin/env bash
echo "run STEPS=${V2D_HF_ARCHIVE_STEPS:-} VIS=${V2D_HF_VISIBILITY:-} $*" >> calls.txt
echo "=== x 开始（步骤 ${V2D_HF_ARCHIVE_STEPS:-}）；可用 1000 MB；xet 缓存 5 MB"
echo "=== x runs/flash_chord：删除前 10 个文件、删除后 7 个；消失 3 个（清单 3 个，其中没删的 ${FAKE_KEPT:-0} 个）；清单外消失的文件：无"
echo "=== x 删除后可用 ${FAKE_AFTER:-1003} MB"
echo "=== x ${FAKE_RUN_LAST:-HF_ARCHIVE_DONE}"
exit ${FAKE_RUN_RC:-0}
'''
FAKE_REPORT = 'import sys\nopen("calls.txt", "a").write("report\\n")\nprint("## 清单：3 个文件")\n'


def run(verified: bool = True, spot: bool = True, dl_exists: bool = False, **fake) -> tuple:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        tools = root / "runs/tools"
        tools.mkdir(parents=True)
        shutil.copy(HERE / "hf_archive_delete_checked.sh", tools)
        (tools / "hf_archive.py").write_text(FAKE_ARCHIVE, encoding="utf-8")
        (tools / "hf_archive_run.sh").write_text(FAKE_RUN, encoding="utf-8")
        (tools / "hf_archive_report.py").write_text(FAKE_REPORT, encoding="utf-8")
        (root / "runs/m.json").write_text(json.dumps([{"file": f"policy_{i}.safetensors", "size": 1048576} for i in range(3)]))
        if verified:
            (root / "runs/out.verified.json").write_text("[1]")
        if spot:
            (root / "runs/out.spotcheck.json").write_text('{"ok": true}')
        if dl_exists:
            (root / "runs/logs").mkdir(parents=True)
            (root / "runs/logs/out_delete.log").write_text("old\n")
        env = dict(os.environ, V2D_HF_VISIBILITY="public", **fake)
        r = subprocess.run(["bash", str(tools / "hf_archive_delete_checked.sh"), "runs/m.json", "u/repo", "mf4090", "runs/out"],
                           cwd=root, env=env, capture_output=True, text=True, errors="replace", timeout=60)
        calls = (root / "calls.txt").read_text().split("\n")[:-1] if (root / "calls.txt").exists() else []
        logs = {p.name: p.read_text(encoding="utf-8") for p in [root / "runs/out.precheck.log", root / "runs/out.dryrun.log",
                                                                  root / "runs/logs/out_delete.log"] if p.exists()}
        return r.returncode, calls, r.stdout, logs


def last(out: str) -> str:
    return out.rstrip().splitlines()[-1]


def test_all_pass() -> None:
    rc, calls, out, logs = run()
    assert rc == 0 and last(out).endswith("HF_DELETE_CHECKED_DONE"), (rc, out)
    assert calls == ["verify", "delete", "run STEPS=delete VIS=public runs/m.json u/repo mf4090 runs/out", "report"], calls
    assert "将删 3 个（清单 3 个），跳过 0 个" in out and "增加 3 MB；清单 3 MiB；相差 0" in out, out
    assert set(logs) == {"out.precheck.log", "out.dryrun.log", "out_delete.log"} and "HF_ARCHIVE_DONE" in logs["out_delete.log"], logs


def test_token_lines_filtered() -> None:
    rc, _, out, logs = run()
    assert rc == 0 and "hf_AAAA" not in out and "hf_AAAA" not in logs["out.precheck.log"], (out, logs)


def test_verify_fail() -> None:
    rc, calls, out, _ = run(FAKE_VERIFY="fail")
    assert rc == 1 and calls == ["verify"] and last(out).endswith("STOPPED 对齐核对（退出码 1）"), (rc, calls, out)


def test_dryrun_problems() -> None:
    for fake, want in (({"FAKE_WILL": "2"}, "将删 2 / 3"), ({"FAKE_WILL": "4"}, "将删 4 / 3"), ({"FAKE_SKIP": "1"}, "跳过 1"),
                       ({"FAKE_DRY_RC": "1"}, "退出码 1")):
        rc, calls, out, _ = run(**fake)
        assert rc == 1 and calls == ["verify", "delete"] and "STOPPED 演习" in last(out) and want in last(out), (fake, rc, calls, out)


def test_delete_problems() -> None:
    cases = (({"FAKE_RUN_RC": "1", "FAKE_RUN_LAST": "HF_ARCHIVE_STOPPED delete（GPU 上有进程：123）"}, "删除（退出码 1）"),
             ({"FAKE_RUN_RC": "1", "FAKE_RUN_LAST": "HF_ARCHIVE_FILES_CHECK_FAILED"}, "删除（退出码 1）"),
             ({"FAKE_RUN_LAST": "HF_ARCHIVE_STOPPED reverify"}, "删除（退出码 0）"),
             ({"FAKE_KEPT": "1"}, "有清单里没删的文件"),
             ({"FAKE_AFTER": "1100"}, "相差 97 MB，超过 50"))
    for fake, want in cases:
        rc, calls, out, logs = run(**fake)
        assert rc == 1 and calls[-1].startswith("run STEPS=delete") and "report" not in calls, (fake, rc, calls)
        assert want in last(out) and "out_delete.log" in logs, (fake, out)


def test_missing_inputs() -> None:
    for kw, want in (({"spot": False}, "缺第一段的核对或抽查结果"), ({"verified": False}, "缺第一段的核对或抽查结果"),
                     ({"dl_exists": True}, "同一批只删一次")):
        rc, calls, out, _ = run(**kw)
        assert rc == 1 and calls == [] and want in out, (kw, rc, calls, out)


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_") and callable(f):
            f()
            print("ok", name)
    print("ALL_OK")
