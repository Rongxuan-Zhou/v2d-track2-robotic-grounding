"""ep7_ckpt_eval.sh（连同 scripts/dev/record、stages/policy/ckpt_nonfinite.py）的行为测试（暂无契约 ID：运行规范待成文）。只用标准库。

临时目录里搭最小仓库：真的 ep7_ckpt_eval.sh、scripts/dev/record、ckpt_nonfinite.py；假的 stages/policy/fc_eval.sh（按环境变量写评测 JSON 与评测日志，
退出码照真的：0、2（评测失败）、6（GPU 被占））、假的 fc_src_check.py（按清单名认副本）、假的 scripts/dev/in-flash-chord（模拟取证 rollout，
可以像 hydra 那样先打印 Traceback 再成功退出）、PATH 前面放假的 nvidia-smi。计划 agent-doc/plan-ep7-eval-and-guards-2026-10-02.md 的 A 部分（第 3 版）。
  1. 评测成功、JSON 四项都对：A-2 满足；取证照跑（挂 #185 副本、走 GPU）；评测挂 8c778f9 副本；V2D_EVAL_WORLDS、V2D_ALLOW_SHARED_GPU 不漏进去；
     最后一行 EP7_EVAL_DONE（评测 成功；评测器记的非有限世界 0；A-2 满足；取证 完成（崩掉的世界 0））
  2. 评测成功但有非有限世界、帧数不对、sha256 不对、世界数不对、缺 non_finite_worlds 键：A-2 不满足，缺键记未知
  3. 评测报终止报错（退出码 2，日志里有 warning 与 ValueError）：记“失败（终止报错）”与评测器报的世界数；取证输出里带 Traceback 仍记完成
  4. 评测因别的原因失败：记“失败（其他，等于没评成…）”；取证失败记失败
  5. fc_eval 第一次退出码 6：等 GPU 后重跑一次成功
  6. 拒绝（退出码 2、不写日志）：日志已存在、policy 不在、policy 含 NaN、评测 JSON 已存在、取证 npz 已存在、两份副本填反
  7. GPU 一直被占：评测前超时是 EP7_EVAL_FAILED；评测后超时记“取证 没跑（GPU 被占）”，仍是 EP7_EVAL_DONE
  8. 试跑模式：打印 EP7_EVAL_DRYRUN_OK，不写日志、不调评测
用法：python3 stages/evaluate/test_ep7_ckpt_eval.py
"""
import hashlib
import json
import os
import shutil
import struct
import subprocess
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUN, STEP, FRAMES, NAME = "t1-ep7-w2048-cap4m-p185-mf4090", "130004992", "896", "ep7_eval_mf"
FAKE_EVAL = r'''#!/usr/bin/env bash
echo "eval|${V2D_FC_SRC-unset}|${V2D_EVAL_WORLDS-unset}|${V2D_ALLOW_SHARED_GPU-unset}|$*" >> runs/fake_calls.txt
O="runs/flash_chord/$1"; ELOG="runs/logs/$1.eval_$2_w4096.log"
n=$(grep -c '^eval|' runs/fake_calls.txt)
if [ -n "${FAKE_BUSY_FIRST:-}" ] && [ "$n" = 1 ]; then echo "fc_eval: GPU 上有别的进程（PID 4242），拒绝评测" >&2; exit 6; fi
{ echo "in-flash-chord: /workspace ← ${V2D_FC_SRC}（假副本）"; echo "=== x fc_eval $1 policy_$2 world_count=4096 ==="; echo "evaluation step 896/896"; } >> "$ELOG"
case "${FAKE_EVAL_MODE:-ok}" in
  termination)
    { echo "warning: 7 worlds recorded non-finite poses and are excluded"; echo "Error executing job with overrides: []"
      echo "ValueError: evaluation must disable task failure termination before scoring the full reference"; } >> "$ELOG"
    echo "fc_eval: 评测失败（退出码 1），见 $ELOG" >&2; exit 2 ;;
  broken)
    echo "RuntimeError: CUDA out of memory" >> "$ELOG"; echo "fc_eval: 评测失败（退出码 1），见 $ELOG" >&2; exit 2 ;;
esac
sha=$(shasum -a 256 "$O/policy_$2.safetensors" 2>/dev/null | cut -d' ' -f1); [ -n "$sha" ] || sha=$(sha256sum "$O/policy_$2.safetensors" | cut -d' ' -f1)
[ -n "${FAKE_SHA:-}" ] && sha="$FAKE_SHA"
if [ "${FAKE_NF:-[]}" = missing ]; then
  printf '{"world_count": %s, "step_count": %s, "checkpoint_sha256": "%s"}\n' "${FAKE_WORLDS:-4096}" "${FAKE_STEPS:-896}" "$sha" > "$O/eval_$2_w4096.json"
else
  printf '{"world_count": %s, "non_finite_worlds": %s, "step_count": %s, "checkpoint_sha256": "%s"}\n' "${FAKE_WORLDS:-4096}" "${FAKE_NF:-[]}" "${FAKE_STEPS:-896}" "$sha" > "$O/eval_$2_w4096.json"
fi
echo "fc_eval: 假评测 $1 policy_$2"
'''
FAKE_SRC_CHECK = r'''import sys
from pathlib import Path
src, manifest = Path(sys.argv[1]), Path(sys.argv[2]).name
kind = (src / "KIND").read_text().strip() if (src / "KIND").exists() else ""
ok = (src / "SOURCE_V2D.txt").exists() and bool(kind) and kind in manifest
print("FC_SRC_OK" if ok else "FC_SRC_FAIL 1")
sys.exit(0 if ok else 1)
'''
FAKE_FC = r'''#!/usr/bin/env bash
cmd="$*"
echo "roll|${V2D_GPU:-}|${V2D_FC_SRC-unset}|$cmd" >> runs/fake_calls.txt
out=$(sed -E 's|.* /repo/(runs/ep7/[^ ]+) --worlds.*|\1|' <<< "$cmd"); mkdir -p "${out}_hydra"
rc="${FAKE_ROLL_RC:-0}"
if [ -n "${FAKE_ROLL_TRACEBACK:-}" ]; then
  echo "Error executing job with overrides: []"; echo "Traceback (most recent call last):"
  echo "ValueError: evaluation must disable task failure termination before scoring the full reference"
fi
if [ "$rc" = 0 ]; then echo npz > "$out.npz"; echo "BLOWUP_ROLLOUT_OK bad_worlds=${FAKE_BAD:-0} evaluator_non_finite=0 terminated=0"; fi
exit "$rc"
'''
FAKE_GPU = r'''#!/usr/bin/env bash
[ "${FAKE_GPU:-}" = busy ] && echo 4242
[ -n "${FAKE_GPU_BUSY_AFTER_EVAL:-}" ] && [ -f "runs/flash_chord/$FAKE_GPU_BUSY_AFTER_EVAL" ] && echo 4242
exit 0
'''


def write_policy(path: Path, values) -> None:
    """最小的 safetensors：一个 F32 张量。mtime 往前拨，ckpt_nonfinite.py 才认它写完了。"""
    data = struct.pack(f"<{len(values)}f", *values)
    header = json.dumps({"w": {"dtype": "F32", "shape": [len(values)], "data_offsets": [0, len(data)]}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + data)
    old = time.time() - 60
    os.utime(path, (old, old))


def setup(root: Path, policy=(1.0, 2.0, 3.0)) -> None:
    for d in ("stages/evaluate", "stages/policy", "scripts/dev", "bin", "runs/logs", f"runs/flash_chord/{RUN}", "third_party/patches"):
        (root / d).mkdir(parents=True, exist_ok=True)
    for f in ("stages/evaluate/ep7_ckpt_eval.sh", "scripts/dev/record", "stages/policy/ckpt_nonfinite.py"):
        shutil.copy(REPO / f, root / f); (root / f).chmod(0o755)
    for f, body in (("stages/policy/fc_eval.sh", FAKE_EVAL), ("scripts/dev/in-flash-chord", FAKE_FC), ("bin/nvidia-smi", FAKE_GPU),
                    ("stages/policy/fc_src_check.py", FAKE_SRC_CHECK)):
        (root / f).write_text(body); (root / f).chmod(0o755)
    (root / ".v2d-source-commit").write_text("deadbeef\n")
    for s, kind in (("src8c", "8c778f9"), ("srcp185", "p185")):
        (root / s / "src/flash_chord").mkdir(parents=True)
        (root / s / "SOURCE_V2D.txt").write_text(f"假副本 {s}\n")
        (root / s / "KIND").write_text(kind + "\n")
    if policy is not None:
        write_policy(root / f"runs/flash_chord/{RUN}/policy_{STEP}.safetensors", policy)


def chain(root: Path, **kw) -> subprocess.CompletedProcess:
    e = dict(os.environ, PATH=f"{root / 'bin'}:{os.environ['PATH']}", V2D_POLL_S="1",
             V2D_EVAL_SRC=str(root / "src8c"), V2D_ROLLOUT_SRC=str(root / "srcp185"), V2D_EVAL_WORLDS="256", V2D_ALLOW_SHARED_GPU="1")
    e.update(kw)
    e = {k: v for k, v in e.items() if v is not None}
    return subprocess.run(["bash", "stages/evaluate/ep7_ckpt_eval.sh", RUN, STEP, FRAMES, NAME], cwd=root, env=e,
                          capture_output=True, text=True, errors="replace", timeout=120)


def log(root: Path) -> str:
    return (root / f"runs/logs/{NAME}.log").read_text(errors="replace")


def last(root: Path) -> str:
    return log(root).strip().splitlines()[-1]


def calls(root: Path) -> list:
    f = root / "runs/fake_calls.txt"
    return f.read_text().splitlines() if f.exists() else []


def test_clean() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d); setup(root)
        r = chain(root)
        assert r.returncode == 0, r.stderr
        assert "EP7_EVAL_DONE（评测 成功；评测器记的非有限世界 0；A-2 满足；取证 完成（崩掉的世界 0））" in last(root), log(root)
        c = calls(root)
        assert len(c) == 2 and c[0] == f"eval|{root / 'src8c'}|unset|unset|{RUN} {STEP}", c
        assert c[1].startswith(f"roll|1|{root / 'srcp185'}|"), c
        assert f"/repo/runs/flash_chord/{RUN}/policy_{STEP}.safetensors /repo/runs/ep7/{RUN}/blowup_s{STEP}_w4096 --worlds 4096" in c[1], c[1]
        for name in ("eval", "evallog", "a2", "blowup"):
            assert (root / f"results/ep7/{name}_{RUN}_s{STEP}.txt").exists(), name
        a2 = (root / f"results/ep7/a2_{RUN}_s{STEP}.txt").read_text(errors="replace")
        assert a2.count("  满足  ") == 4 and "A2_OK" in a2, a2
        evallog = (root / f"results/ep7/evallog_{RUN}_s{STEP}.txt").read_text(errors="replace")
        assert "in-flash-chord: /workspace ← " in evallog and "evaluation step 896/896" in evallog, evallog
        assert (root / f"runs/ep7/{RUN}/blowup_s{STEP}_w4096.npz").exists()
        assert not (root / f"runs/ep7/{RUN}/blowup_s{STEP}_w4096_hydra").exists()
        assert not (root / f"runs/.chain-active-{NAME}").exists()
        assert "脚本 md5：" in log(root)


def test_a2_fail() -> None:
    wrong_sha = hashlib.sha256(b"x").hexdigest()
    for kw, nf in (({"FAKE_NF": "[3, 7]"}, "2"), ({"FAKE_STEPS": "700"}, "0"), ({"FAKE_SHA": wrong_sha}, "0"),
                   ({"FAKE_WORLDS": "256"}, "0"), ({"FAKE_NF": "missing"}, "未知")):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); setup(root)
            r = chain(root, FAKE_BAD="5", **kw)
            assert r.returncode == 0, r.stderr
            assert f"EP7_EVAL_DONE（评测 成功；评测器记的非有限世界 {nf}；A-2 不满足；取证 完成（崩掉的世界 5））" in last(root), (kw, log(root))
            failed = (root / f"results/ep7/a2_{RUN}_s{STEP}.txt.failed").read_text(errors="replace")
            assert "A2_FAIL" in failed and failed.count("  不满足  ") == 1, (kw, failed)


def test_termination_error() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d); setup(root)
        r = chain(root, FAKE_EVAL_MODE="termination", FAKE_ROLL_TRACEBACK="1", FAKE_BAD="9")
        assert r.returncode == 0, r.stderr
        assert "EP7_EVAL_DONE（评测 失败（终止报错）；评测器记的非有限世界 7；A-2 没判；取证 完成（崩掉的世界 9））" in last(root), log(root)
        assert (root / f"results/ep7/eval_{RUN}_s{STEP}.txt.failed").exists()
        blow = root / f"results/ep7/blowup_{RUN}_s{STEP}.txt"   # 输出里有 Traceback 也不判失败
        assert blow.exists() and "Traceback" in blow.read_text(errors="replace")
        evallog = (root / f"results/ep7/evallog_{RUN}_s{STEP}.txt").read_text(errors="replace")
        assert "7 worlds recorded non-finite" in evallog and "must disable task failure termination" in evallog, evallog
        assert not (root / f"results/ep7/a2_{RUN}_s{STEP}.txt").exists()


def test_other_failures() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d); setup(root)
        r = chain(root, FAKE_EVAL_MODE="broken", FAKE_ROLL_RC="5")
        assert r.returncode == 0, r.stderr
        assert "EP7_EVAL_DONE（评测 失败（其他，等于没评成，退出码 2）；评测器记的非有限世界 未知；A-2 没判；取证 失败）" in last(root), log(root)


def test_busy_then_retry() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d); setup(root)
        r = chain(root, FAKE_BUSY_FIRST="1")
        assert r.returncode == 0, r.stderr
        assert "退出码 6" in log(root) and "EP7_EVAL_DONE（评测 成功；" in last(root), log(root)
        assert sum(1 for c in calls(root) if c.startswith("eval|")) == 2
        assert (root / f"results/ep7/eval_{RUN}_s{STEP}.txt").exists()
        assert not (root / f"results/ep7/eval_{RUN}_s{STEP}.txt.failed").exists()


def test_refuse() -> None:
    def refused(prepare=None, policy=(1.0, 2.0, 3.0), **kw) -> None:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); setup(root, policy=policy)
            before = prepare(root) if prepare else None
            r = chain(root, **kw)
            assert r.returncode == 2, (r.returncode, r.stdout, r.stderr)
            assert not calls(root)
            if before is None:
                assert not (root / f"runs/logs/{NAME}.log").exists()
            else:
                assert log(root) == before

    def existing_log(root):
        (root / f"runs/logs/{NAME}.log").write_text("旧日志\n"); return "旧日志\n"

    def existing_json(root):
        (root / f"runs/flash_chord/{RUN}/eval_{STEP}_w4096.json").write_text("{}")

    def existing_npz(root):
        (root / f"runs/ep7/{RUN}").mkdir(parents=True); (root / f"runs/ep7/{RUN}/blowup_s{STEP}_w4096.npz").write_text("old")

    refused(existing_log)
    refused(policy=None)
    refused(policy=(1.0, float("nan"), 3.0))
    refused(existing_json)
    refused(existing_npz)
    with tempfile.TemporaryDirectory() as d:  # 两份副本填反：各自对不上清单
        root = Path(d); setup(root)
        r = chain(root, V2D_EVAL_SRC=str(root / "srcp185"), V2D_ROLLOUT_SRC=str(root / "src8c"))
        assert r.returncode == 2 and not calls(root), (r.returncode, r.stderr)


def test_gpu_busy() -> None:
    with tempfile.TemporaryDirectory() as d:  # 评测前就一直被占
        root = Path(d); setup(root)
        r = chain(root, FAKE_GPU="busy", V2D_GPU_WAIT_S="2")
        assert r.returncode == 1 and not calls(root), (r.returncode, r.stderr)
        assert last(root).endswith("EP7_EVAL_FAILED"), log(root)
    with tempfile.TemporaryDirectory() as d:  # 评测完之后被占：取证没跑，但评测的结论照记
        root = Path(d); setup(root)
        r = chain(root, FAKE_GPU_BUSY_AFTER_EVAL=f"{RUN}/eval_{STEP}_w4096.json", V2D_GPU_WAIT_S="2")
        assert r.returncode == 0, r.stderr
        assert "EP7_EVAL_DONE（评测 成功；评测器记的非有限世界 0；A-2 满足；取证 没跑（GPU 被占））" in last(root), log(root)
        assert len(calls(root)) == 1


def test_dryrun() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d); setup(root)
        r = chain(root, V2D_EP7_EVAL_DRYRUN="1")
        assert r.returncode == 0 and "EP7_EVAL_DRYRUN_OK" in r.stdout, (r.returncode, r.stdout, r.stderr)
        assert not calls(root) and not (root / f"runs/logs/{NAME}.log").exists()


if __name__ == "__main__":
    for t in (test_clean, test_a2_fail, test_termination_error, test_other_failures, test_busy_then_retry, test_refuse, test_gpu_busy, test_dryrun):
        t(); print(f"ok {t.__name__}")
    print("ALL_OK")
