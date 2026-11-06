"""fc_export.sh 的行为测试（暂无契约 ID：运行规范待成文）。只用标准库，Mac 与训练机都能跑。

临时目录里搭最小仓库：真的 fc_export.sh；假的 ckpt_nonfinite.py（按 FAKE_NF 打印“非有限 0 / 10”“非有限 3 / 10”或不完整那一行）、
eval_summary.py、scripts/dev/in-flash-chord（记下收到的命令，按 FAKE_EVAL 写出 JSON 与 parquet、报终止错误、报别的错误或退出码 0 却不写文件）；
PATH 前面放假的 flock 与 nvidia-smi（按 FAKE_GPU 报有没有进程）。
  1. 守卫开着、policy 有限：导出到 export_8c778f9/s<步数>_w1/episode_000006.{json,parquet}，命令与 kit 的 run_policy_evaluation.py 逐项同序；
  2. 规矩一：有 NONFINITE、policy 含非有限参数、扫描结果里没有这一步（不完整）都拒绝，退出码 12，不起容器；关掉守卫时照样导出；
  3. 规矩二的识别：评测器报 must disable task failure termination → 13；别的失败 → 3；退出码 0 却缺文件 → 3；
  4. V2D_FC_SRC 不是 8c778f9 副本 → 2；run 名档位与 run 记录里 parquet 的档位不一致 → 2；GPU 有别的进程 → 6；
  5. 已有完整结果不重复导出；V2D_EXPORT_TRAJ=0 时命令里没有 object_trajectories_output，只要 JSON；
  6. 同一个日志先有一次终止报错、再有一次别的失败：第二次判 3，不判 13（报错分类只看这一次追加的日志）；
  7. V2D_EXPORT_DRYRUN=1：守卫照常（含 NaN 的仍是 12），通过时只打印命令、退出码 0，不起容器、不写文件；
  8. 摘要脚本出错不改变导出的退出码；
  9. Tier 3 校正版（agent-doc/plan-t3-pilot-v2-2026-10-10.md）：parquet 在 runs/motion/tier3_v2/ 下时，v2d_meta.json 到 runs/motion_src/tier3_v2/
     找（tier3/ 下第一轮那份不读）；目录名大写、空后缀、档位与 run 名不符、带后缀而 run 名不含 -<名>- 都拒绝，退出码 2；
     10/16 起 Tier 1 带后缀的目录（tier1_c10，run 名含 -c10-）照常导出（test_tier1_suffix_dir）。
 10. V2D_EXPORT_SUFFIX（agent-doc/plan-first-place-2026-10-14.md 选项零）：已有不带后缀的结果时，带后缀照样重新导出到 s<步数>_w1_<名>/、
     日志名与 hydra.run.dir 也带后缀，原来那份不动；预演打印带后缀的路径、不建目录；不带后缀时行为不变；后缀不合格式（没有下划线、大写、
     斜杠、超过 16 个字符）退出码 2，不起容器。
 11. V2D_EXPORT_PARQUET（k-fix 换参考，agent-doc/plan-t3-kfix-2026-10-16.md 方案 4）：经 eval_playback_override.py（--playback same --parquet，导出参数
     经 --extra 透传）导出到带后缀的目录，不再调 scripts/evaluate_policy.py；预演打印新参考；不给后缀、序列不同、路径不合格、新参考的 episode 不同、
     parquet 不在都拒绝，退出码 2，不起容器。
 12. 评测器写出 JSON 之后退出码非零（例如包装层判作废）：退出码 3，这一次的 JSON 删掉，再跑一次照常起容器，不当成已有结果。
 13. 冒烟 run（smoke-t<档位>-…）照常导出；前缀写错、档位不是 1–3、档位与 run 记录不一致都拒绝（test_smoke_run_name）。
用法：python3 stages/evaluate/test_fc_export.py
"""
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUN, STEP, SEQ = "t1-ep6-w2048-cap4m-mf4090", "349999104", "2026-06-01_11-26-44_monitor_rotate_05"
FAKE_NF = r'''import os, sys
mode = os.environ.get("FAKE_NF", "ok")
print(f"== {sys.argv[1]}（policy 35 份）")
if mode == "ok":
    print("  349,999,104 步：非有限 0 / 10（NaN 0，inf 0）"); print("  全部有限"); sys.exit(0)
if mode == "nan":
    print("  349,999,104 步：非有限 3 / 10（NaN 3，inf 0）"); sys.exit(1)
print("  349,999,104 步：不完整或还在写，跳过"); sys.exit(0)
'''
FAKE_IFC = r'''#!/usr/bin/env bash
echo "$*" >> "$FAKE_CALLS"
echo "in-flash-chord: /workspace ← ${V2D_FC_SRC}（flash_chord 8c778f905c（fake））" >&2
case "${FAKE_EVAL:-ok}" in
  term) echo "RuntimeError: evaluation must disable task failure termination before scoring the full reference" >&2; exit 1 ;;
  boom) echo "RuntimeError: something else" >&2; exit 1 ;;
  empty) exit 0 ;;
  writefail) for tok in $*; do case "$tok" in evaluation.metrics_output=*) f="${tok#evaluation.metrics_output=}"; f="$FAKE_ROOT/${f#/repo/}"; mkdir -p "$(dirname "$f")"; echo '{"world_count": 1}' > "$f" ;; esac; done
             echo "结果作废" >&2; exit 2 ;;
esac
for tok in $*; do
  case "$tok" in
    evaluation.metrics_output=*) f="${tok#evaluation.metrics_output=}"; f="$FAKE_ROOT/${f#/repo/}"; mkdir -p "$(dirname "$f")"; echo '{"world_count": 1}' > "$f" ;;
    evaluation.object_trajectories_output=*) f="${tok#evaluation.object_trajectories_output=}"; f="$FAKE_ROOT/${f#/repo/}"; mkdir -p "$(dirname "$f")"; echo PAR1 > "$f" ;;
    /repo/*/episode_*.json) f="$FAKE_ROOT/${tok#/repo/}"; mkdir -p "$(dirname "$f")"; echo '{"world_count": 1}' > "$f" ;;   # 换参考：包装的位置参数
  esac
done
'''
FAKE_GPU = '#!/usr/bin/env bash\n[ "${FAKE_GPU:-idle}" = busy ] && echo 4242\nexit 0\n'


def setup(root: Path) -> None:
    for d in ("stages/evaluate", "stages/policy", "scripts/dev", "bin", f"runs/flash_chord/{RUN}", f"runs/motion_src/tier1/{SEQ}", "fcsrc", "badsrc"):
        (root / d).mkdir(parents=True, exist_ok=True)
    shutil.copy(HERE / "fc_export.sh", root / "stages/evaluate")
    (root / "stages/policy/ckpt_nonfinite.py").write_text(FAKE_NF)
    (root / "stages/policy/eval_summary.py").write_text("import os, sys\nprint('SUMMARY')\nsys.exit(int(os.environ.get('FAKE_SUMMARY_RC', '0')))\n")
    for name, text in (("scripts/dev/in-flash-chord", FAKE_IFC), ("bin/nvidia-smi", FAKE_GPU), ("bin/flock", "#!/usr/bin/env bash\nexit 0\n")):
        (root / name).write_text(text)
        (root / name).chmod(0o755)
    (root / "fcsrc/SOURCE_V2D.txt").write_text("flash_chord 8c778f905c（v0.4.0 33129dd + patch）\n")
    (root / "badsrc/SOURCE_V2D.txt").write_text("flash_chord v0.4.0（33129dd）+ #185\n")
    d = root / "runs/flash_chord" / RUN
    (d / f"policy_{STEP}.safetensors").write_bytes(b"x")
    parquet = f"/repo/runs/motion/tier1/whole_body/soma/sequence_id={SEQ}/robot_name=g1/data.parquet"
    (d / "run_record.jsonl").write_text(json.dumps({"event": "start", "parquet": parquet}) + "\n" + json.dumps({"event": "start", "parquet": "x"}) + "\n")
    (root / f"runs/motion_src/tier1/{SEQ}/v2d_meta.json").write_text(json.dumps({"tier": 1, "episode": 6, "sequence_id": SEQ}))


def run(root: Path, run_name: str = RUN, **env) -> tuple:
    calls = root / "calls.txt"
    if calls.exists():
        calls.unlink()
    e = dict(os.environ, PATH=f"{root / 'bin'}:{os.environ['PATH']}", FAKE_CALLS=str(calls), FAKE_ROOT=str(root),
             V2D_FC_SRC=str(root / "fcsrc"))
    e.update(env)
    r = subprocess.run(["bash", str(root / "stages/evaluate/fc_export.sh"), run_name, STEP], cwd=root, env=e,
                       capture_output=True, text=True, errors="replace", timeout=60)
    return r.returncode, r.stdout + r.stderr, calls.read_text() if calls.exists() else ""


def out(root: Path, w: str = "1") -> Path:
    return root / f"runs/flash_chord/{RUN}/export_8c778f9/s{STEP}_w{w}/episode_000006"


def test_export_matches_kit_runner() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root)
        rc, text, calls = run(root)
        assert rc == 0 and out(root).with_suffix(".json").is_file() and out(root).with_suffix(".parquet").is_file(), (rc, text)
        toks = calls.split("python scripts/evaluate_policy.py ", 1)[1].split()
        keys = [x.split("=", 1)[0] for x in toks]
        assert keys == ["evaluation.checkpoint", "evaluation.world_count", "evaluation.reset_mode", "evaluation.metrics_output",
                        "evaluation.object_trajectories_output", "hydra.run.dir", "logging.mode"], keys
        assert f"evaluation.checkpoint=/repo/runs/flash_chord/{RUN}/policy_{STEP}.safetensors" in toks and "evaluation.world_count=1" in toks
        assert "evaluation.reset_mode=explicit" in toks and "logging.mode=disabled" in toks and calls.startswith("cd /workspace && "), calls
        assert "SUMMARY" in text and "in-flash-chord: /workspace ← " in text, text
        rc2, text2, calls2 = run(root)
        assert rc2 == 0 and not calls2 and "已有结果" in text2, (rc2, text2)


def test_rule1_guard() -> None:
    for how in ("NONFINITE", "nan", "partial"):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); setup(root)
            env = {}
            if how == "NONFINITE":
                (root / f"runs/flash_chord/{RUN}/NONFINITE").write_text("x")
            else:
                env["FAKE_NF"] = how
            rc, text, calls = run(root, **env)
            assert rc == 12 and not calls and "不导出" in text, (how, rc, text, calls)
            rc, text, calls = run(root, V2D_EXPORT_UNGUARDED="1", **env)
            assert rc == 0 and calls, (how, rc, text)


def test_rule2_and_failures() -> None:
    for mode, want in (("term", 13), ("boom", 3), ("empty", 3)):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); setup(root)
            rc, text, _ = run(root, FAKE_EVAL=mode)
            assert rc == want, (mode, rc, text)
            if mode == "term":
                assert "规矩二" in text, text


def test_refusals() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root)
        rc, text, calls = run(root, V2D_FC_SRC=str(root / "badsrc"))
        assert rc == 2 and not calls and "8c778f9" in text, (rc, text)
        rc, text, calls = run(root, FAKE_GPU="busy")
        assert rc == 6 and not calls, (rc, text)
        t2 = "t2-ep6-w2048-cap4m-mf4090"
        shutil.copytree(root / f"runs/flash_chord/{RUN}", root / f"runs/flash_chord/{t2}")
        rc, text, calls = run(root, run_name=t2)
        assert rc == 2 and not calls and "档位" in text, (rc, text)
        rc, text, calls = run(root, run_name="smoke-t1-ep6-w2048-cap4m-mf4090")
        assert rc == 2 and not calls, (rc, text)


def test_traj_off() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root)
        rc, text, calls = run(root, V2D_EXPORT_WORLDS="4096", V2D_EXPORT_TRAJ="0")
        assert rc == 0 and "object_trajectories_output" not in calls and "evaluation.world_count=4096" in calls, (rc, calls)
        assert out(root, "4096").with_suffix(".json").is_file() and not out(root, "4096").with_suffix(".parquet").exists()


def test_rerun_classification() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root)
        rc1, _, _ = run(root, FAKE_EVAL="term")
        rc2, text2, _ = run(root, FAKE_EVAL="boom")
        assert (rc1, rc2) == (13, 3), (rc1, rc2, text2)


def test_dryrun() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root)
        rc, text, calls = run(root, V2D_EXPORT_DRYRUN="1", FAKE_GPU="busy")
        assert rc == 0 and not calls and "DRYRUN" in text and "evaluation.world_count=1" in text, (rc, text, calls)
        assert not (root / f"runs/flash_chord/{RUN}/export_8c778f9").exists() and not (root / "runs/logs").exists()
        rc, text, calls = run(root, V2D_EXPORT_DRYRUN="1", FAKE_NF="nan")
        assert rc == 12 and not calls, (rc, text)


def test_summary_failure_keeps_exit_code() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root)
        rc, text, _ = run(root, FAKE_SUMMARY_RC="1")
        assert rc == 0 and "摘要打印失败" in text, (rc, text)



def test_t3_set_dir() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root)
        r3 = "t3-ep24-w2048-cap4m-r3a-v2-mf4090"
        d = root / "runs/flash_chord" / r3
        shutil.copytree(root / f"runs/flash_chord/{RUN}", d)
        pq3 = f"/repo/runs/motion/tier3_v2/whole_body/soma/sequence_id={SEQ}/robot_name=g1/data.parquet"
        (d / "run_record.jsonl").write_text(json.dumps({"event": "start", "parquet": pq3}) + "\n")
        for sub, ep in (("tier3_v2", 24), ("tier3", 99)):
            (root / f"runs/motion_src/{sub}/{SEQ}").mkdir(parents=True)
            (root / f"runs/motion_src/{sub}/{SEQ}/v2d_meta.json").write_text(json.dumps({"tier": 3, "episode": ep, "sequence_id": SEQ}))
        rc, text, calls = run(root, run_name=r3)
        assert rc == 0 and (d / f"export_8c778f9/s{STEP}_w1/episode_000024.json").is_file(), (rc, text)
        assert not (d / f"export_8c778f9/s{STEP}_w1/episode_000099.json").exists(), text
        for bad in ("tier3_V2", "tier3_", "tier2_v2", "tier3_kfix"):
            (d / "run_record.jsonl").write_text(json.dumps({"event": "start", "parquet": pq3.replace("tier3_v2", bad)}) + "\n")
            rc, text, calls = run(root, run_name=r3)
            assert rc == 2 and not calls, (bad, rc, text)


def test_tier1_suffix_dir() -> None:
    """10/16 起 Tier 1 也许带后缀的参考目录（tier1_c10），run 名要含 -c10-；不含就拒绝，退出码 2。"""
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root)
        pq1 = f"/repo/runs/motion/tier1_c10/whole_body/soma/sequence_id={SEQ}/robot_name=g1/data.parquet"
        (root / f"runs/motion_src/tier1_c10/{SEQ}").mkdir(parents=True)
        (root / f"runs/motion_src/tier1_c10/{SEQ}/v2d_meta.json").write_text(json.dumps({"tier": 1, "episode": 6, "sequence_id": SEQ}))
        for name, want in (("t1-ep6-w2048-cap4m-r3a-c10-mf4090", 0), ("t1-ep6-w2048-cap4m-r3a-mf4090", 2)):
            d = root / "runs/flash_chord" / name
            shutil.copytree(root / f"runs/flash_chord/{RUN}", d)
            (d / "run_record.jsonl").write_text(json.dumps({"event": "start", "parquet": pq1}) + "\n")
            rc, text, calls = run(root, run_name=name)
            assert rc == want and (bool(calls) == (want == 0)), (name, rc, text)


def test_suffix() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root)
        rc, text, calls = run(root)
        assert rc == 0 and calls, (rc, text)
        first = out(root).with_suffix(".json").read_text()
        rc, text, calls = run(root, V2D_EXPORT_SUFFIX="_rep1")
        rep = root / f"runs/flash_chord/{RUN}/export_8c778f9/s{STEP}_w1_rep1/episode_000006"
        assert rc == 0 and calls and "已有结果" not in text, (rc, text)
        assert rep.with_suffix(".json").is_file() and rep.with_suffix(".parquet").is_file(), text
        assert f"evaluation.metrics_output=/repo/runs/flash_chord/{RUN}/export_8c778f9/s{STEP}_w1_rep1/episode_000006.json" in calls, calls
        assert (root / f"runs/logs/{RUN}.export_8c778f9_s{STEP}_w1_rep1.log").is_file(), text
        assert f"hydra.run.dir=/repo/runs/flash_chord/{RUN}/export_8c778f9/s{STEP}_w1_rep1/hydra/episode_000006" in calls, calls
        rc, text, calls = run(root, V2D_EXPORT_SUFFIX="_rep2", V2D_EXPORT_DRYRUN="1")
        assert rc == 0 and not calls and f"export_8c778f9/s{STEP}_w1_rep2/episode_000006.json" in text, (rc, text)
        assert not (root / f"runs/flash_chord/{RUN}/export_8c778f9/s{STEP}_w1_rep2").exists()
        assert out(root).with_suffix(".json").read_text() == first
        rc, text, calls = run(root, V2D_EXPORT_SUFFIX="_rep1")
        assert rc == 0 and not calls and "已有结果" in text, (rc, text)
        for bad in ("rep1", "_REP1", "_a/b", "_" + "a" * 17, "_"):
            rc, text, calls = run(root, V2D_EXPORT_SUFFIX=bad)
            assert rc == 2 and not calls and "V2D_EXPORT_SUFFIX" in text, (bad, rc, text)


def test_new_reference() -> None:
    """V2D_EXPORT_PARQUET（k-fix）：经包装导出到带后缀的目录；不给后缀、序列不同、episode 不同、parquet 不在都拒绝，退出码 2，不起容器。"""
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root)
        r3 = "t3-ep24-w2048-cap4m-r3a-v2-mf4090"
        d = root / "runs/flash_chord" / r3
        shutil.copytree(root / f"runs/flash_chord/{RUN}", d)
        old = f"/repo/runs/motion/tier3_v2/whole_body/soma/sequence_id={SEQ}/robot_name=g1/data.parquet"
        newp = f"/repo/runs/motion/tier3_kfixf/whole_body/soma/sequence_id={SEQ}/robot_name=g1/data.parquet"
        (d / "run_record.jsonl").write_text(json.dumps({"event": "start", "parquet": old}) + "\n")
        for sub in ("tier3_v2", "tier3_kfixf"):
            (root / f"runs/motion_src/{sub}/{SEQ}").mkdir(parents=True)
            (root / f"runs/motion_src/{sub}/{SEQ}/v2d_meta.json").write_text(json.dumps({"tier": 3, "episode": 24, "sequence_id": SEQ}))
        np_ = root / newp[len("/repo/"):]
        np_.parent.mkdir(parents=True)
        np_.write_bytes(b"PAR1")
        rc, text, calls = run(root, run_name=r3, V2D_EXPORT_PARQUET=newp, V2D_EXPORT_SUFFIX="_kfixf")
        outj = d / f"export_8c778f9/s{STEP}_w1_kfixf/episode_000024.json"
        assert rc == 0 and outj.is_file() and outj.with_suffix(".parquet").is_file(), (rc, text)
        for tok in ("/repo/stages/evaluate/eval_playback_override.py", "--playback same", f"--parquet {newp}",
                    "--extra evaluation.reset_mode=explicit", "--extra logging.mode=disabled",
                    f"--extra evaluation.object_trajectories_output=/repo/runs/flash_chord/{r3}/export_8c778f9/s{STEP}_w1_kfixf/episode_000024.parquet"):
            assert tok in calls, (tok, calls)
        assert "scripts/evaluate_policy.py" not in calls, calls
        rc, text, calls = run(root, run_name=r3, V2D_EXPORT_PARQUET=newp, V2D_EXPORT_SUFFIX="_kfixg", V2D_EXPORT_DRYRUN="1")
        assert rc == 0 and not calls and "新参考" in text and "eval_playback_override.py" in text, (rc, text)
        bad_cases = [("不给后缀", dict(V2D_EXPORT_PARQUET=newp)),
                     ("序列不同", dict(V2D_EXPORT_PARQUET=newp.replace(SEQ, SEQ + "x"), V2D_EXPORT_SUFFIX="_k1")),
                     ("路径不合格", dict(V2D_EXPORT_PARQUET="/tmp/data.parquet", V2D_EXPORT_SUFFIX="_k2"))]
        for name, env in bad_cases:
            rc, text, calls = run(root, run_name=r3, **env)
            assert rc == 2 and not calls, (name, rc, text)
        (root / f"runs/motion_src/tier3_kfixf/{SEQ}/v2d_meta.json").write_text(json.dumps({"tier": 3, "episode": 25, "sequence_id": SEQ}))
        rc, text, calls = run(root, run_name=r3, V2D_EXPORT_PARQUET=newp, V2D_EXPORT_SUFFIX="_k3")
        assert rc == 2 and not calls and "episode" in text, ("episode 不同", rc, text)
        (root / f"runs/motion_src/tier3_kfixf/{SEQ}/v2d_meta.json").write_text(json.dumps({"tier": 3, "episode": 24, "sequence_id": SEQ}))
        np_.unlink()
        rc, text, calls = run(root, run_name=r3, V2D_EXPORT_PARQUET=newp, V2D_EXPORT_SUFFIX="_k4")
        assert rc == 2 and not calls and "不在" in text, ("parquet 不在", rc, text)

def test_smoke_run_name() -> None:
    """13. 冒烟 run（smoke-t<档位>-…，agent-doc/plan-t2st-2026-10-19.md 复核第 1 条）照常导出，档位取 smoke- 后面那一位；
    前缀写错、档位不是 1–3、档位与 run 记录里的 parquet 不一致都拒绝，退出码 2，不起容器。"""
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root)
        for name, want in (("smoke-t1-ep6-w2048-cap4m-mf4090", 0), ("smokey-t1-ep6-w2048-cap4m-mf4090", 2),
                           ("smoke-t4-ep6-w2048-cap4m-mf4090", 2), ("smoke-t2-ep6-w2048-cap4m-mf4090", 2)):
            shutil.copytree(root / f"runs/flash_chord/{RUN}", root / "runs/flash_chord" / name)
            rc, text, calls = run(root, run_name=name)
            assert rc == want and (bool(calls) == (want == 0)), (name, rc, text)
            if want == 0:
                assert (root / f"runs/flash_chord/{name}/export_8c778f9/s{STEP}_w1/episode_000006.json").is_file(), text


def test_failed_run_leaves_no_result() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root)
        rc, text, calls = run(root, FAKE_EVAL="writefail")
        assert rc == 3 and calls and not out(root).with_suffix(".json").exists(), (rc, text)
        rc2, text2, calls2 = run(root)
        assert rc2 == 0 and calls2 and "已有结果" not in text2, (rc2, text2)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("ALL_OK")
