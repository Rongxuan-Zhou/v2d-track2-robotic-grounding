"""precheck_expect.sh 的档位目录核对（守计划 agent-doc/plan-t3-pilot-v2-2026-10-10.md 第 3 节：Tier 3 校正版的参考在 runs/motion/tier3_<名>/ 下，
两版参考不许对错 run；暂无契约 ID）。只用标准库。脚本要 bash 4 以上（mapfile），在训练机上跑。
临时目录里放真的 precheck_expect.sh；in-flash-chord 是桩：把收到的命令记进 calls.txt，打印 ASSERT_OK、退出码 0。每种情形一张期望表、一条 run：
  1. t3 第一轮的写法（tier3/）：PRECHECK_OK，断言里的 parquet 是 /repo/runs/motion/tier3/…
  2. t3 校正版（tier3_v2/，run 名含 -v2-）：PRECHECK_OK，断言里的 parquet 是 /repo/runs/motion/tier3_v2/…
  3. tier3_v2/ 但 run 名不含 -v2-、t2 的 run 指到 tier1/ 或 tier1_c10/、t1 的 run 指到 tier1_c10/ 而 run 名不含 -c10-、t3 的 run 指到 tier1/、队列行的序列与期望表不同、parquet 不在 runs/motion 下：
     PRECHECK_FAIL，这条 run 不调 in-flash-chord；
  4. t1 的 run 名含 -c10-、指到 tier1_c10/（10/16 起 Tier 1 也许带后缀的目录）：PRECHECK_OK。
用法：python3 stages/policy/test_precheck_expect.py
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
HEADER = "run\tsmoke\tsequence_id\tplayback\tsrc_frames\tsupport\tpos_tol\tori_tol\tcapacity\tseed\txeval"
SEQ = "2026-06-05_15-08-08_skinny_wood_chair_push_10"
FAKE_IFC = "#!/usr/bin/env bash\necho \"$*\" >> calls.txt\necho ASSERT_OK\n"


def pq(d, seq=SEQ):
    return f"/repo/runs/motion/{d}/whole_body/soma/sequence_id={seq}/robot_name=g1/data.parquet"


def check(run, parquet):
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        for d in ("stages/policy", "scripts/dev", "runs/cfgcheck"):
            (root / d).mkdir(parents=True)
        shutil.copy(HERE / "precheck_expect.sh", root / "stages/policy")
        (root / "scripts/dev/in-flash-chord").write_text(FAKE_IFC)
        (root / "scripts/dev/in-flash-chord").chmod(0o755)
        for r in (run, f"smoke-{run}"):
            (root / f"runs/cfgcheck/{r}.yaml").write_text("x: 1\n")
        (root / "e.tsv").write_text(HEADER + "\n" + "\t".join([run, f"smoke-{run}", SEQ, "frame", "708", "false", "0.015", "0.086",
                                                                "4000000", "42", "no"]) + "\n")
        (root / "s.txt").write_text(f"V2D_X=1 smoke-{run} {parquet} 2048 4096000 training.replay.capacity=4000000\n")
        (root / "q.txt").write_text(f"V2D_X=1 {run} {parquet} 2048 349999104 training.replay.capacity=4000000\n")
        r = subprocess.run(["bash", "stages/policy/precheck_expect.sh", "e.tsv", "s.txt", "q.txt"], cwd=root, capture_output=True, text=True,
                           timeout=60)
        calls = (root / "calls.txt").read_text() if (root / "calls.txt").exists() else ""
        return r.returncode, r.stdout + r.stderr, calls


def test_first_round_dir():
    rc, out, calls = check("t3-ep8-w2048-cap4m-r3a-mf4090", pq("tier3"))
    assert rc == 0 and out.rstrip().endswith("PRECHECK_OK（1 条 run）"), out
    assert calls.count(f"task.parquet~={pq('tier3')}") == 2, calls


def test_v2_dir():
    rc, out, calls = check("t3-ep8-w2048-cap4m-r3a-v2-mf4090", pq("tier3_v2"))
    assert rc == 0 and out.rstrip().endswith("PRECHECK_OK（1 条 run）"), out
    assert calls.count(f"task.parquet~={pq('tier3_v2')}") == 2 and "/motion/tier3/" not in calls, calls


def test_mismatch_refuses():
    for run, parquet in (("t3-ep8-w2048-cap4m-r3a-mf4090", pq("tier3_v2")),
                         ("t2-ep8-w2048-cap10m-r3a-ml40", pq("tier1")),
                         ("t3-ep8-w2048-cap4m-r3a-mf4090", pq("tier1")),
                         ("t2-ep8-w2048-cap10m-r3a-c10-ml40", pq("tier1_c10")),
                         ("t1-ep8-w2048-cap10m-r3a-ml40", pq("tier1_c10")),
                         ("t3-ep8-w2048-cap4m-r3a-v2-mf4090", pq("tier3_v2", "2026-05-29_15-43-20_watering_can_pour_right_02")),
                         ("t3-ep8-w2048-cap4m-r3a-v2-mf4090", f"/repo/runs/other/tier3_v2/sequence_id={SEQ}/robot_name=g1/data.parquet")):
        rc, out, calls = check(run, parquet)
        assert rc == 1 and "PRECHECK_FAIL" in out and not calls, (run, parquet, out, calls)


def test_tier1_suffix_dir():
    rc, out, calls = check("t1-ep8-w2048-cap10m-r3a-c10-ml40", pq("tier1_c10"))
    assert rc == 0 and out.rstrip().endswith("PRECHECK_OK（1 条 run）"), out
    assert calls.count(f"task.parquet~={pq('tier1_c10')}") == 2, calls

if __name__ == "__main__":
    ver = subprocess.run(["bash", "-c", "echo ${BASH_VERSINFO[0]}"], capture_output=True, text=True).stdout.strip()
    if not ver.isdigit() or int(ver) < 4:
        raise SystemExit(f"test_precheck_expect: 要 bash 4 以上，这里是 {ver}；在训练机上跑")
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("ALL_OK")
