"""make_clone_queue.py（计划 agent-doc/plan-t1r3o-2026-10-31.md 第 3 版）：新行与来源行只差 run 名、标签与两个物体朝向变量；
来源不是变体 A、已带朝向变量、在队列里有两种不同的行、新 run 已存在时都拒绝。只用标准库，python3 直接跑。"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

GEN = Path(__file__).resolve().parent / "make_clone_queue.py"
HEAD = "run\tsmoke\tsequence_id\tplayback\tsrc_frames\tsupport\tpos_tol\tori_tol\tcapacity\tseed\txeval"
VAR = "objective.object_position_std=0.05 termination.object_orientation_threshold=0.6"


def line(run, tag="r3", extra=VAR):
    return f"V2D_WANDB_TAG={tag} V2D_MAX_RETRY=0 V2D_KEEP_STATES=1 {run} /repo/x.parquet 2048 349999104 training.replay.capacity=4000000 reset.seed=43 {extra}"


def setup(root, formal, smoke=None):
    q = root / "stages" / "policy" / "queues"
    q.mkdir(parents=True)
    (root / "stages" / "evaluate" / "exports").mkdir(parents=True)
    run = "t1-ep7-w2048-cap4m-r3a-s43-mf4090"
    (q / "mf4090_old.txt").write_text("# h\n" + "\n".join(formal) + "\n", encoding="utf-8")
    (q / "mf4090_old_smoke.txt").write_text("# h\n" + (smoke or line(f"smoke-{run}", "smoke").replace("349999104", "4096000")) + "\n", encoding="utf-8")
    (q / "mf4090_old.expect.tsv").write_text(f"{HEAD}\n{run}\tsmoke-{run}\tseq\tframe\t896\ttrue\t0.015\t0.086\t4000000\t43\tno\n", encoding="utf-8")
    return run


def gen(root, *args):
    return subprocess.run([sys.executable, str(GEN), *args], cwd=root, capture_output=True, text=True)


def test_clone_only_three_changes():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        run = setup(root, [line("t1-ep7-w2048-cap4m-r3a-s43-mf4090")])
        r = gen(root, "t1zq9", "mf4090", run)
        assert r.returncode == 0, r.stdout + r.stderr
        q = root / "stages" / "policy" / "queues"
        new = [x for x in (q / "mf4090_t1zq9.txt").read_text(encoding="utf-8").splitlines() if not x.startswith("#")]
        assert new == [line("V2D_OBJ_ORI_WEIGHT=1.0 V2D_OBJ_ORI_STD=0.4 t1-ep7-w2048-cap4m-r3o-s43-mf4090", "t1zq9")], new
        sm = [x for x in (q / "mf4090_t1zq9_smoke.txt").read_text(encoding="utf-8").splitlines() if not x.startswith("#")]
        assert sm and "V2D_WANDB_TAG=smoke" in sm[0] and " V2D_OBJ_ORI_WEIGHT=1.0 V2D_OBJ_ORI_STD=0.4 smoke-t1-ep7-w2048-cap4m-r3o-s43-mf4090 " in sm[0], sm
        ex = (q / "mf4090_t1zq9.expect.tsv").read_text(encoding="utf-8").splitlines()
        assert ex[0] == HEAD and ex[1].split("\t")[:2] == ["t1-ep7-w2048-cap4m-r3o-s43-mf4090", "smoke-t1-ep7-w2048-cap4m-r3o-s43-mf4090"], ex
        assert ex[1].split("\t")[2:] == ["seq", "frame", "896", "true", "0.015", "0.086", "4000000", "43", "no"], ex
        cand = (root / "stages" / "evaluate" / "exports" / "cand_t1_t1zq9_mf4090.tsv").read_text(encoding="utf-8").splitlines()
        assert cand[1:] == ["1\t7\tt1-ep7-w2048-cap4m-r3o-s43-mf4090\t349999104"], cand


def test_refusals():
    run = "t1-ep7-w2048-cap4m-r3a-s43-mf4090"
    cases = {"not_variant_a": [line(run, extra="")],
             "has_ori": [line(f"V2D_OBJ_ORI_WEIGHT=1.0 {run}")],
             "two_kinds": [line(run), line(run, extra=VAR + " scene.support=false")]}
    for name, formal in cases.items():
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            setup(root, formal)
            r = gen(root, "t1zq9", "mf4090", run)
            assert r.returncode != 0, (name, r.stdout)
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        setup(root, [line(run), line("t1-ep7-w2048-cap4m-r3o-s43-mf4090")])
        assert gen(root, "t1zq9", "mf4090", run).returncode != 0, "dst exists"
        assert gen(root, "t1zq9", "mf4090", "t1-ep7-w2048-cap4m-s43-mf4090").returncode != 0, "not -r3a-"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("ALL_OK")
