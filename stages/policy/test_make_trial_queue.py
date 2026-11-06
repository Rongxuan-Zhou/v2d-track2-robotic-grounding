"""make_trial_queue.py 的测试（守计划 agent-doc/plan-step3-recipe-trial-2026-10-04.md 第 2 版第 3 节的试验设计；运行规范未成文，暂无契约 ID）。
只用标准库。临时目录里放假的基线期望表与队列（ml40、Tier 2，分在两份文件里，其中 ep20 没有 scene.support 覆盖），跑生成脚本后核：
  1. A 段 6 条：试验 episode 的变体 A seed 42、最后一条是对照 ep22；BC 段 10 条按 episode 交错（基线 seed 43、变体 A seed 43）
  2. 变体覆盖只出现在 r3a 行、种子覆盖只出现在 s43 行；基线原有覆盖照抄；冒烟与正式行除步数、前缀外一致；期望表的种子列 42 / 43
  3. 试验清单与名字对得上
  4. 基线行找不到、或冒烟与正式行的覆盖不一致：报错退出，不写文件
用法：python3 stages/policy/test_make_trial_queue.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEQ = {8: ("2026-06-05_chair", "708", "false"), 12: ("2026-04-25_platform", "700", "false"), 20: ("2026-06-08_cooker", "553", "true"),
       22: ("2026-05-22_ladder", "719", "false"), 24: ("2026-05-29_can", "748", "false"), 26: ("2026-06-05_board", "669", "false")}
ENV = "V2D_WANDB_TAG=t2batch V2D_MAX_RETRY=0 V2D_GPU_APPS=1 V2D_WANDB_UPLOAD_CKPT=0 V2D_KEEP_STATES=1"
SENV = "V2D_WANDB_PROJECT=v2d-selftest V2D_WANDB_TAG=smoke V2D_MAX_RETRY=0 V2D_GPU_APPS=1 V2D_WANDB_UPLOAD_CKPT=0"
HEADER = "run\tsmoke\tsequence_id\tplayback\tsrc_frames\tsupport\tpos_tol\tori_tol\tcapacity\tseed\txeval"


def pq(ep):
    return f"/repo/runs/motion/tier2/whole_body/soma/sequence_id={SEQ[ep][0]}/robot_name=g1/data.parquet"


def setup(root: Path, smoke_override_bug=False, drop=None):
    q = root / "stages" / "policy" / "queues"
    q.mkdir(parents=True)
    for part, eps in (("t2a", [8, 12, 20]), ("t2b", [22, 24, 26])):
        exp, fl, sl = [HEADER], ["# 基线"], ["# 基线冒烟"]
        for ep in eps:
            if ep == drop:
                continue
            run = f"t2-ep{ep}-w2048-cap10m-ml40"
            seq, frames, sup = SEQ[ep]
            exp.append("\t".join([run, f"smoke-{run}", seq, "frame", frames, sup, "0.015", "0.086", "10000000", "42", "no"]))
            over = "" if sup == "true" else " scene.support=false"
            fl.append(f"{ENV} {run} {pq(ep)} 2048 349999104{over}")
            sover = " scene.support=true" if (smoke_override_bug and ep == 24) else over
            sl.append(f"{SENV} smoke-{run} {pq(ep)} 2048 4096000{sover}")
        (q / f"ml40_{part}.expect.tsv").write_text("\n".join(exp) + "\n")
        (q / f"ml40_{part}.txt").write_text("\n".join(fl) + "\n")
        (q / f"ml40_{part}_smoke.txt").write_text("\n".join(sl) + "\n")


def gen(root: Path):
    return subprocess.run([sys.executable, str(HERE / "make_trial_queue.py"), "--tier", "2", "--machine", "ml40", "--root", str(root)],
                          capture_output=True, text=True)


def lines(p: Path):
    return [x for x in p.read_text().splitlines() if x and not x.startswith("#")]


def test_generates_segments():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        setup(root)
        r = gen(root)
        assert r.returncode == 0, r.stdout + r.stderr
        q = root / "stages" / "policy" / "queues"
        a = lines(q / "ml40_r3t2a.txt")
        assert [x.split()[5] for x in a] == [f"t2-ep{e}-w2048-cap10m-r3a-ml40" for e in (8, 12, 20, 24, 26, 22)], a
        bc = lines(q / "ml40_r3t2bc.txt")
        assert [x.split()[5] for x in bc] == [f"t2-ep{e}-w2048-cap10m-{k}-ml40" for e in (8, 12, 20, 24, 26) for k in ("s43", "r3a-s43")], bc
        for x in a + bc:
            tok = x.split()
            run, over = tok[5], tok[9:]
            assert tok[:5] == "V2D_WANDB_TAG=r3 V2D_MAX_RETRY=0 V2D_GPU_APPS=1 V2D_WANDB_UPLOAD_CKPT=0 V2D_KEEP_STATES=1".split(), x
            assert tok[8] == "349999104", x
            assert (("objective.object_position_std=0.05" in over and "termination.object_orientation_threshold=0.6" in over) == ("-r3a-" in run)), x
            assert (("reset.seed=43" in over and "training.seed=43" in over) == ("s43" in run)), x
            assert ("scene.support=false" in over) == ("-ep20-" not in run), x
        for stem in ("ml40_r3t2a", "ml40_r3t2bc"):
            f, s = lines(q / f"{stem}.txt"), lines(q / f"{stem}_smoke.txt")
            for fx, sx in zip(f, s):
                ft, st = fx.split(), sx.split()
                assert st[5] == "smoke-" + ft[5] and st[8] == "4096000" and st[6:8] == ft[6:8] and st[9:] == ft[9:], (fx, sx)
            exp = lines(q / f"{stem}.expect.tsv")
            assert exp[0] == HEADER
            for row, fx in zip(exp[1:], f):
                c = row.split("\t")
                assert c[0] == fx.split()[5] and c[1] == "smoke-" + c[0] and c[9] == ("43" if "s43" in c[0] else "42") and c[10] == "no", row
        spec = json.loads((root / "stages" / "evaluate" / "trials" / "r3_t2_ml40.json").read_text())
        assert [e["ep"] for e in spec["episodes"]] == [8, 12, 20, 24, 26] and spec["canary"]["ep"] == 22
        assert spec["episodes"][2] == {"ep": 20, "base42": "t2-ep20-w2048-cap10m-ml40", "var42": "t2-ep20-w2048-cap10m-r3a-ml40",
                                       "base43": "t2-ep20-w2048-cap10m-s43-ml40", "var43": "t2-ep20-w2048-cap10m-r3a-s43-ml40"}, spec
        assert spec["steps"] == 349999104 and spec["worlds"] == 4096 and spec["tier"] == 2


def test_missing_baseline_refuses():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        setup(root, drop=24)
        r = gen(root)
        assert r.returncode != 0 and "t2-ep24-w2048-cap10m-ml40 的期望表行 找到 0 种" in r.stdout + r.stderr, r.stdout + r.stderr
        assert not (root / "stages" / "policy" / "queues" / "ml40_r3t2a.txt").exists()


def test_inconsistent_smoke_refuses():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        setup(root, smoke_override_bug=True)
        r = gen(root)
        assert r.returncode != 0 and "冒烟与正式队列行 parquet 或覆盖不一致" in r.stdout + r.stderr, r.stdout + r.stderr


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("ALL_OK")
