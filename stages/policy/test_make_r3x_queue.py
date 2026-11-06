"""make_r3x_queue.py 的测试（守计划 agent-doc/plan-t1-r3x-2026-10-08.md 第 3 节的 run 设计；运行规范未成文，暂无契约 ID）。
只用标准库。临时目录里放假的 ml40 基线期望表与队列（Tier 1 的 ep0、ep5、ep15，分在两份文件里，都带 scene.support=false），跑生成脚本后核：
  1. 6 条 run 的名字与顺序：先 seed 42 的 ep0、ep15、ep5，再 seed 43 的同样三条
  2. 每行都有 buffer 400 万与变体 A 两项、基线的 scene.support=false；种子覆盖只出现在 s43 行；冒烟与正式行除步数、前缀外一致；
     期望表的 buffer 列 4000000、种子列 42 / 43，其余列照抄基线
  3. 导出清单 6 行与 run 一一对应
  4. 基线行找不到、或基线已经覆盖了 buffer：报错退出，不写文件
用法：python3 stages/policy/test_make_r3x_queue.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEQ = {0: ("2026-06-04_stool", "778"), 5: ("2026-03-18_box", "1003"), 15: ("2026-05-06_sign", "715")}
ENV = "V2D_WANDB_TAG=batch V2D_MAX_RETRY=0 V2D_GPU_APPS=1 V2D_WANDB_UPLOAD_CKPT=0"
SENV = "V2D_WANDB_PROJECT=v2d-selftest V2D_WANDB_TAG=smoke V2D_MAX_RETRY=0 V2D_GPU_APPS=1 V2D_WANDB_UPLOAD_CKPT=0"
HEADER = "run\tsmoke\tsequence_id\tplayback\tsrc_frames\tsupport\tpos_tol\tori_tol\tcapacity\tseed\txeval"
VAR = ["objective.object_position_std=0.05", "termination.object_orientation_threshold=0.6"]


def pq(ep):
    return f"/repo/runs/motion/tier1/whole_body/soma/sequence_id={SEQ[ep][0]}/robot_name=g1/data.parquet"


def setup(root: Path, drop=None, cap_override=False):
    q = root / "stages" / "policy" / "queues"
    q.mkdir(parents=True)
    for part, eps in (("batch1a", [0, 5]), ("batch1b", [15])):
        exp, fl, sl = [HEADER], ["# 基线"], ["# 基线冒烟"]
        for ep in eps:
            if ep == drop:
                continue
            run = f"t1-ep{ep}-w2048-cap10m-ml40"
            seq, frames = SEQ[ep]
            exp.append("\t".join([run, f"smoke-{run}", seq, "frame", frames, "false", "0.015", "0.086", "10000000", "42", "no"]))
            over = " scene.support=false" + (" training.replay.capacity=4000000" if cap_override and ep == 5 else "")
            fl.append(f"{ENV} {run} {pq(ep)} 2048 349999104{over}")
            sl.append(f"{SENV} smoke-{run} {pq(ep)} 2048 4096000{over}")
        (q / f"ml40_{part}.expect.tsv").write_text("\n".join(exp) + "\n")
        (q / f"ml40_{part}.txt").write_text("\n".join(fl) + "\n")
        (q / f"ml40_{part}_smoke.txt").write_text("\n".join(sl) + "\n")


def gen(root: Path):
    return subprocess.run([sys.executable, str(HERE / "make_r3x_queue.py"), "--root", str(root)], capture_output=True, text=True)


def lines(p: Path):
    return [x for x in p.read_text().splitlines() if x and not x.startswith("#")]


def test_generates_runs():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        setup(root)
        r = gen(root)
        assert r.returncode == 0, r.stdout + r.stderr
        q = root / "stages" / "policy" / "queues"
        want = [f"t1-ep{e}-w2048-cap4m-r3a-mf4090" for e in (0, 15, 5)] + [f"t1-ep{e}-w2048-cap4m-r3a-s43-mf4090" for e in (0, 15, 5)]
        f, s = lines(q / "mf4090_r3x.txt"), lines(q / "mf4090_r3x_smoke.txt")
        assert [x.split()[5] for x in f] == want, f
        for fx, sx in zip(f, s):
            ft, st = fx.split(), sx.split()
            run, over = ft[5], ft[9:]
            assert ft[:5] == "V2D_WANDB_TAG=r3x V2D_MAX_RETRY=0 V2D_GPU_APPS=1 V2D_WANDB_UPLOAD_CKPT=0 V2D_KEEP_STATES=1".split(), fx
            assert st[:5] == SENV.split(), sx
            assert ft[6] == pq(int(run.split("-")[1][2:])) and ft[7] == "2048" and ft[8] == "349999104", fx
            assert over[0] == "training.replay.capacity=4000000" and "scene.support=false" in over and all(v in over for v in VAR), fx
            assert ("reset.seed=43" in over and "training.seed=43" in over) == ("-s43-" in run), fx
            assert st[5] == "smoke-" + run and st[8] == "4096000" and st[6:8] == ft[6:8] and st[9:] == over, (fx, sx)
        exp = lines(q / "mf4090_r3x.expect.tsv")
        assert exp[0] == HEADER
        for row, run in zip(exp[1:], want):
            c = row.split("\t")
            ep = int(run.split("-")[1][2:])
            assert c[:2] == [run, "smoke-" + run] and c[2:8] == [SEQ[ep][0], "frame", SEQ[ep][1], "false", "0.015", "0.086"], row
            assert c[8:] == ["4000000", "43" if "-s43-" in run else "42", "no"], row
        cand = lines(root / "stages" / "evaluate" / "exports" / "cand_t1_r3x_mf4090.tsv")
        assert cand == [f"1\t{run.split('-')[1][2:]}\t{run}\t349999104" for run in want], cand


def test_missing_baseline_refuses():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        setup(root, drop=15)
        r = gen(root)
        assert r.returncode != 0 and "t1-ep15-w2048-cap10m-ml40 的期望表行 找到 0 种" in r.stdout + r.stderr, r.stdout + r.stderr
        assert not (root / "stages" / "policy" / "queues" / "mf4090_r3x.txt").exists()


def test_capacity_override_refuses():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        setup(root, cap_override=True)
        r = gen(root)
        assert r.returncode != 0 and "buffer 不是默认的 1000 万" in r.stdout + r.stderr, r.stdout + r.stderr
        assert not (root / "stages" / "evaluate" / "exports" / "cand_t1_r3x_mf4090.tsv").exists()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("ALL_OK")
