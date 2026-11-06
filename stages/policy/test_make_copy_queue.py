"""make_copy_queue.py 的测试（守计划 agent-doc/plan-t3dyn-pilot-2026-10-14.md 第 3、4 节：试点的 run 照抄 best_hm 那批的队列行，只换 run 名、
参考目录与 W&B 标签；运行规范未成文，暂无契约 ID）。只用标准库。
临时目录里放假的 best_hm 队列（t3p4a 有 ep15、ep3，t3p4b 有 ep12、ep16，行的写法同真队列），跑生成脚本后核：
  1. t3pdyn：3 条 run，顺序 ep15、ep3、ep16，名字 t3-ep<N>-w2048-cap4m-r3a-dyn4-mf4090；正式行与源行只差 W&B 标签、run 名、parquet 目录三处，
     冒烟行只差 run 名与 parquet 目录，期望表只差前两列；导出清单档位 3、步数 349999104
  2. t3qdyn：ep12 两条，dyn4 与 dyn4s 各指到 tier3_dyn4、tier3_dyn4s，名字以 -ml40 结尾；400 万 buffer 覆盖与期望表 4000000 照留
  3. --check：生成后 COPY_CHECK_OK；手改一个文件后 COPY_CHECK_FAIL、退出码 1
  4. 源 run 缺、在两份队列里各有一行、parquet 不在 tier3_hm 下、正式步数不对、新 run 名已在别的队列里：都报错，什么文件都不写
  5. t3qdyn_s43：两条 run 名 t3-ep12-w2048-cap4m-r3a-{dyn4,dyn4s}-s43-ml40；正式行、冒烟行在变体 A 两项之前多出 reset.seed=43 training.seed=43、
     其余同 t3qdyn；期望表种子列 43、其余同 t3qdyn；源行已有种子覆盖、变体 A 的位置项不是恰好一个、源期望表种子列不是 42 时报错、
     什么文件都不写；候选导出清单抬头写“本段改种子”
用法：python3 stages/policy/test_make_copy_queue.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
HEADER = "run\tsmoke\tsequence_id\tplayback\tsrc_frames\tsupport\tpos_tol\tori_tol\tcapacity\tseed\txeval"
VAR = "objective.object_position_std=0.05 termination.object_orientation_threshold=0.6"
EPS = {15: ("seq15_sign", "false"), 3: ("seq3_mug", "true"), 12: ("seq12_platform", "false"), 16: ("seq16_bottle", "true")}


def lines(ep, steps="349999104", motion="tier3_hm", seedtok="", seedcol="42"):
    seq, sup = EPS[ep]
    run = f"t3-ep{ep}-w2048-cap4m-r3a-hm-mf4090"
    pq = f"/repo/runs/motion/{motion}/whole_body/soma/sequence_id={seq}/robot_name=g1/data.parquet"
    over = "training.replay.capacity=4000000 " + ("scene.support=false " if sup == "false" else "") + seedtok + VAR
    formal = f"V2D_WANDB_TAG=t3p4 V2D_MAX_RETRY=0 V2D_GPU_APPS=1 V2D_WANDB_UPLOAD_CKPT=0 V2D_KEEP_STATES=1 {run} {pq} 2048 {steps} {over}"
    smoke = (f"V2D_WANDB_PROJECT=v2d-selftest V2D_WANDB_TAG=smoke V2D_MAX_RETRY=0 V2D_GPU_APPS=1 V2D_WANDB_UPLOAD_CKPT=0 smoke-{run} {pq} 2048 4096000 {over}")
    exp = "\t".join([run, f"smoke-{run}", seq, "frame", str(500 + ep), sup, "0.015", "0.086", "4000000", seedcol, "no"])
    return formal, smoke, exp


def setup(t: Path, a_eps=(15, 3), b_eps=(12, 16), tweak=None):
    q = t / "stages" / "policy" / "queues"
    q.mkdir(parents=True)
    for seg, eps in (("t3p4a", a_eps), ("t3p4b", b_eps)):
        rows = [lines(ep, **(tweak or {}).get(ep, {})) for ep in eps]
        head = "# Tier 3 最终交付版 best_hm（假的）。make_var_queue.py 生成，不要手改。"
        (q / f"mf4090_{seg}.txt").write_text("\n".join([head, *(r[0] for r in rows)]) + "\n")
        (q / f"mf4090_{seg}_smoke.txt").write_text("\n".join([head, *(r[1] for r in rows)]) + "\n")
        (q / f"mf4090_{seg}.expect.tsv").write_text("\n".join([HEADER, *(r[2] for r in rows)]) + "\n")
    return q


def gen(t: Path, part: str, *opts):
    return subprocess.run([sys.executable, str(HERE / "make_copy_queue.py"), part, "--root", str(t), *opts], capture_output=True, text=True)


def body(p: Path):
    return [x for x in p.read_text().splitlines() if x and not x.startswith("#")]


def test_t3pdyn():
    with tempfile.TemporaryDirectory() as d:
        t = Path(d)
        q = setup(t)
        r = gen(t, "t3pdyn")
        assert r.returncode == 0, r.stdout + r.stderr
        runs = [f"t3-ep{ep}-w2048-cap4m-r3a-dyn4-mf4090" for ep in (15, 3, 16)]
        fo, sm, ex = body(q / "mf4090_t3pdyn.txt"), body(q / "mf4090_t3pdyn_smoke.txt"), body(q / "mf4090_t3pdyn.expect.tsv")
        assert ex[0] == HEADER and [x.split("\t")[0] for x in ex[1:]] == runs, ex
        for i, ep in enumerate((15, 3, 16)):
            f0, s0, e0 = lines(ep)
            src = f"t3-ep{ep}-w2048-cap4m-r3a-hm-mf4090"
            want_f = f0.replace("V2D_WANDB_TAG=t3p4", "V2D_WANDB_TAG=t3dyn").replace(src, runs[i]).replace("/tier3_hm/", "/tier3_dyn4/")
            assert fo[i] == want_f, (fo[i], want_f)
            assert sm[i] == s0.replace(src, runs[i]).replace("/tier3_hm/", "/tier3_dyn4/"), sm[i]
            assert ex[i + 1] == e0.replace(src, runs[i]), ex[i + 1]
        cand = body(t / "stages" / "evaluate" / "exports" / "cand_t3_t3pdyn_mf4090.tsv")
        assert cand == [f"3\t{ep}\t{r}\t349999104" for ep, r in zip((15, 3, 16), runs)], cand


def test_t3qdyn():
    with tempfile.TemporaryDirectory() as d:
        t = Path(d)
        q = setup(t)
        r = gen(t, "t3qdyn")
        assert r.returncode == 0, r.stdout + r.stderr
        fo, ex = body(q / "ml40_t3qdyn.txt"), body(q / "ml40_t3qdyn.expect.tsv")
        assert [x.split()[5] for x in fo] == ["t3-ep12-w2048-cap4m-r3a-dyn4-ml40", "t3-ep12-w2048-cap4m-r3a-dyn4s-ml40"], fo
        assert "/tier3_dyn4/" in fo[0].split()[6] and "/tier3_dyn4s/" in fo[1].split()[6], fo
        assert all("training.replay.capacity=4000000" in x.split() and "scene.support=false" in x.split() for x in fo), fo
        assert [x.split("\t")[8] for x in ex[1:]] == ["4000000", "4000000"], ex
        cand = body(t / "stages" / "evaluate" / "exports" / "cand_t3_t3qdyn_ml40.tsv")
        assert [x.split("\t")[:2] for x in cand] == [["3", "12"], ["3", "12"]], cand


def test_check():
    with tempfile.TemporaryDirectory() as d:
        t = Path(d)
        q = setup(t)
        assert gen(t, "t3pdyn").returncode == 0
        r = gen(t, "t3pdyn", "--check")
        assert r.returncode == 0 and r.stdout.strip().endswith("COPY_CHECK_OK"), r.stdout + r.stderr
        assert "V2D_WANDB_TAG=t3p4 → V2D_WANDB_TAG=t3dyn" in r.stdout, r.stdout
        f = q / "mf4090_t3pdyn.txt"
        f.write_text(f.read_text().replace("2048 349999104", "2048 349999105", 1))
        r = gen(t, "t3pdyn", "--check")
        assert r.returncode == 1 and r.stdout.strip().endswith("COPY_CHECK_FAIL"), r.stdout + r.stderr


def test_refuses():
    cases = {
        "missing": dict(b_eps=(12,)),
        "twice": dict(a_eps=(15, 3, 16)),
        "motion": dict(tweak={15: {"motion": "tier3_v2"}}),
        "steps": dict(tweak={3: {"steps": "300000000"}}),
    }
    for name, kw in [*cases.items(), ("taken", {})]:
        with tempfile.TemporaryDirectory() as d:
            t = Path(d)
            q = setup(t, **kw)
            if name == "taken":
                (q / "mf4090_other.txt").write_text("V2D_WANDB_TAG=x t3-ep3-w2048-cap4m-r3a-dyn4-mf4090 /repo/runs/motion/tier3_dyn4/a 2048 1\n")
            r = gen(t, "t3pdyn")
            assert r.returncode != 0 and "make_copy_queue:" in r.stdout + r.stderr, (name, r.stdout + r.stderr)
            assert not (q / "mf4090_t3pdyn.txt").exists() and not (t / "stages" / "evaluate").exists(), name


def test_t3qdyn_s43():
    with tempfile.TemporaryDirectory() as d:
        t = Path(d)
        q = setup(t)
        assert gen(t, "t3qdyn").returncode == 0
        r = gen(t, "t3qdyn_s43")
        assert r.returncode == 0, r.stdout + r.stderr
        base_f, base_s, base_e = body(q / "ml40_t3qdyn.txt"), body(q / "ml40_t3qdyn_smoke.txt"), body(q / "ml40_t3qdyn.expect.tsv")
        fo, sm, ex = body(q / "ml40_t3qdyn_s43.txt"), body(q / "ml40_t3qdyn_s43_smoke.txt"), body(q / "ml40_t3qdyn_s43.expect.tsv")
        runs = ["t3-ep12-w2048-cap4m-r3a-dyn4-s43-ml40", "t3-ep12-w2048-cap4m-r3a-dyn4s-s43-ml40"]
        olds = ["t3-ep12-w2048-cap4m-r3a-dyn4-ml40", "t3-ep12-w2048-cap4m-r3a-dyn4s-ml40"]
        seedvar = "reset.seed=43 training.seed=43 " + VAR
        for i in range(2):
            assert fo[i] == base_f[i].replace(olds[i], runs[i]).replace(VAR, seedvar), (fo[i], base_f[i])
            assert sm[i] == base_s[i].replace(olds[i], runs[i]).replace(VAR, seedvar), (sm[i], base_s[i])
            want = base_e[i + 1].replace(olds[i], runs[i]).split("\t")
            want[9] = "43"
            assert ex[i + 1] == "\t".join(want), (ex[i + 1], want)
        cand = body(t / "stages" / "evaluate" / "exports" / "cand_t3_t3qdyn_s43_ml40.tsv")
        assert cand == [f"3\t12\t{x}\t349999104" for x in runs], cand
        r = gen(t, "t3qdyn_s43", "--check")
        assert r.returncode == 0 and "插入 reset.seed=43 training.seed=43" in r.stdout and "42 → 43" in r.stdout, r.stdout
    for tweak, msg in (({"seedtok": "reset.seed=7 "}, "已有种子覆盖"),
                       ({"seedtok": "objective.object_position_std=0.05 "}, "不是恰好一个"),
                       ({"seedcol": "7"}, "种子列是 7")):
        with tempfile.TemporaryDirectory() as d:
            t = Path(d)
            q = setup(t, tweak={12: tweak})
            r = gen(t, "t3qdyn_s43")
            assert r.returncode != 0 and msg in r.stdout + r.stderr, (tweak, r.stdout + r.stderr)
            assert not (q / "ml40_t3qdyn_s43.txt").exists() and not (t / "stages" / "evaluate").exists(), tweak
    with tempfile.TemporaryDirectory() as d:
        t = Path(d)
        setup(t)
        assert gen(t, "t3qdyn_s43").returncode == 0
        head = (t / "stages" / "evaluate" / "exports" / "cand_t3_t3qdyn_s43_ml40.tsv").read_text().splitlines()[0]
        assert "本段改种子：seed 43。" in head, head


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("ALL_OK")
