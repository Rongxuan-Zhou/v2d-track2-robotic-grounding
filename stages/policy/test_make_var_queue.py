"""make_var_queue.py 的测试（守计划 agent-doc/plan-t2-scaleout-2026-10-09.md 第 2、3 节与 agent-doc/plan-t3-pilot-2026-10-09.md 第 2、3 节的
run 设计；运行规范未成文，暂无契约 ID）。只用标准库。
临时目录里放假的 Tier 2 基线（30 条 episode：偶数在 ml40、1000 万 buffer，奇数在 mf4090、带 training.replay.capacity=4000000；
ep1 用 -p185-ml40 的名字；support 为 true 的不带 scene.support=false）与两份导出清单，跑生成脚本后核：
  1. t2xa、t2xb 两段：15 条与 9 条 run 的名字与计划顺序；每行没有 buffer 覆盖、有变体 A 两项、基线有的 scene.support=false 照留；
     冒烟与正式行除步数、前缀外一致；W&B 标签 t2x；期望表的 buffer 列 10000000、种子 42，其余列照抄基线；导出清单与 run 一一对应、档位 2
  2. t3p：2 条 run；parquet 换到 /runs/motion/tier3/、序列名不变；覆盖以 400 万 buffer 开头；期望表 buffer 列 4000000；导出清单档位 3；
     t3p2（agent-doc/plan-t3-pilot-v2-2026-10-10.md）：段名也以 t3p 开头，要落到 t3p2 这份清单（取最长的）；run 名带 -v2，
     parquet 在 /runs/motion/tier3_v2/ 下，W&B 标签 t3p2，其余同 t3p；
     t3p3a、t3p3b（agent-doc/plan-t3-v2-train10-2026-10-10.md）：4 条与 6 条，顺序照计划；基线取导出清单（奇数号在 mf4090、
     带 400 万 buffer 覆盖，去掉后照样以 400 万开头）；run 名、parquet 目录同 t3p2，W&B 标签 t3p3
  3. 基线覆盖里有计划没说的键、或一条 episode 在导出清单里出现两次：报错退出，不写文件
  4. 续跑段 t2xa2 --episodes 16,7：只有这两条，顺序照清单（ep7 在前），文件名带段名
  5. 段名不对：预设段又给 --episodes、非预设段不给 --episodes、--episodes 里有清单外的 episode、段名不以清单名开头，都报错不写文件
  6. t1xa、t1ya（agent-doc/plan-next-training-2026-10-12.md 第 3 节 A、C）：基线取 t1_select_r3x.tsv 里的 Tier 1 run（偶数号在 ml40、
     1000 万 buffer，奇数号在 mf4090、带 400 万覆盖）；t1xa 7 条在 mf4090、一律 400 万、W&B 标签 t1x；t1ya 6 条在 ml40、没有 buffer 覆盖、标签 t1y；
     parquet 在 /runs/motion/tier1/；导出清单档位 1。ml40 的 ep10 在第二张期望表里多一行、只差 xeval（真实情形同 t1-ep10-w2048-cap10m-ml40），
     照常生成；那一行若还差别的列（帧数），报“找到 2 种”、不写文件（make_trial_queue.baseline 的改动，同计划审查 M2）
     t1xb、t1yb（agent-doc/plan-t1-chase-2026-10-13.md）：t1xb 5 条在 mf4090、顺序 ep3、19、13、10、14，一律 400 万、标签 t1xb，
     5 条的行与 t1xa 的同一 run 相同（期望表逐字，正式行只差 W&B 标签，冒烟行相同）；t1yb 5 条在 ml40、顺序 ep16、2、3、19、13，
     没有 buffer 覆盖、标签 t1yb，夹具里奇数号的基线在 mf4090、带的 400 万覆盖要去掉；ep16、ep2 的行与 t1ya 的同样相同
  6b. t3pk、t3ps（agent-doc/plan-t3-kfix-retrain-2026-10-16.md）：取法同 t3p3，参考目录换成 tier3_kfix、tier3_v2s，run 名带 -kfix-、-v2s-；
  7. t3p4a、t3p4b、t3qa、t3qb（agent-doc/plan-d-t3hm-train-2026-10-12.md）：四段合起来恰好 29 集、不重复；run 名带 -hm-，parquet 在
     /runs/motion/tier3_hm/；mf4090 两段带 400 万覆盖、期望表 4000000，ml40 两段没有 buffer 覆盖、期望表 10000000；支撑面按 best_hm 的 12 集桌面定，
     与基线不同的几种都核：ep1（基线是 p185 run）、ep15（基线开着支撑面、best_hm 只有地面：要加覆盖、列改 false）、ep17（基线关着、best_hm 有桌面：
     去掉覆盖、列改 true）；ml40 的续跑段 t3qc --episodes 12,13 照总顺序生成（段之间挪集只要 --episodes）
  8. --seed（agent-doc/plan-first-place-2026-10-14.md 选项甲）：t1xb_s43 --episodes 14 --seed 43 在 mf4090 出 t1-ep14-w2048-cap4m-r3a-s43-mf4090，
     t1yb_s43 --episodes 19 --seed 43 在 ml40 出 t1-ep19-w2048-cap10m-r3a-s43-ml40；覆盖是 seed 42 那条的覆盖在变体 A 两项前插
     reset.seed=43 training.seed=43，冒烟行同样；期望表种子列 43、其余列与 seed 42 那条相同；导出清单对得上、抬头注明改种子；
     预设段加 --seed、种子 0、10000、显式 42、段名 _s43 却不给 --seed、给 --seed 43 而段名不以 _s43 结尾，都报错不写文件。
  9. r3o（agent-doc/plan-r3o-2026-10-15.md 2.1–2.3）：t1oa、t1ob 在 mf4090（400 万覆盖在最前、标签 t1o），t2oa、t2ob 在 ml40（没有 buffer 覆盖、标签 t2o），
     顺序照计划；冒烟与正式行在原有环境变量之后、run 名之前多 V2D_OBJ_ORI_WEIGHT=1.0 V2D_OBJ_ORI_STD=0.4，run 名带 -r3o-，覆盖以变体 A 两项结尾；
     现在交的是变体 A 的 run（夹具里 ep20 改成 -r3a-）时改用它的基线配方 run，parquet 相同；抬头写 v040_p185_nfg_oo；
     t1xa 这类原有的段一行也不带 V2D_OBJ_ORI_。
用法：python3 stages/policy/test_make_var_queue.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENV = "V2D_WANDB_TAG=batch V2D_MAX_RETRY=0 V2D_GPU_APPS=1 V2D_WANDB_UPLOAD_CKPT=0"
SENV = "V2D_WANDB_PROJECT=v2d-selftest V2D_WANDB_TAG=smoke V2D_MAX_RETRY=0 V2D_GPU_APPS=1 V2D_WANDB_UPLOAD_CKPT=0"
HEADER = "run\tsmoke\tsequence_id\tplayback\tsrc_frames\tsupport\tpos_tol\tori_tol\tcapacity\tseed\txeval"
VAR = ["objective.object_position_std=0.05", "termination.object_orientation_threshold=0.6"]
T2XA = [29, 23, 6, 18, 7, 16, 0, 21, 15, 10, 27, 3, 5, 11, 17]
T2XB = [9, 1, 13, 19, 2, 4, 14, 25, 28]
T1XA = [17, 21, 10, 13, 19, 3, 14]
T1YA = [2, 11, 16, 4, 9, 25]
T1XB = [3, 19, 13, 10, 14]
T1YB = [16, 2, 3, 19, 13]


def seq(ep):
    return f"2026-01-{ep:02d}_obj_{ep}"


def frames(ep):
    return str(500 + ep)


def support(ep):
    return "true" if ep % 3 == 0 else "false"


def base_run(ep):
    if ep == 1:
        return "t2-ep1-w2048-cap10m-p185-ml40"
    return f"t2-ep{ep}-w2048-cap10m-ml40" if ep % 2 == 0 else f"t2-ep{ep}-w2048-cap4m-mf4090"


def pq(ep, tier=2, motion=None):
    return f"/repo/runs/motion/{motion or f'tier{tier}'}/whole_body/soma/sequence_id={seq(ep)}/robot_name=g1/data.parquet"


def t1_base_run(ep):
    return f"t1-ep{ep}-w2048-cap10m-ml40" if ep % 2 == 0 else f"t1-ep{ep}-w2048-cap4m-mf4090"


def setup_t1(q: Path, ex: Path, dup_conflict=False):
    rows = {"ml40": [], "mf4090": []}
    for ep in range(30):
        run = t1_base_run(ep)
        m = run.rsplit("-", 1)[1]
        over = ([] if support(ep) == "true" else ["scene.support=false"]) + (["training.replay.capacity=4000000"] if m == "mf4090" else [])
        rows[m].append((ep, run, "10000000" if m == "ml40" else "4000000", over))
    sel = []
    for m, items in rows.items():
        exp, fl, sl = [HEADER], ["# Tier 1 基线"], ["# Tier 1 基线冒烟"]
        for ep, run, cap, over in items:
            exp.append("\t".join([run, f"smoke-{run}", seq(ep), "frame", frames(ep), support(ep), "0.015", "0.086", cap, "42", "no"]))
            o = (" " + " ".join(over)) if over else ""
            fl.append(f"{ENV} {run} {pq(ep, 1)} 2048 349999104{o}")
            sl.append(f"{SENV} smoke-{run} {pq(ep, 1)} 2048 4096000{o}")
            sel.append(f"1\t{ep}\t{run}\t349999104\t{m}")
        (q / f"{m}_t1base.expect.tsv").write_text("\n".join(exp) + "\n")
        (q / f"{m}_t1base.txt").write_text("\n".join(fl) + "\n")
        (q / f"{m}_t1base_smoke.txt").write_text("\n".join(sl) + "\n")
    run = t1_base_run(10)   # 第二张表里的同一条 run：只差 xeval；dup_conflict 时帧数也不同
    (q / "ml40_t1dup.expect.tsv").write_text("\n".join([HEADER, "\t".join(
        [run, f"smoke-{run}", seq(10), "frame", frames(10) + ("1" if dup_conflict else ""), support(10), "0.015", "0.086", "10000000", "42", "yes"])]) + "\n")
    (ex / "t1_select_r3x.tsv").write_text("\n".join(sel) + "\n")


def setup(root: Path, bad_key_ep=None, dup_ep=None, t1_dup_conflict=False):
    q = root / "stages" / "policy" / "queues"
    ex = root / "stages" / "evaluate" / "exports"
    q.mkdir(parents=True)
    ex.mkdir(parents=True)
    setup_t1(q, ex, t1_dup_conflict)
    rows = {"ml40": [], "mf4090": []}
    for ep in range(30):
        run = base_run(ep)
        m = run.rsplit("-", 1)[1]
        cap = "10000000" if m == "ml40" else "4000000"
        over = ([] if support(ep) == "true" else ["scene.support=false"]) + (["training.replay.capacity=4000000"] if m == "mf4090" else [])
        if ep == bad_key_ep:
            over.append("training.batch_size=512")
        rows[m].append((ep, run, cap, over))
    for m, items in rows.items():
        exp, fl, sl = [HEADER], ["# 基线"], ["# 基线冒烟"]
        for ep, run, cap, over in items:
            exp.append("\t".join([run, f"smoke-{run}", seq(ep), "frame", frames(ep), support(ep), "0.015", "0.086", cap, "42", "no"]))
            o = (" " + " ".join(over)) if over else ""
            fl.append(f"{ENV} {run} {pq(ep)} 2048 349999104{o}")
            sl.append(f"{SENV} smoke-{run} {pq(ep)} 2048 4096000{o}")
        (q / f"{m}_t2a.expect.tsv").write_text("\n".join(exp) + "\n")
        (q / f"{m}_t2a.txt").write_text("\n".join(fl) + "\n")
        (q / f"{m}_t2a_smoke.txt").write_text("\n".join(sl) + "\n")
        lines = ["# 导出清单"] + [f"2\t{ep}\t{run}\t349999104" for ep, run, _, _ in items]
        if m == "ml40" and dup_ep is not None:
            lines.append(f"2\t{dup_ep}\tt2-ep{dup_ep}-w2048-cap10m-other-ml40\t349999104")
        (ex / f"{m}.tsv").write_text("\n".join(lines) + "\n")
    # t3p 的基线是 ml40 上 cap10m 的 ep8、ep24：两者都是偶数号，上面已经放在 ml40


def gen(root: Path, name: str, *extra):
    return subprocess.run([sys.executable, str(HERE / "make_var_queue.py"), name, "--root", str(root), *extra], capture_output=True, text=True)


def lines(p: Path):
    return [x for x in p.read_text().splitlines() if x and not x.startswith("#")]


def check_set(root, name, eps, run_fmt, tier, cap_col, cap_over, env_tag, machine, motion=None):
    q = root / "stages" / "policy" / "queues"
    want = [run_fmt.format(ep=e) for e in eps]
    f, s = lines(q / f"{machine}_{name}.txt"), lines(q / f"{machine}_{name}_smoke.txt")
    assert [x.split()[5] for x in f] == want, f
    for fx, sx, ep in zip(f, s, eps):
        ft, st = fx.split(), sx.split()
        over = ft[9:]
        assert ft[:5] == f"V2D_WANDB_TAG={env_tag} V2D_MAX_RETRY=0 V2D_GPU_APPS=1 V2D_WANDB_UPLOAD_CKPT=0 V2D_KEEP_STATES=1".split(), fx
        assert st[:5] == SENV.split(), sx
        assert ft[6] == pq(ep, tier, motion) and ft[7] == "2048" and ft[8] == "349999104", fx
        caps = [o for o in over if o.startswith("training.replay.capacity=")]
        assert caps == ([cap_over] if cap_over else []) and (not cap_over or over[0] == cap_over), fx
        assert ("scene.support=false" in over) == (support(ep) == "false") and over[-2:] == VAR, fx
        assert st[5] == "smoke-" + ft[5] and st[8] == "4096000" and st[6:8] == ft[6:8] and st[9:] == over, (fx, sx)
    exp = lines(q / f"{machine}_{name}.expect.tsv")
    assert exp[0] == HEADER
    for row, run, ep in zip(exp[1:], want, eps):
        c = row.split("\t")
        assert c[:2] == [run, "smoke-" + run] and c[2:8] == [seq(ep), "frame", frames(ep), support(ep), "0.015", "0.086"], row
        assert c[8:] == [cap_col, "42", "no"], row
    assert len(exp) == len(eps) + 1
    cand = lines(root / "stages" / "evaluate" / "exports" / f"cand_t{tier}_{name}_{machine}.tsv")
    assert cand == [f"{tier}\t{ep}\t{run}\t349999104" for ep, run in zip(eps, want)], cand


def test_t2x_parts():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        setup(root)
        for part, eps in (("t2xa", T2XA), ("t2xb", T2XB)):
            r = gen(root, part)
            assert r.returncode == 0, r.stdout + r.stderr
            check_set(root, part, eps, "t2-ep{ep}-w2048-cap10m-r3a-ml40", 2, "10000000", None, "t2x", "ml40")


def test_resume_part():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        setup(root)
        r = gen(root, "t2xa2", "--episodes", "16,7")
        assert r.returncode == 0, r.stdout + r.stderr
        check_set(root, "t2xa2", [7, 16], "t2-ep{ep}-w2048-cap10m-r3a-ml40", 2, "10000000", None, "t2x", "ml40")


def test_bad_parts_refuse():
    for args in (["t2xa", "--episodes", "7"], ["t2xz"], ["t2xr", "--episodes", "8"], ["t2xr", "--episodes", "7,7"], ["foo"]):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            setup(root)
            r = gen(root, *args)
            assert r.returncode != 0, (args, r.stdout + r.stderr)
            assert not list((root / "stages" / "policy" / "queues").glob("ml40_t2x*")), args


def test_t3p():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        setup(root)
        r = gen(root, "t3p")
        assert r.returncode == 0, r.stdout + r.stderr
        check_set(root, "t3p", [8, 24], "t3-ep{ep}-w2048-cap4m-r3a-mf4090", 3, "4000000", "training.replay.capacity=4000000", "t3p", "mf4090")


def test_t3p2():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        setup(root)
        r = gen(root, "t3p2")
        assert r.returncode == 0, r.stdout + r.stderr
        check_set(root, "t3p2", [8, 24], "t3-ep{ep}-w2048-cap4m-r3a-v2-mf4090", 3, "4000000", "training.replay.capacity=4000000", "t3p2",
                  "mf4090", motion="tier3_v2")
        assert not (root / "stages" / "policy" / "queues" / "mf4090_t3p.txt").exists()


def test_t3p3_parts():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        setup(root)
        for part, eps in (("t3p3a", [20, 29, 0, 15]), ("t3p3b", [18, 21, 22, 23, 26, 17])):
            r = gen(root, part)
            assert r.returncode == 0, r.stdout + r.stderr
            check_set(root, part, eps, "t3-ep{ep}-w2048-cap4m-r3a-v2-mf4090", 3, "4000000", "training.replay.capacity=4000000", "t3p3",
                      "mf4090", motion="tier3_v2")
        r = gen(root, "t3p3a", "--episodes", "20")
        assert r.returncode != 0, r.stdout + r.stderr


def test_t3pk_t3ps_parts():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        setup(root)
        for part, eps, run_fmt, motion in (("t3pk", [0, 18, 29], "t3-ep{ep}-w2048-cap4m-r3a-kfix-mf4090", "tier3_kfix"),
                                           ("t3ps", [20], "t3-ep{ep}-w2048-cap4m-r3a-v2s-mf4090", "tier3_v2s")):
            r = gen(root, part)
            assert r.returncode == 0, r.stdout + r.stderr
            check_set(root, part, eps, run_fmt, 3, "4000000", "training.replay.capacity=4000000", part, "mf4090", motion=motion)


TABLES_HM = [2, 3, 6, 7, 9, 11, 13, 16, 17, 20, 25, 27]
TABLES_GB5 = sorted(set(TABLES_HM) | {1})   # 计划 agent-doc/plan-t3g5-2026-10-21.md：gb5 给 ep1 补了倚靠用的方盒


def test_t3p4_t3q_parts():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        setup(root)
        q = root / "stages" / "policy" / "queues"
        parts = {"t3p4a": ("mf4090", [15, 17, 1, 2, 3]), "t3p4b": ("mf4090", [12, 13, 14, 16, 19, 21, 25, 27]),
                 "t3qa": ("ml40", [4, 5, 6, 7, 8, 9, 10, 11]), "t3qb": ("ml40", [0, 18, 22, 24, 26, 20, 23, 29])}
        allep = [e for _, eps in parts.values() for e in eps]
        assert sorted(allep) == [e for e in range(30) if e != 28] and len(allep) == 29, allep
        for part, (m, eps) in parts.items():
            r = gen(root, part)
            assert r.returncode == 0, r.stdout + r.stderr
            tag = "t3p4" if m == "mf4090" else "t3q"
            fmt = "t3-ep{ep}-w2048-cap4m-r3a-hm-mf4090" if m == "mf4090" else "t3-ep{ep}-w2048-cap10m-r3a-hm-ml40"
            f, sm = lines(q / f"{m}_{part}.txt"), lines(q / f"{m}_{part}_smoke.txt")
            assert [x.split()[5] for x in f] == [fmt.format(ep=e) for e in eps], f
            exp = lines(q / f"{m}_{part}.expect.tsv")[1:]
            for fx, sx, row, ep in zip(f, sm, exp, eps):
                ft, over, c = fx.split(), fx.split()[9:], row.split("\t")
                assert ft[0] == f"V2D_WANDB_TAG={tag}" and ft[6] == pq(ep, 3, "tier3_hm"), fx
                caps = [o for o in over if o.startswith("training.replay.capacity=")]
                assert caps == (["training.replay.capacity=4000000"] if m == "mf4090" else []), fx
                on = ep in TABLES_HM
                assert ("scene.support=false" in over) == (not on) and over.count("scene.support=false") <= 1 and over[-2:] == VAR, (ep, fx)
                assert c[5] == ("true" if on else "false") and c[2:5] == [seq(ep), "frame", frames(ep)], (ep, row)
                assert c[8] == ("4000000" if m == "mf4090" else "10000000"), row
                assert sx.split()[9:] == over, (fx, sx)
        # 与基线不同的几种：ep15 基线开着（15 % 3 == 0）、best_hm 只有地面；ep17 基线关着、best_hm 有桌面；ep1 的基线是 p185 run、关着
        assert support(15) == "true" and support(17) == "false" and support(1) == "false" and base_run(1).endswith("-p185-ml40")
        r = gen(root, "t3qc", "--episodes", "13,12")
        assert r.returncode == 0, r.stdout + r.stderr
        assert [x.split()[5] for x in lines(q / "ml40_t3qc.txt")] == ["t3-ep12-w2048-cap10m-r3a-hm-ml40", "t3-ep13-w2048-cap10m-r3a-hm-ml40"]


def test_t1_parts():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        setup(root)
        r = gen(root, "t1xa")
        assert r.returncode == 0, r.stdout + r.stderr
        check_set(root, "t1xa", T1XA, "t1-ep{ep}-w2048-cap4m-r3a-mf4090", 1, "4000000", "training.replay.capacity=4000000", "t1x", "mf4090")
        r = gen(root, "t1ya")
        assert r.returncode == 0, r.stdout + r.stderr
        check_set(root, "t1ya", T1YA, "t1-ep{ep}-w2048-cap10m-r3a-ml40", 1, "10000000", None, "t1y", "ml40")
        r = gen(root, "t1xb")
        assert r.returncode == 0, r.stdout + r.stderr
        check_set(root, "t1xb", T1XB, "t1-ep{ep}-w2048-cap4m-r3a-mf4090", 1, "4000000", "training.replay.capacity=4000000", "t1xb", "mf4090")
        r = gen(root, "t1yb")
        assert r.returncode == 0, r.stdout + r.stderr
        check_set(root, "t1yb", T1YB, "t1-ep{ep}-w2048-cap10m-r3a-ml40", 1, "10000000", None, "t1yb", "ml40")
        # 同一条 run 在两段里都有：期望表的行逐字相同，正式行只差 W&B 标签，冒烟行相同
        q = root / "stages" / "policy" / "queues"
        same = [("mf4090", "t1xa", "t1xb", "t1x", ep, f"t1-ep{ep}-w2048-cap4m-r3a-mf4090") for ep in T1XB]
        same += [("ml40", "t1ya", "t1yb", "t1y", ep, f"t1-ep{ep}-w2048-cap10m-r3a-ml40") for ep in (16, 2)]
        for m, a, b, tag, ep, run in same:
            pick = lambda p: [x for x in lines(p) if f" {run} " in f" {x} " or x.startswith(run + "\t")]   # noqa: E731
            ps = lambda p: [x for x in lines(p) if f" smoke-{run} " in f" {x} "]   # noqa: E731
            assert pick(q / f"{m}_{a}.expect.tsv") == pick(q / f"{m}_{b}.expect.tsv") != [], run
            fa, fb = pick(q / f"{m}_{a}.txt"), pick(q / f"{m}_{b}.txt")
            assert len(fa) == len(fb) == 1 and fa[0].replace(f"V2D_WANDB_TAG={tag} ", f"V2D_WANDB_TAG={b} ") == fb[0], run
            assert ps(q / f"{m}_{a}_smoke.txt") == ps(q / f"{m}_{b}_smoke.txt") != [], run
        r = gen(root, "t1xa", "--episodes", "10")
        assert r.returncode != 0, r.stdout + r.stderr


def test_seed_option():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        setup(root)
        q = root / "stages" / "policy" / "queues"
        for m, base_part, part, ep, run42, run43 in (
                ("mf4090", "t1xb", "t1xb_s43", 14, "t1-ep14-w2048-cap4m-r3a-mf4090", "t1-ep14-w2048-cap4m-r3a-s43-mf4090"),
                ("ml40", "t1yb", "t1yb_s43", 19, "t1-ep19-w2048-cap10m-r3a-ml40", "t1-ep19-w2048-cap10m-r3a-s43-ml40")):
            assert gen(root, base_part).returncode == 0
            r = gen(root, part, "--episodes", str(ep), "--seed", "43")
            assert r.returncode == 0, r.stdout + r.stderr
            f43, s43 = lines(q / f"{m}_{part}.txt"), lines(q / f"{m}_{part}_smoke.txt")
            f42 = [x for x in lines(q / f"{m}_{base_part}.txt") if f" {run42} " in f" {x} "]
            s42 = [x for x in lines(q / f"{m}_{base_part}_smoke.txt") if f" smoke-{run42} " in f" {x} "]
            assert len(f43) == len(s43) == len(f42) == len(s42) == 1, (f43, s43)
            ft, bt = f43[0].split(), f42[0].split()
            assert ft[5] == run43 and ft[6:9] == bt[6:9], (ft, bt)
            assert ft[9:] == bt[9:-2] + ["reset.seed=43", "training.seed=43"] + VAR, (ft[9:], bt[9:])
            st, sb = s43[0].split(), s42[0].split()
            assert st[5] == "smoke-" + run43 and st[6:9] == sb[6:9] and st[9:] == ft[9:], (st, sb)
            e43 = lines(q / f"{m}_{part}.expect.tsv")[1].split("\t")
            e42 = [x for x in lines(q / f"{m}_{base_part}.expect.tsv") if x.startswith(run42 + "\t")][0].split("\t")
            assert e43[:2] == [run43, "smoke-" + run43] and e43[2:9] == e42[2:9] and e43[9:] == ["43", "no"], (e43, e42)
            cand = lines(root / "stages" / "evaluate" / "exports" / f"cand_t1_{part}_{m}.tsv")
            assert cand == [f"1\t{ep}\t{run43}\t349999104"], cand
            note = "本段改种子：种子 43（reset.seed=43 training.seed=43）"
            assert note in (q / f"{m}_{part}.txt").read_text()
            assert note in (root / "stages" / "evaluate" / "exports" / f"cand_t1_{part}_{m}.tsv").read_text()
        bad = (["t1xb", "--seed", "43"], ["t1xb_s0", "--episodes", "14", "--seed", "0"],
               ["t1xb_s10000", "--episodes", "14", "--seed", "10000"], ["t1xb_s42", "--episodes", "14", "--seed", "42"],
               ["t1xb_s44", "--episodes", "14"], ["t1xb_new", "--episodes", "14", "--seed", "43"],
               ["t1xb_s44", "--episodes", "14", "--seed", "43"])
        before = (q / "mf4090_t1xb.txt").read_text()
        for args in bad:
            r = gen(root, *args)
            assert r.returncode != 0, (args, r.stdout)
            assert args[0] == "t1xb" or not (q / f"mf4090_{args[0]}.txt").exists(), args
        assert (q / "mf4090_t1xb.txt").read_text() == before


def test_t1_xeval_conflict_refuses():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        setup(root, t1_dup_conflict=True)
        r = gen(root, "t1xa")
        out = r.stdout + r.stderr
        assert r.returncode != 0 and "找到 2 种" in out, out
        assert not (root / "stages" / "policy" / "queues" / "mf4090_t1xa.txt").exists()


def test_unknown_override_refuses():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        setup(root, bad_key_ep=6)
        r = gen(root, "t2xa")
        out = r.stdout + r.stderr
        assert r.returncode != 0 and "计划没说怎么处理的键" in out, out
        assert not (root / "stages" / "policy" / "queues" / "ml40_t2xa.txt").exists()


def test_duplicate_export_row_refuses():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        setup(root, dup_ep=10)
        r = gen(root, "t2xa")
        assert r.returncode != 0 and "不止一行" in r.stdout + r.stderr, r.stdout + r.stderr
        assert not (root / "stages" / "evaluate" / "exports" / "cand_t2_t2xa_ml40.tsv").exists()


def test_r3o_parts():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        setup(root)
        sel = root / "stages" / "evaluate" / "exports" / "t1_select_r3x.tsv"
        sel.write_text(sel.read_text().replace("t1-ep20-w2048-cap10m-ml40", "t1-ep20-w2048-cap10m-r3a-ml40"))
        q = root / "stages" / "policy" / "queues"
        r3o = ["V2D_OBJ_ORI_WEIGHT=1.0", "V2D_OBJ_ORI_STD=0.4"]
        cases = [("t1oa", [20, 25, 28, 11], "mf4090", 1, "t1-ep{ep}-w2048-cap4m-r3o-mf4090", "training.replay.capacity=4000000", "t1o"),
                 ("t1ob", [4, 9, 1, 2], "mf4090", 1, "t1-ep{ep}-w2048-cap4m-r3o-mf4090", "training.replay.capacity=4000000", "t1o"),
                 ("t2oa", [12, 25, 28, 1], "ml40", 2, "t2-ep{ep}-w2048-cap10m-r3o-ml40", None, "t2o"),
                 ("t2ob", [4, 17, 9, 2, 11], "ml40", 2, "t2-ep{ep}-w2048-cap10m-r3o-ml40", None, "t2o")]
        for name, eps, m, tier, fmt, cap, tag in cases:
            p = gen(root, name)
            assert p.returncode == 0, p.stdout + p.stderr
            f, s = lines(q / f"{m}_{name}.txt"), lines(q / f"{m}_{name}_smoke.txt")
            assert len(f) == len(s) == len(eps)
            for fx, sx, ep in zip(f, s, eps):
                ft, st = fx.split(), sx.split()
                assert ft[:7] == f"V2D_WANDB_TAG={tag} V2D_MAX_RETRY=0 V2D_GPU_APPS=1 V2D_WANDB_UPLOAD_CKPT=0 V2D_KEEP_STATES=1".split() + r3o, fx
                assert st[:7] == SENV.split() + r3o, sx
                assert ft[7] == fmt.format(ep=ep) and st[7] == "smoke-" + ft[7], (fx, sx)
                assert ft[8] == pq(ep, tier) and ft[9:11] == ["2048", "349999104"] and st[8:11] == [pq(ep, tier), "2048", "4096000"], fx
                over = ft[11:]
                assert over[-2:] == VAR and over.count(VAR[0]) == 1 and st[11:] == over, fx
                caps = [o for o in over if o.startswith("training.replay.capacity=")]
                assert caps == ([cap] if cap else []) and (not cap or over[0] == cap), fx
            head = (q / f"{m}_{name}.txt").read_text().splitlines()[0]
            assert "v040_p185_nfg_oo" in head and "V2D_OBJ_ORI_WEIGHT=1.0 V2D_OBJ_ORI_STD=0.4" in head, head
            cand = lines(root / "stages" / "evaluate" / "exports" / f"cand_t{tier}_{name}_{m}.tsv")
            assert cand == [f"{tier}\t{ep}\t{fmt.format(ep=ep)}\t349999104" for ep in eps], cand
        p = gen(root, "t1xa")
        assert p.returncode == 0, p.stderr
        assert not any("V2D_OBJ_ORI_" in x for x in lines(q / "mf4090_t1xa.txt") + lines(q / "mf4090_t1xa_smoke.txt"))


def test_t1c_part():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        setup(root)
        sel = root / "stages" / "evaluate" / "exports" / "t1_select_r3x.tsv"
        sel.write_text(sel.read_text().replace("t1-ep25-w2048-cap10m-ml40", "t1-ep25-w2048-cap10m-r3a-ml40"))
        p = gen(root, "t1c")
        assert p.returncode == 0, p.stdout + p.stderr
        check_set(root, "t1c", [25, 28, 11, 1], "t1-ep{ep}-w2048-cap10m-r3a-c10-ml40", 1, "10000000", None, "t1c", "ml40", motion="tier1_c10")
        f = lines(root / "stages" / "policy" / "queues" / "ml40_t1c.txt")
        assert all("/runs/motion/tier1_c10/" in x and "/runs/motion/tier1/" not in x and "V2D_OBJ_ORI_" not in x for x in f), f


def test_t1c2_part():
    """路 A（agent-doc/plan-c10-extend-2026-10-16.md）：t1c 清单的 ep9、ep2 用 --episodes 另起一段，预设段 t1c 不变。"""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        setup(root)
        p = gen(root, "t1c2", "--episodes", "2,9")
        assert p.returncode == 0, p.stdout + p.stderr
        check_set(root, "t1c2", [9, 2], "t1-ep{ep}-w2048-cap10m-r3a-c10-ml40", 1, "10000000", None, "t1c", "ml40", motion="tier1_c10")
        assert gen(root, "t1c", "--episodes", "9").returncode != 0
        assert gen(root, "t1c3", "--episodes", "4").returncode != 0   # ep4 不在 t1c 清单里


def test_t3pg_t3qg_parts():
    """gb3（计划 agent-doc/plan-t3-gb3-switch-2026-10-17.md 做法 1c）与 gb5（计划 agent-doc/plan-t3g5-2026-10-21.md）：与 t3p4、t3q 同集的行遮掉 run 名、参考目录、W&B 标签后逐字相同。"""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        setup(root)
        q = root / "stages" / "policy" / "queues"
        for part, m, eps, ref_part, ref_tag, fmt, ref_fmt, motion, new_tag in (
                ("t3pg", "mf4090", [20, 0, 22, 21], "t3p4z", "t3p4", "t3-ep{ep}-w2048-cap4m-r3a-gb3-mf4090", "t3-ep{ep}-w2048-cap4m-r3a-hm-mf4090", "tier3_gb3", "t3pg"),
                ("t3qg", "ml40", [6, 8, 7, 29], "t3qz", "t3q", "t3-ep{ep}-w2048-cap10m-r3a-gb3-ml40", "t3-ep{ep}-w2048-cap10m-r3a-hm-ml40", "tier3_gb3", "t3qg"),
                # 计划 agent-doc/plan-t3g5-2026-10-21.md：gb5 的两段与 hm 段同集的行遮掉后逐字相同
                ("t3pg5a", "mf4090", [5, 10, 15, 27], "t3p4y", "t3p4", "t3-ep{ep}-w2048-cap4m-r3a-gb5-mf4090", "t3-ep{ep}-w2048-cap4m-r3a-hm-mf4090", "tier3_gb5", "t3pg5"),
                ("t3pg5b", "mf4090", [13, 19, 1, 9, 17], "t3p4x", "t3p4", "t3-ep{ep}-w2048-cap4m-r3a-gb5-mf4090", "t3-ep{ep}-w2048-cap4m-r3a-hm-mf4090", "tier3_gb5", "t3pg5"),
                ("t3qg5a", "ml40", [3, 12, 16], "t3qy", "t3q", "t3-ep{ep}-w2048-cap10m-r3a-gb5-ml40", "t3-ep{ep}-w2048-cap10m-r3a-hm-ml40", "tier3_gb5", "t3qg5"),
                ("t3qg5b", "ml40", [14, 23, 2, 4, 11, 25], "t3qx", "t3q", "t3-ep{ep}-w2048-cap10m-r3a-gb5-ml40", "t3-ep{ep}-w2048-cap10m-r3a-hm-ml40", "tier3_gb5", "t3qg5")):
            r = gen(root, part)
            assert r.returncode == 0, r.stdout + r.stderr
            r = gen(root, ref_part, "--episodes", ",".join(map(str, eps)))
            assert r.returncode == 0, r.stdout + r.stderr
            f = lines(q / f"{m}_{part}.txt")
            assert [x.split()[5] for x in f] == [fmt.format(ep=e) for e in eps], f
            assert all(f"/runs/motion/{motion}/" in x and "/runs/motion/tier3_hm/" not in x for x in f), f

            def mask(x, run_fmt, motion, tag, ep):
                x = x.replace(run_fmt.format(ep=ep), "<RUN>").replace(f"/runs/motion/{motion}/", "/runs/motion/<REF>/").replace(f"V2D_WANDB_TAG={tag}", "V2D_WANDB_TAG=<TAG>")
                x = x.replace(" scene.support=false", "")   # 支撑面单独核（gb5 多开 ep1，计划 agent-doc/plan-t3g5-2026-10-21.md 复核 MAJOR 2）
                if "\t" in x:   # 期望表：第 6 列是支撑面
                    c = x.split("\t"); c[5] = "<SUP>"; x = "\t".join(c)
                return x

            for suffix in (".txt", "_smoke.txt", ".expect.tsv"):
                new_l = [x for x in (q / f"{m}_{part}{suffix}").read_text().splitlines() if x and not x.startswith("#")]
                ref_l = [x for x in (q / f"{m}_{ref_part}{suffix}").read_text().splitlines() if x and not x.startswith("#")]
                if suffix == ".expect.tsv":
                    assert new_l[0] == ref_l[0], (new_l[0], ref_l[0])
                    new_l, ref_l = new_l[1:], ref_l[1:]
                by_ep = {}
                for x in ref_l:
                    for e in eps:
                        if ref_fmt.format(ep=e) in x:
                            by_ep[e] = mask(x, ref_fmt, "tier3_hm", ref_tag, e)
                assert len(new_l) == len(eps) and sorted(by_ep) == sorted(eps), (suffix, new_l, by_ep)
                for x, e in zip(new_l, eps):
                    assert mask(x, fmt, motion, new_tag, e) == by_ep[e], (suffix, e, x, by_ep[e])
            on = [e for e in eps if e in (TABLES_GB5 if motion == "tier3_gb5" else TABLES_HM)]
            assert all(("scene.support=false" in x) == (int(x.split()[5].split("-")[1][2:]) not in on) for x in f), f


def test_t2m_part():
    """路 B：Tier 2 段换参考目录 /runs/motion/tier2/ → /runs/motion/tier2_c5/，mf4090、400 万，基线取 Tier 2 导出清单（ep1 的基线在 ml40）。"""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        setup(root)
        p = gen(root, "t2m")
        assert p.returncode == 0, p.stdout + p.stderr
        check_set(root, "t2m", [25, 9, 1], "t2-ep{ep}-w2048-cap4m-r3a-c5-mf4090", 2, "4000000", "training.replay.capacity=4000000", "t2m", "mf4090",
                  motion="tier2_c5")
        f = lines(root / "stages" / "policy" / "queues" / "mf4090_t2m.txt")
        assert all("/runs/motion/tier2_c5/" in x and "/runs/motion/tier2/" not in x and "V2D_OBJ_ORI_" not in x for x in f), f



def test_t2fc_part():
    """计划 agent-doc/plan-t2fc-2026-10-20.md：Tier 2 段换参考目录 /runs/motion/tier2/ → /runs/motion/tier2_fc8/，ml40、1000 万（不加 capacity 覆盖），
    基线取 Tier 2 导出清单（ep25 的基线在 mf4090、ep4 的在 ml40），run 名带 -fc8-。"""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        setup(root)
        p = gen(root, "t2fc")
        assert p.returncode == 0, p.stdout + p.stderr
        f = lines(root / "stages" / "policy" / "queues" / "ml40_t2fc.txt")
        assert len(f) == 2 and [x.split()[5] for x in f] == ["t2-ep25-w2048-cap10m-r3a-fc8-ml40", "t2-ep4-w2048-cap10m-r3a-fc8-ml40"], f
        assert all("/runs/motion/tier2_fc8/" in x and "/runs/motion/tier2/" not in x and "training.replay.capacity=" not in x for x in f), f



def test_ho_parts():
    """计划 agent-doc/plan-t12lift-2026-10-20.md：t2ho（Tier 2 ep25、ep28）与 t1ho（Tier 1 ep25）换参考目录到 tier2_ho / tier1_ho，ml40、1000 万，run 名带 -ho-。"""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        setup(root)
        for part, tier, eps in (("t2ho", 2, [25, 28]), ("t1ho", 1, [25])):
            p = gen(root, part)
            assert p.returncode == 0, p.stdout + p.stderr
            f = lines(root / "stages" / "policy" / "queues" / f"ml40_{part}.txt")
            assert [x.split()[5] for x in f] == [f"t{tier}-ep{e}-w2048-cap10m-r3a-ho-ml40" for e in eps], f
            assert all(f"/runs/motion/tier{tier}_ho/" in x and f"/runs/motion/tier{tier}/" not in x and "training.replay.capacity=" not in x for x in f), f



def test_t2tm_part():
    """计划 agent-doc/plan-t2term-2026-10-20.md：t2tm 是 ep25、ep28 的变体 A 加 termination.object_position_threshold=0.3，冒烟与正式都加，ml40、1000 万，run 名带 -ot3-。"""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        setup(root)
        p = gen(root, "t2tm")
        assert p.returncode == 0, p.stdout + p.stderr
        q = root / "stages" / "policy" / "queues"
        f, sm = lines(q / "ml40_t2tm.txt"), lines(q / "ml40_t2tm_smoke.txt")
        assert [x.split()[5] for x in f] == [f"t2-ep{e}-w2048-cap10m-r3a-ot3-ml40" for e in (25, 28)], f
        for x in f + sm:
            assert x.split().count("termination.object_position_threshold=0.3") == 1, x
            assert "training.replay.capacity=" not in x, x


def test_t2mtr_part():
    """计划 agent-doc/plan-t2mtr-2026-10-21.md 第 1 节第 1 条：t2mtr 是 ep25、ep4、ep28 的变体 A 加 termination.object_position_threshold=0.3，冒烟与正式都加，mf4090、400 万，run 名带 -ot3-。"""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        setup(root)
        p = gen(root, "t2mtr")
        assert p.returncode == 0, p.stdout + p.stderr
        q = root / "stages" / "policy" / "queues"
        f, sm = lines(q / "mf4090_t2mtr.txt"), lines(q / "mf4090_t2mtr_smoke.txt")
        assert [x.split()[5] for x in f] == [f"t2-ep{e}-w2048-cap4m-r3a-ot3-mf4090" for e in (25, 4, 28)], f
        for x in f + sm:
            assert x.split().count("termination.object_position_threshold=0.3") == 1, x
            assert x.split().count("training.replay.capacity=4000000") == 1, x
            assert "seed=" not in x, x


def test_t3qs_s43():
    """计划 agent-doc/plan-t3s43-2026-10-26.md 第 1 节第 1 条（复核 MAJOR 1）：t3qs_s43 --episodes 5,10,3,14 --seed 43 生成 gb5、ml40 的 seed 43 run，顺序 ep5、10、3、14，ep3 开支撑面。"""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        setup(root)
        p = gen(root, "t3qs_s43", "--episodes", "5,10,3,14", "--seed", "43")
        assert p.returncode == 0, p.stdout + p.stderr
        q = root / "stages" / "policy" / "queues"
        f, sm = lines(q / "ml40_t3qs_s43.txt"), lines(q / "ml40_t3qs_s43_smoke.txt")
        assert [x.split()[5] for x in f] == [f"t3-ep{e}-w2048-cap10m-r3a-gb5-s43-ml40" for e in (5, 10, 3, 14)], f
        for x in f + sm:
            assert "reset.seed=43" in x.split() and "training.seed=43" in x.split(), x
            assert "/runs/motion/tier3_gb5/" in x, x
            assert ("scene.support=false" in x.split()) == ("-ep3-" not in x), x


def test_t3far_parts():
    """计划 agent-doc/plan-t3far-2026-10-27.md 第 1 节第 1 条：t3qf1 --episodes 15,27 生成 gb5、ml40（1000 万）的 seed 42 run，顺序 ep15、27；
    t3pg5c_s43 --episodes 13,19 --seed 43 生成 gb5、mf4090（400 万）的 seed 43 run，顺序 ep13、19（10/21 B 段留下的 seed 42 空目录不重名）。"""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        setup(root)
        q = root / "stages" / "policy" / "queues"
        p = gen(root, "t3qf1", "--episodes", "15,27")
        assert p.returncode == 0, p.stdout + p.stderr
        f = lines(q / "ml40_t3qf1.txt")
        assert [x.split()[5] for x in f] == [f"t3-ep{e}-w2048-cap10m-r3a-gb5-ml40" for e in (15, 27)], f
        p = gen(root, "t3pg5c_s43", "--episodes", "13,19", "--seed", "43")
        assert p.returncode == 0, p.stdout + p.stderr
        g = lines(q / "mf4090_t3pg5c_s43.txt")
        assert [x.split()[5] for x in g] == [f"t3-ep{e}-w2048-cap4m-r3a-gb5-s43-mf4090" for e in (13, 19)], g
        for x in f + lines(q / "ml40_t3qf1_smoke.txt"):
            assert "seed=" not in x, x
            assert "/runs/motion/tier3_gb5/" in x, x
        for x in g + lines(q / "mf4090_t3pg5c_s43_smoke.txt"):
            assert "reset.seed=43" in x.split() and "training.seed=43" in x.split(), x
            assert "/runs/motion/tier3_gb5/" in x, x
        for x in g:
            assert "training.replay.capacity=4000000" in x.split(), x
        for x in f:
            assert "training.replay.capacity=4000000" not in x.split(), x


def test_precision_parts():
    """计划 agent-doc/plan-precision-2026-10-18.md 第 1 节：t2mv 是 ep23 的变体 A（mf4090、400 万）；t2xp、t1xp 是 r3p（变体 A 只把 σ 改成 0.03），
    基线取现在交的 run 去掉 -r3a- 的那条，覆盖以 r3p 两项结尾，run 名带 -r3p-。计划 agent-doc/plan-t2mx-2026-10-19.md：t2mx 是 ep13、19、14 的变体 A（mf4090、400 万）。"""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        setup(root)
        q = root / "stages" / "policy" / "queues"
        var_p = ["objective.object_position_std=0.03", VAR[1]]
        cases = [("t2mv", [23], "mf4090", 2, "t2-ep{ep}-w2048-cap4m-r3a-mf4090", "training.replay.capacity=4000000", "t2mv", VAR),
                 ("t2xp", [14, 13, 19, 16], "ml40", 2, "t2-ep{ep}-w2048-cap10m-r3p-ml40", None, "t2xp", var_p),
                 ("t1xp", [10, 29, 19], "mf4090", 1, "t1-ep{ep}-w2048-cap4m-r3p-mf4090", "training.replay.capacity=4000000", "t1xp", var_p),
                 ("t2mx", [13, 19, 14], "mf4090", 2, "t2-ep{ep}-w2048-cap4m-r3a-mf4090", "training.replay.capacity=4000000", "t2mx", VAR)]
        for name, eps, m, tier, fmt, cap, tag, var in cases:
            p = gen(root, name)
            assert p.returncode == 0, p.stdout + p.stderr
            f, s = lines(q / f"{m}_{name}.txt"), lines(q / f"{m}_{name}_smoke.txt")
            assert len(f) == len(s) == len(eps)
            for fx, sx, ep in zip(f, s, eps):
                ft, st = fx.split(), sx.split()
                assert ft[0] == f"V2D_WANDB_TAG={tag}" and ft[5] == fmt.format(ep=ep) and st[5] == "smoke-" + ft[5], (fx, sx)
                assert ft[6] == pq(ep, tier) and ft[7:9] == ["2048", "349999104"] and st[6:9] == [pq(ep, tier), "2048", "4096000"], fx
                over = ft[9:]
                assert over[-2:] == var and st[9:] == over, fx
                assert sum(o.startswith("objective.object_position_std=") for o in over) == 1, fx
                caps = [o for o in over if o.startswith("training.replay.capacity=")]
                assert caps == ([cap] if cap else []) and (not cap or over[0] == cap), fx
            head = (q / f"{m}_{name}.txt").read_text().splitlines()[0]
            assert ("r3p = objective.object_position_std=0.03" in head) == (var == var_p), head
            cand = lines(root / "stages" / "evaluate" / "exports" / f"cand_t{tier}_{name}_{m}.tsv")
            assert cand == [f"{tier}\t{ep}\t{fmt.format(ep=ep)}\t349999104" for ep in eps], cand

def test_smoke_extra():
    """计划 agent-doc/plan-t2sd-2026-10-19.md：t2msd 是 ep25 的变体 A（mf4090、400 万），冒烟行比正式行多一项
    training.checkpoint.save_interval_environment_steps=409600，正式行不加；别的段的冒烟行与正式行覆盖相同（见 test_precision_parts）。"""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        setup(root)
        q = root / "stages" / "policy" / "queues"
        p = gen(root, "t2msd")
        assert p.returncode == 0, p.stdout + p.stderr
        f, s = lines(q / "mf4090_t2msd.txt"), lines(q / "mf4090_t2msd_smoke.txt")
        assert len(f) == len(s) == 1
        ft, st = f[0].split(), s[0].split()
        assert ft[5] == "t2-ep25-w2048-cap4m-r3a-sd-mf4090" and st[5] == "smoke-" + ft[5], (f, s)
        assert st[9:] == ft[9:] + ["training.checkpoint.save_interval_environment_steps=409600"], (f, s)
        assert ft[9:][-2:] == VAR and not any(o.startswith("training.checkpoint.") for o in ft[9:]), f
        assert "冒烟另加 training.checkpoint.save_interval_environment_steps=409600" in (q / "mf4090_t2msd.txt").read_text().splitlines()[0]


def test_t1palm_parts():
    """计划 agent-doc/plan-t1palm-2026-10-30.md（复核 MINOR 7）：t1zh 的每一行等于 t1za 同集的行，只把 run 名的 -r3a- 换成 -r3h-、
    标签 t1za 换成 t1zh，覆盖末尾多一项 objective.palm_position_std=0.05；t1zq（mf4090、400 万）每行覆盖以变体 A 加这一项结尾。"""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        setup(root)
        q = root / "stages" / "policy" / "queues"
        for part in ("t1za", "t1zh", "t1zq"):
            r = gen(root, part)
            assert r.returncode == 0, r.stdout + r.stderr
        palm = "objective.palm_position_std=0.05"
        for kind in ("", "_smoke"):
            za = {x.split()[5].replace("-r3a-", "-r3h-"): x.split() for x in lines(q / f"ml40_t1za{kind}.txt")}
            zh = lines(q / f"ml40_t1zh{kind}.txt")
            assert [x.split()[5] for x in zh] == [("smoke-" if kind else "") + f"t1-ep{e}-w2048-cap10m-r3h-ml40" for e in (25, 9, 4)], zh
            for x in zh:
                t = x.split()
                base = [v.replace("V2D_WANDB_TAG=t1za", "V2D_WANDB_TAG=t1zh") for v in za[t[5]]]
                base[5] = t[5]
                assert t == base + [palm], (t, base)
            zq = lines(q / f"mf4090_t1zq{kind}.txt")
            assert [x.split()[5] for x in zq] == [("smoke-" if kind else "") + f"t1-ep{e}-w2048-cap4m-r3h-mf4090" for e in (20, 17)], zq
            for x in zq:
                over = x.split()[9:]
                assert over[0] == "training.replay.capacity=4000000" and over[-3:-1] == VAR and over[-1] == palm and over.count(palm) == 1, x


def test_t1palm3_parts():
    """计划 agent-doc/plan-t1palm-2026-10-30.md 第 3 版：t1zi、t1zr 与 t1zh、t1zq 逐行相同，只把 -r3h- 换成 -r3i-、标签换段名、
    手掌尺度 0.05 换成 0.1；t1zi 只有 ep25、9。"""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        setup(root)
        q = root / "stages" / "policy" / "queues"
        for part in ("t1za", "t1zh", "t1zq", "t1zi", "t1zr"):
            r = gen(root, part)
            assert r.returncode == 0, r.stdout + r.stderr
        for kind in ("", "_smoke"):
            for m, old, new, eps, cap in (("ml40", "t1zh", "t1zi", (25, 9), "cap10m"), ("mf4090", "t1zq", "t1zr", (20, 17), "cap4m")):
                ref = {x.split()[5].replace("-r3h-", "-r3i-"): x.split() for x in lines(q / f"{m}_{old}{kind}.txt")}
                got = lines(q / f"{m}_{new}{kind}.txt")
                assert [x.split()[5] for x in got] == [("smoke-" if kind else "") + f"t1-ep{e}-w2048-{cap}-r3i-{m}" for e in eps], got
                for x in got:
                    t = x.split()
                    base = [v.replace(f"V2D_WANDB_TAG={old}", f"V2D_WANDB_TAG={new}") for v in ref[t[5]]]
                    base[5] = t[5]
                    base[-1] = "objective.palm_position_std=0.1"
                    assert t == base and x.count("palm_position_std") == 1, (t, base)



def test_t1hard_novar():
    """计划 agent-doc/plan-t1hard-2026-10-31.md：t1hr、t1hp、t1hf 是基础配方（novar），不带变体 A 的覆盖；只多声明的那一项
    （t1hr 两个 r3o 变量、run 名带 -r3o-；t1hp 手掌位置终止 0.08；t1hf 抓握闭合项 20 与 400 万容量），冒烟与正式一致。"""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        setup(root)
        q = root / "stages" / "policy" / "queues"
        want = {"t1hr": ("ml40", [], True, ["t1-ep17-w2048-cap10m-r3o-nv-ml40"]),
                "t1hp": ("ml40", ["termination.palm_position_threshold=0.08"], False,
                         ["t1-ep25-w2048-cap10m-pt08-ml40", "t1-ep9-w2048-cap10m-pt08-ml40"]),
                "t1hf": ("mf4090", ["training.replay.capacity=4000000", "objective.force_closure_weight=20.0"], False,
                         ["t1-ep25-w2048-cap4m-fc20-mf4090", "t1-ep9-w2048-cap4m-fc20-mf4090"])}
        for part, (m, extra, r3o, runs) in want.items():
            r = gen(root, part)
            assert r.returncode == 0, r.stdout + r.stderr
            for kind in ("", "_smoke"):
                got = lines(q / f"{m}_{part}{kind}.txt")
                names = [t for x in got for t in x.split() if t.startswith(("t1-", "smoke-t1-"))]
                assert names == [("smoke-" if kind else "") + n for n in runs], names
                for x in got:
                    t = x.split()
                    assert "object_position_std" not in x and "object_orientation_threshold" not in x, x
                    assert (" V2D_OBJ_ORI_WEIGHT=1.0 V2D_OBJ_ORI_STD=0.4 " in f" {x} ") == r3o, x
                    i = next(k for k, v in enumerate(t) if v.isdigit() and k > 0 and t[k - 1] == "2048")
                    over = [v for v in t[i + 1:] if v != "scene.support=false"]
                    assert over == extra, (part, over)


def test_t1mppe_pt08():
    """计划 agent-doc/plan-t1mppe-2026-11-01.md：t1hq 是 ep4 的基础配方加手掌位置终止 0.08（novar，不带变体 A）；
    t1hq_s43 --episodes 25 --seed 43 得 ep25 的 pt08 seed 43，只多两个种子覆盖；冒烟与正式一致。"""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        setup(root)
        q = root / "stages" / "policy" / "queues"
        want = {("t1hq", ()): (["termination.palm_position_threshold=0.08"], ["t1-ep4-w2048-cap10m-pt08-ml40"]),
                ("t1hq_s43", ("--episodes", "25", "--seed", "43")):
                    (["reset.seed=43", "training.seed=43", "termination.palm_position_threshold=0.08"], ["t1-ep25-w2048-cap10m-pt08-s43-ml40"])}
        for (part, args), (extra, runs) in want.items():
            r = gen(root, part, *args)
            assert r.returncode == 0, r.stdout + r.stderr
            for kind in ("", "_smoke"):
                got = lines(q / f"ml40_{part}{kind}.txt")
                names = [t for x in got for t in x.split() if t.startswith(("t1-", "smoke-t1-"))]
                assert names == [("smoke-" if kind else "") + n for n in runs], names
                for x in got:
                    t = x.split()
                    assert "object_position_std" not in x and "object_orientation_threshold" not in x and "V2D_OBJ_ORI" not in x, x
                    i = next(k for k, v in enumerate(t) if v.isdigit() and k > 0 and t[k - 1] == "2048")
                    over = [v for v in t[i + 1:] if v != "scene.support=false"]
                    assert sorted(over) == sorted(extra), (part, over)
        r = gen(root, "t1hq", "--episodes", "25")
        assert r.returncode != 0, "预设段不许再给 --episodes"


def test_t1fill():
    """计划 agent-doc/plan-t1fill-2026-11-01.md：t1hq2 是 ep25 的基础配方加手掌位置终止 0.05；t1hr2 是 ep26、23 的基础配方加物体朝向项（-r3o-nv-）；
    都不带变体 A，只多声明的那一项，冒烟与正式一致。"""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        setup(root)
        q = root / "stages" / "policy" / "queues"
        want = {"t1hq2": (["termination.palm_position_threshold=0.05"], False, ["t1-ep25-w2048-cap10m-pt05-ml40"]),
                "t1hr2": ([], True, ["t1-ep26-w2048-cap10m-r3o-nv-ml40", "t1-ep23-w2048-cap10m-r3o-nv-ml40"])}
        for part, (extra, r3o, runs) in want.items():
            r = gen(root, part)
            assert r.returncode == 0, r.stdout + r.stderr
            for kind in ("", "_smoke"):
                got = lines(q / f"ml40_{part}{kind}.txt")
                names = [t for x in got for t in x.split() if t.startswith(("t1-", "smoke-t1-"))]
                assert names == [("smoke-" if kind else "") + n for n in runs], names
                for x in got:
                    t = x.split()
                    assert "object_position_std" not in x and "object_orientation_threshold" not in x, x
                    assert (" V2D_OBJ_ORI_WEIGHT=1.0 V2D_OBJ_ORI_STD=0.4 " in f" {x} ") == r3o, x
                    i = next(k for k, v in enumerate(t) if v.isdigit() and k > 0 and t[k - 1] == "2048")
                    over = [v for v in t[i + 1:] if v != "scene.support=false"]
                    assert over == extra, (part, over)
        r = gen(root, "t1zs2")
        assert r.returncode == 0, r.stdout + r.stderr
        for kind in ("", "_smoke"):
            got = lines(q / f"mf4090_t1zs2{kind}.txt")
            names = [t for x in got for t in x.split() if t.startswith(("t1-", "smoke-t1-"))]
            assert names == [("smoke-" if kind else "") + f"t1-ep{e}-w2048-cap4m-r3o-mf4090" for e in (26, 12)], names
            for x in got:
                assert " V2D_OBJ_ORI_WEIGHT=1.0 V2D_OBJ_ORI_STD=0.4 " in f" {x} ", x
                assert "objective.object_position_std=0.05" in x and "termination.object_orientation_threshold=0.6" in x, x
                assert "training.replay.capacity=4000000" in x and x.count("capacity") == 1, x


def test_t1zw_r3o_10m():
    """计划 agent-doc/plan-t1r3o-2026-10-31.md 第 2 版：t1zw、t1zw2 在 ml40 用 1000 万（不写容量覆盖），t1zt4 在 mf4090 只含 ep7、22；
    三段都是 r3o（变体 A 两项 + 两个物体朝向变量），run 名与段对应，冒烟与正式一致。"""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        setup(root)
        q = root / "stages" / "policy" / "queues"
        want = {"t1zw": ("ml40", ["t1-ep19-w2048-cap10m-r3o-ml40", "t1-ep13-w2048-cap10m-r3o-ml40"], False),
                "t1zw2": ("ml40", ["t1-ep3-w2048-cap10m-r3o-ml40", "t1-ep16-w2048-cap10m-r3o-ml40"], False),
                "t1zt4": ("mf4090", ["t1-ep7-w2048-cap4m-r3o-mf4090", "t1-ep22-w2048-cap4m-r3o-mf4090"], True)}
        for part, (m, runs, cap4m) in want.items():
            r = gen(root, part)
            assert r.returncode == 0, r.stdout + r.stderr
            for kind in ("", "_smoke"):
                got = lines(q / f"{m}_{part}{kind}.txt")
                names = [t for x in got for t in x.split() if t.startswith(("t1-", "smoke-t1-"))]
                assert names == [("smoke-" if kind else "") + n for n in runs], names
                for x in got:
                    assert " V2D_OBJ_ORI_WEIGHT=1.0 V2D_OBJ_ORI_STD=0.4 " in f" {x} ", x
                    assert "objective.object_position_std=0.05" in x and "termination.object_orientation_threshold=0.6" in x, x
                    assert ("training.replay.capacity=4000000" in x) == cap4m and x.count("capacity") <= 1, x



def test_t1night_fc20():
    """计划 agent-doc/plan-t1night-2026-11-01.md：t1hg 是 ml40 上 ep4、9 的基础配方加抓握闭合项权重 20（1000 万）；t1hf2 是 mf4090 上 ep1、4 的同一改动（400 万），
    t1hf2_s43 --episodes 9 --seed 43 得 ep9 seed 43；t1hr_s43、t1hr2_s43 得 ep17、ep26 朝向项的 seed 43。都不带变体 A，冒烟与正式一致。"""
    fc = "objective.force_closure_weight=20.0"
    s43 = ["reset.seed=43", "training.seed=43"]
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        setup(root)
        q = root / "stages" / "policy" / "queues"
        want = {("t1hg", ()): ("ml40", [fc], False, ["t1-ep4-w2048-cap10m-fc20-ml40", "t1-ep9-w2048-cap10m-fc20-ml40"]),
                ("t1hf2", ()): ("mf4090", ["training.replay.capacity=4000000", fc], False,
                                ["t1-ep1-w2048-cap4m-fc20-mf4090", "t1-ep4-w2048-cap4m-fc20-mf4090"]),
                ("t1hf2_s43", ("--episodes", "9", "--seed", "43")): ("mf4090", ["training.replay.capacity=4000000", *s43, fc], False,
                                                                       ["t1-ep9-w2048-cap4m-fc20-s43-mf4090"]),
                ("t1hg_s43", ("--episodes", "4", "--seed", "43")): ("ml40", [*s43, fc], False, ["t1-ep4-w2048-cap10m-fc20-s43-ml40"]),
                ("t1hr_s43", ("--episodes", "17", "--seed", "43")): ("ml40", s43, True, ["t1-ep17-w2048-cap10m-r3o-nv-s43-ml40"]),
                ("t1hr2_s43", ("--episodes", "26", "--seed", "43")): ("ml40", s43, True, ["t1-ep26-w2048-cap10m-r3o-nv-s43-ml40"])}
        for (part, args), (mach, extra, r3o, runs) in want.items():
            r = gen(root, part, *args)
            assert r.returncode == 0, r.stdout + r.stderr
            for kind in ("", "_smoke"):
                got = lines(q / f"{mach}_{part}{kind}.txt")
                names = [t for x in got for t in x.split() if t.startswith(("t1-", "smoke-t1-"))]
                assert names == [("smoke-" if kind else "") + n for n in runs], names
                for x in got:
                    t = x.split()
                    assert "object_position_std" not in x and "object_orientation_threshold" not in x, x
                    assert (" V2D_OBJ_ORI_WEIGHT=1.0 V2D_OBJ_ORI_STD=0.4 " in f" {x} ") == r3o, x
                    i = next(k for k, v in enumerate(t) if v.isdigit() and k > 0 and t[k - 1] == "2048")
                    over = [v for v in t[i + 1:] if v != "scene.support=false" and not v.startswith("training.replay.capacity=") or v == "training.replay.capacity=4000000"]
                    assert sorted(over) == sorted(extra), (part, over)
        r = gen(root, "t1hf2", "--episodes", "9")
        assert r.returncode != 0, "预设段不许再给 --episodes"


def test_t1ep1():
    """计划 agent-doc/plan-t1ep1-2026-11-02.md：t1hg2 --episodes 1 是 ml40 上 ep1 的基础配方加抓握闭合项权重 20（1000 万），再加 --seed 43 得 seed 43；
    t1hp2、t1hp3 是 mf4090 上 ep1 的基础配方加手掌位置终止 0.05、0.08（400 万）。都不带变体 A，冒烟与正式一致；预设段 t1hg 仍只有 ep4、9。"""
    fc = "objective.force_closure_weight=20.0"
    s43 = ["reset.seed=43", "training.seed=43"]
    cap = "training.replay.capacity=4000000"
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        setup(root)
        q = root / "stages" / "policy" / "queues"
        want = {("t1hg2", ("--episodes", "1")): ("ml40", [fc], ["t1-ep1-w2048-cap10m-fc20-ml40"]),
                ("t1hg2_s43", ("--episodes", "1", "--seed", "43")): ("ml40", [*s43, fc], ["t1-ep1-w2048-cap10m-fc20-s43-ml40"]),
                ("t1hp2", ()): ("mf4090", [cap, "termination.palm_position_threshold=0.05"], ["t1-ep1-w2048-cap4m-pt05-mf4090"]),
                ("t1hp3", ()): ("mf4090", [cap, "termination.palm_position_threshold=0.08"], ["t1-ep1-w2048-cap4m-pt08-mf4090"]),
                ("t1hg", ()): ("ml40", [fc], ["t1-ep4-w2048-cap10m-fc20-ml40", "t1-ep9-w2048-cap10m-fc20-ml40"])}
        for (part, args), (mach, extra, runs) in want.items():
            r = gen(root, part, *args)
            assert r.returncode == 0, r.stdout + r.stderr
            for kind in ("", "_smoke"):
                got = lines(q / f"{mach}_{part}{kind}.txt")
                names = [t for x in got for t in x.split() if t.startswith(("t1-", "smoke-t1-"))]
                assert names == [("smoke-" if kind else "") + n for n in runs], names
                for x in got:
                    t = x.split()
                    assert "object_position_std" not in x and "V2D_OBJ_ORI_WEIGHT" not in x, x
                    i = next(k for k, v in enumerate(t) if v.isdigit() and k > 0 and t[k - 1] == "2048")
                    over = [v for v in t[i + 1:] if v != "scene.support=false" and not v.startswith("training.replay.capacity=") or v == cap]
                    assert sorted(over) == sorted(extra), (part, over)

def test_t2auc2():
    """计划 agent-doc/plan-t2auc2-2026-11-02.md：t2zo、t2zo2 是 mf4090 上 Tier 2 ep10、0 与 ep26、21 的 r3o（400 万），
    t2o3、t2o3b 是 ml40 上同 4 集的 r3o（1000 万）；行里带 Tier 2 参考与物体朝向项，不出现 Tier 1 的参考。"""
    cap = "training.replay.capacity=4000000"
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        setup(root)
        q = root / "stages" / "policy" / "queues"
        want = {"t2zo": ("mf4090", ["t2-ep10-w2048-cap4m-r3o-mf4090", "t2-ep0-w2048-cap4m-r3o-mf4090"]),
                "t2zo2": ("mf4090", ["t2-ep26-w2048-cap4m-r3o-mf4090", "t2-ep21-w2048-cap4m-r3o-mf4090"]),
                "t2o3": ("ml40", ["t2-ep10-w2048-cap10m-r3o-ml40", "t2-ep0-w2048-cap10m-r3o-ml40"]),
                "t2o3b": ("ml40", ["t2-ep26-w2048-cap10m-r3o-ml40", "t2-ep21-w2048-cap10m-r3o-ml40"])}
        for part, (mach, runs) in want.items():
            r = gen(root, part)
            assert r.returncode == 0, r.stdout + r.stderr
            for kind in ("", "_smoke"):
                got = lines(q / f"{mach}_{part}{kind}.txt")
                names = [t for x in got for t in x.split() if t.startswith(("t2-", "smoke-t2-"))]
                assert names == [("smoke-" if kind else "") + n for n in runs], names
                for x in got:
                    assert "V2D_OBJ_ORI_WEIGHT=1.0" in x and "/motion/tier2/" in x and "/motion/tier1" not in x, x
                    assert (cap in x) == (mach == "mf4090"), x

def test_t2auc3():
    """计划 agent-doc/plan-t2auc3-2026-11-02.md：t2l3、t2l3b 是第三台 l40n 上 Tier 2 ep18、6 与 ep15、8 的 r3o（1000 万）；
    文件名与 run 名都带 l40n，行里带 Tier 2 参考与物体朝向项，不带 400 万 buffer，不出现 Tier 1 的参考。"""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        setup(root)
        q = root / "stages" / "policy" / "queues"
        want = {"t2l3": ["t2-ep18-w2048-cap10m-r3o-l40n", "t2-ep6-w2048-cap10m-r3o-l40n"],
                "t2l3b": ["t2-ep15-w2048-cap10m-r3o-l40n", "t2-ep8-w2048-cap10m-r3o-l40n"]}
        for part, runs in want.items():
            r = gen(root, part)
            assert r.returncode == 0, r.stdout + r.stderr
            for kind in ("", "_smoke"):
                got = lines(q / f"l40n_{part}{kind}.txt")
                names = [t for x in got for t in x.split() if t.startswith(("t2-", "smoke-t2-"))]
                assert names == [("smoke-" if kind else "") + n for n in runs], names
                for x in got:
                    assert "V2D_OBJ_ORI_WEIGHT=1.0" in x and "/motion/tier2/" in x and "/motion/tier1" not in x, x
                    assert "training.replay.capacity=4000000" not in x, x

def test_final1():
    """计划 agent-doc/plan-final-2026-11-03.md：t2l4、t2l4b、t2l4c 是 l40n 上 Tier 2 ep5、22，ep16、20，ep29、27 的 r3o（1000 万）；
    规则同 test_t2auc3。"""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        setup(root)
        q = root / "stages" / "policy" / "queues"
        want = {"t2l4": [5, 22], "t2l4b": [16, 20], "t2l4c": [29, 27]}
        for part, eps in want.items():
            runs = [f"t2-ep{e}-w2048-cap10m-r3o-l40n" for e in eps]
            r = gen(root, part)
            assert r.returncode == 0, r.stdout + r.stderr
            for kind in ("", "_smoke"):
                got = lines(q / f"l40n_{part}{kind}.txt")
                names = [t for x in got for t in x.split() if t.startswith(("t2-", "smoke-t2-"))]
                assert names == [("smoke-" if kind else "") + n for n in runs], names
                for x in got:
                    assert "V2D_OBJ_ORI_WEIGHT=1.0" in x and "/motion/tier2/" in x and "/motion/tier1" not in x, x
                    assert "training.replay.capacity=4000000" not in x, x

if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("ALL_OK")
