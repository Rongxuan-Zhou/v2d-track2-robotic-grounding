"""ckpt_sweep_pick.py、seed_gate.py、dump_knock.py、ckpt_sweep.sh 的测试：守 agent-doc/plan-ep25-sweep-seeds-2026-10-18.md 第 2 节（合格评测的定义、
选法“最低值 + 1 cm 内取步数最大”、细扫取选中那一步的前后各一份、每档选点严格低于上限才导出、链的先后、只报的逐世界 dump 与总是退出码 0）
与第 3 节（补 seed 的闸）。只用标准库。
用法：python3 stages/evaluate/test_ckpt_sweep_pick.py
"""
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PICK = HERE / "ckpt_sweep_pick.py"
GATE = HERE / "seed_gate.py"
KNOCK = HERE / "dump_knock.py"
SWEEP = HERE / "ckpt_sweep.sh"
FRAMES = 859


def ev(root: Path, run: str, step: int, mppe: float, nf=None, frames=FRAMES, ck_run=None, auc=0.1, sp=0.0, mp=0.0, worlds=4096):
    d = root / "runs/flash_chord" / run
    d.mkdir(parents=True, exist_ok=True)
    j = {"checkpoint": f"/repo/runs/flash_chord/{ck_run or run}/policy_{step}.safetensors", "world_count": worlds, "step_count": frames,
         "non_finite_worlds": [] if nf is None else nf,
         "metrics": {"add_auc": auc, "spider_sr_uncentered": sp, "maniptrans_sr": mp, "mppe_cm": mppe}}
    (d / f"eval_{step}_w4096.json").write_text(json.dumps(j))


def pol(root: Path, run: str, steps):
    d = root / "runs/flash_chord" / run
    d.mkdir(parents=True, exist_ok=True)
    (d / "DONE").write_text("")
    for s in steps:
        (d / f"policy_{s}.safetensors").write_text("x")


def table(root: Path, rows, name="coarse.tsv") -> Path:
    p = root / name
    p.write_text("# 注释\n" + "".join(f"eval\t{r}\t{s}\tframe\t{FRAMES}\n" for r, s in rows))
    return p


def run_pick(root: Path, *args):
    return subprocess.run([sys.executable, str(PICK), *map(str, args)], cwd=root, capture_output=True, text=True)


def test_refine_and_final():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        a, b, c, d, f = "t1-ep25-a-ml40", "t1-ep25-b-ml40", "t2-ep25-c-ml40", "t2-ep25-d-ml40", "t2-ep25-f-ml40"
        for r in (a, b, c, d, f):
            pol(root, r, [10, 20, 30, 40, 50])
        ev(root, a, 20, 80.0)
        ev(root, a, 40, 60.0)                     # a：最低在 40 → 补 30、50
        ev(root, b, 10, 10.0, nf=[3])             # 有非有限世界 → 不合格
        ev(root, b, 30, 70.0)                     # b：合格的只有 30 → 补 20、40
        ev(root, b, 20, 5.0, frames=858)          # 帧数不对 → 不合格（20 仍要补）
        ev(root, c, 10, 50.0)                     # c：最低在第一份 → 只补 20
        ev(root, c, 30, 90.0)
        ev(root, d, 40, 1.0, ck_run=a)            # checkpoint 属于别的 run → 不合格；d 没有合格评测 → 不补
        ev(root, d, 50, 1.0, worlds=1)            # 世界数不对 → 不合格
        ev(root, f, 10, 49.6)                     # f：30 比最低的 10 只高 0.7 cm（1 cm 内）→ 取步数大的 30 → 补 20、40
        ev(root, f, 30, 50.3)
        co = table(root, [(a, 20), (a, 40), (b, 30), (c, 10), (c, 30), (d, 40), (f, 10), (f, 30)])
        r = run_pick(root, "refine", co, root / "refine.tsv")
        assert r.returncode == 0, r.stderr
        got = sorted(tuple(x.split("\t")[1:3]) for x in (root / "refine.tsv").read_text().splitlines())
        assert got == sorted([(a, "30"), (a, "50"), (b, "20"), (b, "40"), (c, "20"), (f, "20"), (f, "40")]), got
        assert r.stdout.strip().endswith("CKPT_SWEEP_PICK_DONE")
        # 细扫之后：Tier 1 a@30 55.0、a@50 55.5（1 cm 内）→ 取步数大的 a@50；Tier 2 最低 f@10 49.6，1 cm 内有 c@10 50.0、f@30 50.3 → 取 f@30
        ev(root, a, 50, 55.5)
        ev(root, a, 30, 55.0)
        ev(root, f, 50, 60.0)                     # 不在 1 cm 内，步数再大也不选
        r = run_pick(root, "final", co, root / "x.tsv", 100)
        assert r.returncode == 0, r.stderr
        assert (root / "x.tsv").read_text().splitlines() == [f"1\t25\t{a}\t50", f"2\t25\t{f}\t30"], (root / "x.tsv").read_text()
        # 上限 52：Tier 1 选中的 55.5 不低于上限 → 不写；Tier 2 的 50.3 写
        r = run_pick(root, "final", co, root / "x2.tsv", 52)
        assert (root / "x2.tsv").read_text().splitlines() == [f"2\t25\t{f}\t30"], (root / "x2.tsv").read_text()
        # 上限 50.3：两档都不写（选中那一点要严格低于上限）
        r = run_pick(root, "final", co, root / "x3.tsv", 50.3)
        assert (root / "x3.tsv").read_text() == ""


def test_bad_table():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        (root / "bad.tsv").write_text("eval\tt1-ep25-a-ml40\t10\treal\t859\n")
        assert run_pick(root, "refine", root / "bad.tsv", root / "o.tsv").returncode == 2
        (root / "bad2.tsv").write_text("eval\tt1-ep25-a-ml40\t10\tframe\t859\neval\tt1-ep25-a-ml40\t20\tframe\t860\n")
        assert run_pick(root, "refine", root / "bad2.tsv", root / "o.tsv").returncode == 2
        (root / "empty.tsv").write_text("# 只有注释\n")
        assert run_pick(root, "final", root / "empty.tsv", root / "o.tsv", 90).returncode == 2


def gate_case(root: Path, run: str, w: float, x: float, key="maniptrans_sr", export=True, nf=None):
    ev(root, run, 349999104, 10.0, nf=nf, **({"mp": w} if key == "maniptrans_sr" else {"sp": w}))
    if export:
        d = root / "runs/flash_chord" / run / "export_8c778f9/s349999104_w1"
        d.mkdir(parents=True, exist_ok=True)
        (d / "episode_000014.json").write_text(json.dumps({"checkpoint": f"/repo/runs/flash_chord/{run}/policy_349999104.safetensors",
                                                           "world_count": 1, "metrics": {key: x}}))
    return subprocess.run([sys.executable, str(GATE), run, "14", key, "0.3"], cwd=root, capture_output=True, text=True)


def test_seed_gate():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        r = gate_case(root, "t1-ep14-a-mf4090", 0.6, 0.0)
        assert r.returncode == 0 and r.stdout.strip().endswith("SEED_GATE PASS"), r.stdout + r.stderr
        r = gate_case(root, "t1-ep14-b-mf4090", 0.6, 1.0)
        assert r.returncode == 1 and "导出已成功" in r.stdout, r.stdout
        r = gate_case(root, "t1-ep14-c-mf4090", 0.2, 0.0)
        assert r.returncode == 1 and "4096 世界太低" in r.stdout, r.stdout
        r = gate_case(root, "t1-ep14-d-mf4090", 0.6, 0.0, export=False)
        assert r.returncode == 1 and "读不到" in r.stdout, r.stdout
        r = gate_case(root, "t1-ep14-e-mf4090", 0.6, 0.0, nf=[1])
        assert r.returncode == 1 and "不合格" in r.stdout, r.stdout
        r = gate_case(root, "t1-ep14-f-mf4090", 0.5, 0.0, key="spider_sr_uncentered")
        assert r.returncode == 0, r.stdout
        r = subprocess.run([sys.executable, str(GATE), "x", "14", "add_auc", "0.3"], cwd=root, capture_output=True, text=True)
        assert r.returncode == 2


def npy(descr: str, fmt: str, values) -> bytes:
    head = repr({"descr": descr, "fortran_order": False, "shape": (len(values),)})
    head += " " * ((64 - (10 + len(head) + 1) % 64) % 64) + "\n"
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(head)) + head.encode("latin1") + struct.pack("<" + fmt * len(values), *values)


def write_dump(path: Path, pos, sp, mp, pos_descr=("<f8", "d")):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("mp_position_m.npy", npy(pos_descr[0], pos_descr[1], pos))
        z.writestr("sp_success.npy", npy("|b1", "?", sp))
        z.writestr("mp_success.npy", npy("|b1", "?", mp))
        z.writestr("tracking_error.npy", npy("<f8", "d", [0.0]))


def test_dump_knock():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        pos = [0.45, 0.46, 1.10, 1.20, 0.44, 0.90, 0.47, 0.45, 0.30, 1.05]   # 超过 0.8 m 的 4 个
        for descr in (("<f8", "d"), ("<f4", "f")):
            write_dump(root / "d.npz", pos, [False] * 9 + [True], [False] * 10, descr)
            r = subprocess.run([sys.executable, str(KNOCK), str(root / "d.npz")], capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
            assert "平均位置误差超过 0.8 m 的世界 0.4000；SP 成功 0.1000，MP 成功 0.0000" in r.stdout, r.stdout
            assert r.stdout.strip().endswith("DUMP_KNOCK_DONE")
        r = subprocess.run([sys.executable, str(KNOCK), str(root / "d.npz"), "1.15"], capture_output=True, text=True)
        assert "超过 1.15 m 的世界 0.1000" in r.stdout, r.stdout


FAKE_EVAL = r"""#!/usr/bin/env bash
# 假的 day2_eval_chain.sh：给附加表的每一行写评测 JSON（MPPE 按步数查表：10→80、20→90、30→100、40→5、50→120），日志最后一行同真链
EXTRA=$2; NAME=$3; echo "$*" >> runs/calls.txt
n=0
while IFS=$'\t' read -r kind run step pb src; do
  [ "$kind" = eval ] || continue
  m=$(python3 -c "import sys; print({10: 80, 20: 90, 30: 100, 40: 5, 50: 120}[int(sys.argv[1])])" "$step")
  python3 -c "import json,sys; json.dump({'checkpoint': f'/repo/runs/flash_chord/{sys.argv[1]}/policy_{sys.argv[2]}.safetensors', 'world_count': 4096, 'step_count': 859, 'non_finite_worlds': [], 'metrics': {'add_auc': 0.1, 'spider_sr_uncentered': 0.0, 'maniptrans_sr': 0.0, 'mppe_cm': float(sys.argv[3])}}, open(f'runs/flash_chord/{sys.argv[1]}/eval_{sys.argv[2]}_w4096.json', 'w'))" "$run" "$step" "$m"
  n=$((n + 1))
done < <(grep -v '^#' "$EXTRA")
echo "=== 2026-10-18 06:40:00 CST DAY2_EVAL_DONE（成功 ${n}、失败 0）" >> "runs/logs/$NAME.log"
"""
FAKE_EXPORT = r"""#!/usr/bin/env bash
echo "export $* FALLBACK=${V2D_FALLBACK_MAX:-}" >> runs/calls.txt
echo "=== x EXPORT_CHAIN_DONE（成功 $(grep -c . "$1")、失败 0）" >> "runs/logs/$2.log"
"""
FAKE_RECORD = r"""#!/usr/bin/env bash
N=$1; shift; mkdir -p "$(dirname "results/$N.txt")"; "$@" > "results/$N.txt" 2>&1
"""
FAKE_DUMP = r"""#!/usr/bin/env bash
# 假的 fc_eval_dump.sh：写一份 3 个世界的 dump（一个世界超过 0.8 m）
echo "dump $* SRC=${V2D_FC_SRC:+set}" >> runs/calls.txt
python3 -c "import sys; sys.path.insert(0, sys.argv[3]); import test_ckpt_sweep_pick as t; from pathlib import Path; t.write_dump(Path(f'runs/flash_chord/{sys.argv[1]}/eval_dump/eval_{sys.argv[2]}_w4096.npz'), [0.4, 0.5, 1.1], [False] * 3, [False] * 3)" "$1" "$2" "$V2D_TEST_DIR"
"""


def test_sweep_chain():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        for sub in ("stages/evaluate", "scripts/dev", "runs/logs", "src"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        shutil.copy(SWEEP, root / "stages/evaluate/ckpt_sweep.sh")
        shutil.copy(PICK, root / "stages/evaluate/ckpt_sweep_pick.py")
        shutil.copy(KNOCK, root / "stages/evaluate/dump_knock.py")
        for name, body in (("stages/evaluate/day2_eval_chain.sh", FAKE_EVAL), ("stages/evaluate/export_chain.sh", FAKE_EXPORT),
                           ("scripts/dev/record", FAKE_RECORD), ("stages/evaluate/fc_eval_dump.sh", FAKE_DUMP)):
            (root / name).write_text(body)
            os.chmod(root / name, 0o755)
        (root / "src/SOURCE_V2D.txt").write_text("8c778f905c test\n")
        run = "t2-ep25-z-mf4090"
        pol(root, run, [10, 20, 30, 40, 50])
        co = table(root, [(run, 20), (run, 50)])
        env = dict(os.environ, V2D_FC_SRC=str(root / "src"), V2D_TEST_DIR=str(HERE))
        r = subprocess.run(["bash", "stages/evaluate/ckpt_sweep.sh", "sw", str(co), "90"], cwd=root, env=dict(env, V2D_SWEEP_DRYRUN="1"),
                           capture_output=True, text=True)
        assert r.returncode == 0 and r.stdout.strip().endswith("CKPT_SWEEP_DRYRUN_OK"), r.stdout + r.stderr
        assert not (root / "runs/logs/sw.log").exists()
        r = subprocess.run(["bash", "stages/evaluate/ckpt_sweep.sh", "sw", str(co), "90"], cwd=root, env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        log = (root / "runs/logs/sw.log").read_text()
        assert log.strip().splitlines()[-1].split(" ", 4)[-1].startswith("CKPT_SWEEP_DONE"), log
        # 粗扫 20（90）、50（120）→ 最低 20 → 细扫补 10、30；30 的 MPPE 是 100，10 的是 80 → 不再细扫；最低是 10（80）< 90 → 导出 10
        assert (root / "runs/ckpt_sweep/sw_refine.tsv").read_text().count("eval") == 2
        assert (root / "runs/ckpt_sweep/sw_export.tsv").read_text() == f"2\t25\t{run}\t10\n"
        calls = (root / "runs/calls.txt").read_text()
        assert "export runs/ckpt_sweep/sw_export.tsv sw_x FALLBACK=0" in calls, calls
        assert f"dump {run} 10 SRC=set" in calls, calls
        assert "超过 0.8 m 的世界 0.3333" in (root / "results/ckpt_sweep" / f"knock_{run}_s10.txt").read_text()
        assert "dump 1 份" in log.strip().splitlines()[-1], log
        assert not list((root / "runs").glob(".chain-active-*"))
        # 第二次起：日志已在 → 退出码 2
        r = subprocess.run(["bash", "stages/evaluate/ckpt_sweep.sh", "sw", str(co), "90"], cwd=root, env=env, capture_output=True, text=True)
        assert r.returncode == 2
        # 没有 policy 的行 → 退出码 2
        bad = table(root, [(run, 60)], "bad.tsv")
        r = subprocess.run(["bash", "stages/evaluate/ckpt_sweep.sh", "sw2", str(bad), "90"], cwd=root, env=env, capture_output=True, text=True)
        assert r.returncode == 2 and "没有 runs/flash_chord" in r.stderr, r.stderr


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("TEST_CKPT_SWEEP_PICK_OK")
