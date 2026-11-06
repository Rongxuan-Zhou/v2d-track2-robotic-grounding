#!/usr/bin/env python3
"""derive_r3o_rows.py 的测试（只用标准库，在临时目录里跑，不写仓库）。计划 agent-doc/plan-t3auc1-2026-11-03.md 第 1 节：
每行恰好多两个变量、其余字段与源行逐字相同（run 名、正式行标签、+10m 的 buffer 覆盖除外）、r3o_lines.awk 过；
反例：源里没有的 run、+10m 用在 1000 万的源上、目标已存在，都要拒绝。用法：python3 stages/policy/test_derive_r3o_rows.py   最后一行 ALL_OK"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "stages/policy/derive_r3o_rows.py"
AWK = REPO / "stages/policy/r3o_lines.awk"
ORI = ["V2D_OBJ_ORI_WEIGHT=1.0", "V2D_OBJ_ORI_STD=0.4"]
SRC = {"ml40_t3qb2n12": "t3-ep12-w2048-cap10m-r3a-gb5b2n12hdyn7-ml40", "mf4090_t3pg4": "t3-ep0-w2048-cap4m-r3a-gb4-mf4090"}


def run(root, *args):
    return subprocess.run([sys.executable, str(SCRIPT), "--root", str(root), *args], capture_output=True, text=True)


def body(p):
    return [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln and not ln.startswith("#")]


def src_line(base, tok):
    return [ln for ln in body(REPO / f"stages/policy/queues/{base}") if tok in ln.split()][0]


def check_same(new, old, old_run, new_run, seg, drop_cap):
    a, b = new.split(" "), old.split(" ")
    assert a.count(ORI[0]) == 1 and a.count(ORI[1]) == 1, new
    a = [t for t in a if t not in ORI]
    if drop_cap:
        b = [t for t in b if t != "training.replay.capacity=4000000"]
    assert len(a) == len(b), (a, b)
    for x, y in zip(a, b):
        if y in (old_run, f"smoke-{old_run}"):
            assert x == y.replace(old_run, new_run), (x, y)
        elif y.startswith("V2D_WANDB_TAG=") and y != "V2D_WANDB_TAG=smoke":
            assert x == f"V2D_WANDB_TAG={seg}", x
        else:
            assert x == y, (x, y)


def awk_ok(p):
    out = subprocess.run(["awk", "-v", "oo=1", "-f", str(AWK)], input="\n".join(body(p)) + "\n", capture_output=True, text=True).stdout
    return out.strip().splitlines()[-1] == "0"


def main():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "stages/policy/queues").mkdir(parents=True)
        (root / "stages/evaluate/exports").mkdir(parents=True)
        for base in SRC:
            for suf in (".txt", "_smoke.txt", ".expect.tsv"):
                shutil.copy(REPO / f"stages/policy/queues/{base}{suf}", root / f"stages/policy/queues/{base}{suf}")
        r = run(root, "--seg", "t3qr3oX", "--machine", "ml40", f"ml40_t3qb2n12:{SRC['ml40_t3qb2n12']}",
                f"mf4090_t3pg4:{SRC['mf4090_t3pg4']}+10m")
        assert r.stdout.strip().endswith("DERIVE_R3O_OK t3qr3oX 2"), r.stdout + r.stderr
        q = root / "stages/policy/queues"
        new12, new0 = "t3-ep12-w2048-cap10m-r3o-gb5b2n12hdyn7-ml40", "t3-ep0-w2048-cap10m-r3o-gb4-ml40"
        f, s = body(q / "ml40_t3qr3oX.txt"), body(q / "ml40_t3qr3oX_smoke.txt")
        check_same(f[0], src_line("ml40_t3qb2n12.txt", SRC["ml40_t3qb2n12"]), SRC["ml40_t3qb2n12"], new12, "t3qr3oX", False)
        check_same(f[1], src_line("mf4090_t3pg4.txt", SRC["mf4090_t3pg4"]), SRC["mf4090_t3pg4"], new0, "t3qr3oX", True)
        check_same(s[0], src_line("ml40_t3qb2n12_smoke.txt", "smoke-" + SRC["ml40_t3qb2n12"]), SRC["ml40_t3qb2n12"], new12, "t3qr3oX", False)
        check_same(s[1], src_line("mf4090_t3pg4_smoke.txt", "smoke-" + SRC["mf4090_t3pg4"]), SRC["mf4090_t3pg4"], new0, "t3qr3oX", True)
        assert "training.replay.capacity" not in f[1] and "training.replay.capacity" not in s[1]
        assert awk_ok(q / "ml40_t3qr3oX.txt") and awk_ok(q / "ml40_t3qr3oX_smoke.txt")
        e = body(q / "ml40_t3qr3oX.expect.tsv")
        assert e[1].split("\t")[:2] == [new12, f"smoke-{new12}"] and e[1].split("\t")[8] == "10000000", e[1]
        assert e[2].split("\t")[:2] == [new0, f"smoke-{new0}"] and e[2].split("\t")[8] == "10000000", e[2]
        c = body(root / "stages/evaluate/exports/cand_t3_t3qr3oX_ml40.tsv")
        assert c == [f"3\t12\t{new12}\t349999104", f"3\t0\t{new0}\t349999104"], c
        print("POS_OK")
        r = run(root, "--seg", "t3qr3oX", "--machine", "ml40", f"ml40_t3qb2n12:{SRC['ml40_t3qb2n12']}")
        assert "已存在" in r.stdout and r.returncode != 0, r.stdout
        r = run(root, "--seg", "t3qr3oY", "--machine", "ml40", "ml40_t3qb2n12:t3-ep99-w2048-cap10m-r3a-x-ml40")
        assert "DERIVE_R3O_FAIL" in r.stdout and r.returncode != 0, r.stdout
        r = run(root, "--seg", "t3qr3oZ", "--machine", "ml40", f"ml40_t3qb2n12:{SRC['ml40_t3qb2n12']}+10m")
        assert "+10m 只许 400 万的源" in r.stdout and r.returncode != 0, r.stdout
        r = run(root, "--seg", "t3pr3oW", "--machine", "mf4090", f"ml40_t3qb2n12:{SRC['ml40_t3qb2n12']}")
        assert "机器名不是" in r.stdout and r.returncode != 0, r.stdout
        assert not (q / "mf4090_t3pr3oW.txt").exists() and not (q / "ml40_t3qr3oY.txt").exists()
        print("NEG_OK")
    print("ALL_OK")


if __name__ == "__main__":
    main()
