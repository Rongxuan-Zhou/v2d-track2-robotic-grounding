"""select_t2.py 的测试：守 agent-doc/plan-t2-select-b-2026-10-12.md 第 2 节第 2 步与第 3 节（乙的三层平手、候选资格、现在交的 run 必须有资格）。
只用标准库。临时目录里造 30 条现在交的 run、几张结果表与评测记录：
  ep0 新候选 MP-SR 更高 → 换；ep1 MP-SR 相同、ADD-AUC 更高 → 换；ep2 两项都相同 → 留现在交的；ep3 两条新候选都比现在交的好且相同 → 按 run 名；
  ep4 更好的候选状态是 no_eval、ep5 更好的候选没有评测记录、ep6 更好的候选在两张表里都是 ok、ep7 更好的候选的评测记录属于别的 run → 都没有资格，留现在交的；
  现在交的 run 没有资格 → 报错退出，不写选择清单。
用法：python3 stages/evaluate/test_select_t2.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEP = "349999104"


def ev(d: Path, run: str, mp: float, auc: float, ck_run=None):
    j = {"checkpoint": f"/repo/runs/flash_chord/{ck_run or run}/policy_{STEP}.safetensors", "environment_steps": int(STEP),
         "maniptrans_sr": mp, "add_auc": auc}
    (d / f"eval_{run}_s{STEP}.txt").write_text("前面的输出\nsummary_json " + json.dumps(j) + "\n")


def row(run, ep, status="ok"):
    return "\t".join(["2", str(ep), run, STEP, status, f"runs/x/{run}/episode_{ep:06d}.json", f"runs/x/{run}/episode_{ep:06d}.parquet", "a" * 64, "b" * 64])


def setup(root: Path, current_bad=False):
    ex, tb, evd = root / "ex", root / "tables", root / "evals"
    for d in (ex, tb, evd):
        d.mkdir()
    cur = [f"t2-ep{e}-w2048-cap10m-ml40" for e in range(30)]
    (ex / "ml40.tsv").write_text("\n".join(f"2\t{e}\t{r}\t{STEP}" for e, r in enumerate(cur)) + "\n")
    t1 = ["tier\tepisode\trun\tstep\tstatus\tjson\tparquet\tjson_sha256\tparquet_sha256"] + [row(r, e) for e, r in enumerate(cur) if not (current_bad and e == 9)]
    for e, r in enumerate(cur):
        ev(evd, r, 0.10, 0.30)
    new = {0: [("t2-ep0-new-ml40", 0.90, 0.10)], 1: [("t2-ep1-new-ml40", 0.10, 0.50)], 2: [("t2-ep2-new-ml40", 0.10, 0.30)],
           3: [("t2-ep3-b-ml40", 0.50, 0.40), ("t2-ep3-a-ml40", 0.50, 0.40)], 4: [("t2-ep4-new-ml40", 0.99, 0.90)],
           5: [("t2-ep5-new-ml40", 0.99, 0.90)], 6: [("t2-ep6-new-ml40", 0.99, 0.90)], 7: [("t2-ep7-new-ml40", 0.99, 0.90)]}
    t2 = ["tier\tepisode\trun\tstep\tstatus\tjson\tparquet\tjson_sha256\tparquet_sha256"]
    for e, items in new.items():
        for r, mp, auc in items:
            t2.append(row(r, e, "no_eval" if e == 4 else "ok"))
            if e != 5:
                ev(evd, r, mp, auc, ck_run="t2-ep7-other-ml40" if e == 7 else None)
    t3 = ["tier\tepisode\trun\tstep\tstatus\tjson\tparquet\tjson_sha256\tparquet_sha256", row("t2-ep6-new-ml40", 6)]
    (tb / "export2_ml40.tsv").write_text("\n".join(t1) + "\n")
    (tb / "cand_a_ml40.tsv").write_text("\n".join(t2) + "\n")
    (tb / "cand_b_ml40.tsv").write_text("\n".join(t3) + "\n")
    return ex, tb, evd


def run(root: Path, ex, tb, evd):
    out = root / "sel.tsv"
    r = subprocess.run([sys.executable, str(HERE / "select_t2.py"), "--current", str(ex / "ml40.tsv"), "--tables", *map(str, sorted(tb.glob("*.tsv"))),
                        "--evals", str(evd), "--out-tsv", str(out)], capture_output=True, text=True)
    return r, out


def test_rule_and_eligibility():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        r, out = run(root, *setup(root))
        assert r.returncode == 0, r.stdout + r.stderr
        got = {int(c.split("\t")[1]): c.split("\t")[2] for c in out.read_text().splitlines()}
        assert len(got) == 30
        assert got[0] == "t2-ep0-new-ml40" and got[1] == "t2-ep1-new-ml40", got
        assert got[2] == "t2-ep2-w2048-cap10m-ml40" and got[3] == "t2-ep3-a-ml40", got
        for e in (4, 5, 6, 7):
            assert got[e] == f"t2-ep{e}-w2048-cap10m-ml40", (e, got[e])
        assert all(c.split("\t")[0] == "2" and c.split("\t")[3] == STEP and c.split("\t")[4] == "ml40" for c in out.read_text().splitlines())
        s = r.stdout
        assert "SELECT_T2_DONE" in s and "换了的 episode 3 条" in s, s
        assert "t2-ep5-new-ml40：没有评测记录" in s and "t2-ep6-new-ml40：状态 ok 的行有 2 行" in s and "t2-ep7-new-ml40：" in s, s
        assert "t2-ep4-new-ml40" not in out.read_text()


def test_current_ineligible_refuses():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        r, out = run(root, *setup(root, current_bad=True))
        assert r.returncode != 0 and "ep9 现在交的" in r.stdout + r.stderr, r.stdout + r.stderr
        assert not out.exists()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("ALL_OK")
