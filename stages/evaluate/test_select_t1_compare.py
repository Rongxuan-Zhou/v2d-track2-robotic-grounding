"""select_t1_compare.py 的测试：守 agent-doc/plan-t1-r3x-2026-10-08.md 第 4 节的选择门 C1–C5。只用标准库。
造的数据：上一次 30 条 episode 各一个候选 r<ep>（ep7 另有 r7b），选中 r<ep>；这一次加 ep0 的两个新候选 n0a（成功）、n0b（失败），选中 n0a。
用法：python3 stages/evaluate/test_select_t1_compare.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
METRICS = ("add_auc", "spider_sr", "maniptrans_sr", "mppe_cm")


def cand(run, ep, mp, sp, auc, mppe, w=None):
    return {"run": run, "episode": ep, "add_auc": auc, "spider_sr": sp, "maniptrans_sr": mp, "mppe_cm": mppe, "mp_sr_4096": w}


def setup(d: Path, drift=False, move=False, unknown=False, worse=False, bad_pack=False):
    old = {f"r{ep}": cand(f"r{ep}", ep, 0.0, 1.0, 0.3 + ep / 100, 5.0 + ep, 0.1) for ep in range(30)}
    old["r7b"] = cand("r7b", 7, 0.0, 0.0, 0.1, 40.0, 0.0)
    new = json.loads(json.dumps(old))
    new["n0a"] = cand("n0a", 0, 1.0, 1.0, 0.8, 1.5, 0.9)
    new["n0b"] = cand("n0b", 0, 0.0, 0.0, 0.05, 60.0, 0.2)
    if drift:
        new["r3"]["add_auc"] += 1e-6
    if unknown:
        new["zz"] = cand("zz", 9, 0.0, 0.0, 0.1, 30.0)
    old_sel = {ep: f"r{ep}" for ep in range(30)}
    new_sel = {**old_sel, 0: "n0b" if worse else "n0a"}
    if move:
        new_sel[7] = "r7b"
    (d / "old.json").write_text(json.dumps(old))
    (d / "new.json").write_text(json.dumps(new))
    for name, sel in (("old.tsv", old_sel), ("new.tsv", new_sel)):
        (d / name).write_text("".join(f"1\t{ep}\t{r}\t349999104\tmf4090\n" for ep, r in sorted(sel.items())))
    (d / "list.tsv").write_text("# 清单\n1\t0\tn0a\t349999104\n1\t0\tn0b\t349999104\n1\t15\tn15a\t349999104\n")
    mean = {m: sum(new[r][m] for r in new_sel.values()) / 30 for m in METRICS}
    if bad_pack:
        mean["maniptrans_sr"] += 0.01
    (d / "select.txt").write_text("# command : x\nCHOSEN ...\nPACKED " + " ".join(f"{m}={mean[m]:.6f}" for m in METRICS) + "\nSELECT_T1_DONE\n")


def run(d: Path):
    r = subprocess.run([sys.executable, str(HERE / "select_t1_compare.py"), "--old-scores", str(d / "old.json"), "--old-tsv", str(d / "old.tsv"),
                        "--new-scores", str(d / "new.json"), "--new-tsv", str(d / "new.tsv"), "--new-list", str(d / "list.tsv"),
                        "--select-log", str(d / "select.txt")], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def check(expect_ok, tag=None, **kw):
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        setup(d, **kw)
        rc, out = run(d)
        last = out.strip().splitlines()[-1]
        if expect_ok:
            assert rc == 0 and last == "COMPARE_OK", out
            assert "清单里没有分数的（只报）['n15a']" in out and "ep0：上一次 r0，这一次 n0a（换了）" in out, out
        else:
            assert rc == 1 and last.startswith("COMPARE_FAIL") and tag in last, out


def test_ok():
    check(True)


def test_old_score_drift_fails():
    check(False, "C1", drift=True)


def test_unknown_candidate_fails():
    check(False, "C2", unknown=True)


def test_other_episode_moved_fails():
    check(False, "C3", move=True)


def test_worse_choice_fails():
    check(False, "C4 ep0", worse=True)


def test_packed_mismatch_fails():
    check(False, "C5", bad_pack=True)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("ALL_OK")
