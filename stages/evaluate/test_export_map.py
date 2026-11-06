"""export_map.py 的测试（暂无契约 ID：运行规范待成文）。只用标准库。计划 agent-doc/plan-submission-exports-2026-10-02.md 第 3.1 节。
  1. 仓库里已提交的 stages/evaluate/exports/{mf4090,ml40}.tsv 与按当前已提交的结果、队列重新生成的逐字节相同；
     两档各 30 条、每条恰好一行，机器与文件名一致，步数一律 349999104，Tier 1 的 ep20 用批量那条；
  2. 假仓库：同一条 episode 两条 run、少一条 episode：报错退出，不写文件；只认 playback=frame 的判定，-rt- 的 run 不算；
  3. 输出已存在：拒绝覆盖；
  4. 带防护的重训（10/3）：已提交的清单里两档的 ep7 用 -p185g- 的 run；假仓库里同一条 episode 另有 -p185g- 的 run 时只用它，
     同一条 episode 两条 -p185g- 的 run 报错。
用法：python3 stages/evaluate/test_export_map.py
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "stages/evaluate/export_map.py"


def run(root: Path, out: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), "--root", str(root)], env=dict(os.environ, V2D_EXPORT_DIR=str(out)),
                          capture_output=True, text=True, timeout=60)


def rows(path: Path) -> list:
    return [line.split("\t") for line in path.read_text(encoding="utf-8").splitlines() if line and not line.startswith("#")]


def test_committed_matches_regenerated() -> None:
    with tempfile.TemporaryDirectory() as t:
        out = Path(t) / "x"
        r = run(REPO, out)
        assert r.returncode == 0, r.stderr
        seen = {1: [], 2: []}
        for m in ("mf4090", "ml40"):
            committed = REPO / f"stages/evaluate/exports/{m}.tsv"
            assert committed.read_bytes() == (out / f"{m}.tsv").read_bytes(), f"{m}.tsv 与重新生成的不一致"
            for tier, ep, run_name, step in rows(committed):
                assert run_name.endswith(f"-{m}") and run_name.startswith(f"t{tier}-ep{ep}-w2048-") and step == "349999104", (m, run_name)
                seen[int(tier)].append(int(ep))
        assert sorted(seen[1]) == list(range(30)) and sorted(seen[2]) == list(range(30)), seen
        allrows = rows(REPO / "stages/evaluate/exports/mf4090.tsv") + rows(REPO / "stages/evaluate/exports/ml40.tsv")
        assert ["1", "20", "t1-ep20-w2048-cap4m-mf4090", "349999104"] in allrows
        assert ["1", "7", "t1-ep7-w2048-cap4m-p185g-mf4090", "349999104"] in allrows
        assert ["2", "7", "t2-ep7-w2048-cap10m-p185g-ml40", "349999104"] in allrows


def fake(root: Path, dup: bool = False, drop: int = -1, guarded: tuple = ()) -> None:
    (root / "results/day2").mkdir(parents=True); (root / "stages/policy/queues").mkdir(parents=True)
    lines = []
    for ep in range(30):
        if ep == drop:
            continue
        m = "mf4090" if ep % 2 else "ml40"; cap = "cap4m" if m == "mf4090" else "cap10m"
        lines.append("summary_json " + json.dumps({"run": f"t1-ep{ep}-w2048-{cap}-{m}", "playback": "frame", "verdict": "PASS"}))
        lines.append("summary_json " + json.dumps({"run": f"t1-ep{ep}-w2048-{cap}-rt-{m}", "playback": "real", "verdict": "PASS"}))
    if dup:
        lines.append("summary_json " + json.dumps({"run": "t1-ep3-w2048-cap10m-ml40", "playback": "frame", "verdict": "FAIL"}))
    (root / "results/day2/batch_verdict_x.txt").write_text("\n".join(lines) + "\n")
    hdr = "run\tx\n"
    (root / "stages/policy/queues/mf4090_box.expect.tsv").write_text(hdr)
    (root / "stages/policy/queues/ml40_box.expect.tsv").write_text(hdr)
    for m in ("ml40", "mf4090"):
        for s in "ab":
            eps = [ep for ep in range(30) if (ep % 2 == (m == "mf4090")) and (ep < 15) == (s == "a")]
            cap = "cap4m" if m == "mf4090" else "cap10m"
            (root / f"stages/policy/queues/{m}_t2{s}.expect.tsv").write_text(hdr + "".join(f"t2-ep{ep}-w2048-{cap}-{m}\t1\n" for ep in eps))
    (root / "stages/policy/queues/ml40_t2c.expect.tsv").write_text(hdr)
    for m in ("mf4090", "ml40"):
        (root / f"stages/policy/queues/{m}_nfg.expect.tsv").write_text(hdr + "".join(f"{r}\t1\n" for r in guarded if r.endswith(m)))


def test_fake_ok_and_failures() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t) / "r"; fake(root); out = Path(t) / "o"
        r = run(root, out)
        assert r.returncode == 0, r.stderr
        assert not any("-rt-" in x[2] for m in ("mf4090", "ml40") for x in rows(out / f"{m}.tsv"))
        r = run(root, out)
        assert r.returncode != 0 and "拒绝覆盖" in r.stderr, r.stderr
    for kw, word in (({"dup": True}, "Tier 1 ep3"), ({"drop": 7}, "Tier 1 ep7：没有 run")):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "r"; fake(root, **kw); out = Path(t) / "o"
            r = run(root, out)
            assert r.returncode != 0 and word in r.stderr and not out.exists(), (kw, r.stderr)


def test_guarded_retrain() -> None:
    g1, g2 = "t1-ep7-w2048-cap4m-p185g-mf4090", "t2-ep4-w2048-cap10m-p185g-ml40"
    with tempfile.TemporaryDirectory() as t:
        root = Path(t) / "r"; fake(root, guarded=(g1, g2)); out = Path(t) / "o"
        r = run(root, out)
        assert r.returncode == 0, r.stderr
        allrows = rows(out / "mf4090.tsv") + rows(out / "ml40.tsv")
        assert ["1", "7", g1, "349999104"] in allrows and ["2", "4", g2, "349999104"] in allrows, allrows
        assert not any(x[1] == "7" and x[0] == "1" and x[2] != g1 for x in allrows)
        assert sum(x[0] == "1" for x in allrows) == 30 and sum(x[0] == "2" for x in allrows) == 30
    with tempfile.TemporaryDirectory() as t:
        root = Path(t) / "r"; fake(root, guarded=(g1, "t1-ep7-w2048-cap10m-p185g-ml40")); out = Path(t) / "o"
        r = run(root, out)
        assert r.returncode != 0 and "Tier 1 ep7" in r.stderr and not out.exists(), r.stderr


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"ok  {name}")
    print("ALL_OK")
