"""merge_frefs_t3.py 的测试（暂无契约 ID：运行规范待成文；计划 agent-doc/plan-t3-kfix-2026-10-16.md 方案 4）。只用标准库：合并工具只核 sha256、
不解析 npz，测试里的 npz 用任意字节。
  1. 正常：源 A（manifest_mf4090：ep0、ep18；manifest_ml40：ep5）取 ep0、ep18，源 B（manifest_mf4090：ep0、ep1、ep2，带 ROTATION.json）取 ep1、ep2；
     目标里 4 个 npz 与源逐字节相同，manifest_mf4090.tsv 4 行按集排序、列与源相同，没有 manifest_ml40.tsv；MERGE.json 记了两个源、各取的集、
     B 的 ROTATION.json 的 sha256 与 label；frefs_t3_match.manifest_rows 读得通；
  2. 拒绝（退出码非 0，不建目标目录）：同一集来自两个源；要的集不在源 manifest 里；源 npz 被改过；--take 写法错；
     目标目录已存在（不动它）。
用法：python3 stages/evaluate/test_merge_frefs_t3.py；最后一行 ALL_OK。
"""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "stages/evaluate"))
from frefs_t3_match import COLS, manifest_rows

TOOL = REPO / "stages/evaluate/merge_frefs_t3.py"


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def make_src(d: Path, rows: dict, rotation=None) -> None:
    d.mkdir(parents=True)
    for mach, eps in rows.items():
        lines = ["\t".join(COLS)]
        for ep in eps:
            f = d / f"episode_{ep:06d}.npz"
            f.write_bytes(f"{d.name}-{ep}".encode() * 50)
            lines.append("\t".join([str(ep), f"t3-ep{ep}-run", "349999104", "tier3_v2", f"seq{ep}", "100", "a" * 64, "b" * 64, f.name, sha256(f)]))
        (d / f"manifest_{mach}.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if rotation:
        (d / "ROTATION.json").write_text(json.dumps(rotation), encoding="utf-8")


def run(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(TOOL), *map(str, args)], capture_output=True, text=True, check=False)


def setup(root: Path) -> tuple:
    a, b = root / "A", root / "B"
    make_src(a, {"mf4090": (0, 18), "ml40": (5,)})
    make_src(b, {"mf4090": (0, 1, 2)}, {"label": "gridG_yaw", "rotation": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]})
    return a, b


def test_normal() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        a, b = setup(root)
        out = root / "out"
        r = run("--out", out, "--take", f"{a}=0,18", "--take", f"{b}=1,2")
        assert r.returncode == 0 and r.stdout.strip().endswith("MERGE_FREFS_OK 4"), r.stdout + r.stderr
        for src, ep in ((a, 0), (a, 18), (b, 1), (b, 2)):
            f = f"episode_{ep:06d}.npz"
            assert (out / f).read_bytes() == (src / f).read_bytes(), ep
        lines = (out / "manifest_mf4090.tsv").read_text(encoding="utf-8").splitlines()
        assert lines[0].split("\t") == COLS and [ln.split("\t")[0] for ln in lines[1:]] == ["0", "1", "2", "18"], lines
        assert not (out / "manifest_ml40.tsv").exists()
        rec = json.loads((out / "MERGE.json").read_text(encoding="utf-8"))
        assert [s["episodes"] for s in rec["sources"]] == [[0, 18], [1, 2]]
        assert rec["sources"][1]["rotation_label"] == "gridG_yaw" and rec["sources"][1]["rotation_json_sha256"] == sha256(b / "ROTATION.json")
        assert sorted(manifest_rows(out)) == [0, 1, 2, 18]
    print("  1. 正常：两个源按集合并，npz 逐字节相同，manifest 按集排序，MERGE.json 齐全，frefs_t3_match 读得通")


def test_refuse() -> None:
    cases = []
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        a, b = setup(root)
        cases.append(("同一集来自两个源", run("--out", root / "o1", "--take", f"{a}=0", "--take", f"{b}=0,1"), root / "o1"))
        cases.append(("要的集不在 manifest 里", run("--out", root / "o2", "--take", f"{a}=0,7"), root / "o2"))
        cases.append(("--take 写法错", run("--out", root / "o3", "--take", f"{a}"), root / "o3"))
        (a / "episode_000018.npz").write_bytes(b"tampered")
        cases.append(("源 npz 被改过", run("--out", root / "o4", "--take", f"{a}=0,18"), root / "o4"))
        for name, r, out in cases:
            assert r.returncode != 0 and not out.exists(), (name, r.stdout, r.stderr)
        cases.append(("--take-plain 的源目录有 ROTATION.json", run("--out", root / "o6", "--take-plain", f"{b}=1"), root / "o6"))
        for name, r, out in cases[-1:]:
            assert r.returncode != 0 and not out.exists(), (name, r.stdout, r.stderr)
        keep = root / "o5"
        keep.mkdir()
        (keep / "x").write_text("x")
        r = run("--out", keep, "--take", f"{b}=1")
        assert r.returncode != 0 and (keep / "x").read_text() == "x" and len(list(keep.iterdir())) == 1, r.stdout
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        a, b = setup(root)
        r = run("--out", root / "o7", "--take-plain", f"{a}=0", "--take", f"{b}=1,2")
        assert r.returncode == 0 and r.stdout.strip().endswith("MERGE_FREFS_OK 3"), r.stdout + r.stderr
    print("  2. 拒绝：同一集来自两个源、要的集不在 manifest 里、--take 写法错、源 npz 被改过、目标目录已存在、--take-plain 的源有 ROTATION.json；"
          "--take-plain 用在没有 ROTATION.json 的源上照常合并")


if __name__ == "__main__":
    test_normal()
    test_refuse()
    print("ALL_OK")
