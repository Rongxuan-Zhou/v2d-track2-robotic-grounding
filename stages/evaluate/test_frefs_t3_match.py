"""frefs_t3_match.py 的测试（暂无契约 ID：运行规范待成文；计划 agent-doc/plan-r3o-2026-10-15.md 2.4.1）。只用标准库。

临时目录里放结果目录（ep3、ep7 的 metrics JSON，记着 checkpoint 与 motion parquet）与帧参考目录（两台机器各一份 manifest、两个 npz）。
  1. 对得上：FREFS_T3_MATCH_OK 2；帧参考里多出来的集只报不拒；
  2. 逐集 FAIL（退出码 1）：run、步数、档位目录、序列与导出不符；npz 缺、sha256 不对；这一集在 manifest 里 0 行或 2 行；JSON 的路径认不出；
     结果目录里这一集只有 parquet 没有 JSON，或只有 JSON 没有 parquet；
  3. manifest 表头不对：FAIL 1；结果目录里既没有 JSON 也没有 parquet：FAIL。
用法：python3 stages/evaluate/test_frefs_t3_match.py
"""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
HEAD = "episode\trun\tstep\ttier_dir\tsequence\tframes\tposes_sha256\tsource_parquet_sha256\tfile\tsha256\n"
EPS = {3: ("t3-ep3-w2048-cap4m-r3a-hm-mf4090", "349999104", "tier3_hm", "seq_three", "mf4090"),
       7: ("t3-ep7-w2048-cap10m-r3a-hm-ml40", "349999104", "tier3_hm", "seq_seven", "ml40")}


def setup(root: Path) -> tuple:
    res, fref = root / "res", root / "fref"
    res.mkdir()
    fref.mkdir()
    man = {"mf4090": HEAD, "ml40": HEAD}
    for ep, (run, step, tdir, seq, m) in EPS.items():
        (res / f"episode_{ep:06d}.json").write_text(json.dumps({
            "checkpoint": f"/repo/runs/flash_chord/{run}/policy_{step}.safetensors",
            "parquet": f"/repo/runs/motion/{tdir}/whole_body/soma/sequence_id={seq}/robot_name=g1/data.parquet"}))
        (res / f"episode_{ep:06d}.parquet").write_text("p")
        npz = fref / f"episode_{ep:06d}.npz"
        npz.write_bytes(f"npz {ep}".encode())
        man[m] += f"{ep}\t{run}\t{step}\t{tdir}\t{seq}\t40\tpsha\tssha\t{npz.name}\t{hashlib.sha256(npz.read_bytes()).hexdigest()}\n"
    for m, text in man.items():
        (fref / f"manifest_{m}.tsv").write_text(text)
    return res, fref


def match(res: Path, fref: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(REPO / "stages/evaluate/frefs_t3_match.py"), str(res), str(fref)], capture_output=True, text=True, timeout=60)


def test_ok() -> None:
    with tempfile.TemporaryDirectory() as t:
        res, fref = setup(Path(t))
        r = match(res, fref)
        assert r.returncode == 0 and r.stdout.rstrip().endswith("FREFS_T3_MATCH_OK 2") and "FAIL" not in r.stdout, r.stdout + r.stderr
        (fref / "manifest_ml40.tsv").write_text((fref / "manifest_ml40.tsv").read_text()
                                                + "9\tt3-ep9-x\t1\ttier3_hm\ts9\t40\tp\ts\tepisode_000009.npz\tabc\n")
        r = match(res, fref)
        assert r.returncode == 0 and "帧参考里有、结果目录里没有的集（打包器不读）：[9]" in r.stdout, r.stdout


def edit(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    assert old in text, (path, old)
    path.write_text(text.replace(old, new))


def test_episode_failures() -> None:
    cases = {
        "run": (lambda res, fref: edit(fref / "manifest_mf4090.tsv", "t3-ep3-w2048-cap4m-r3a-hm-mf4090", "t3-ep3-w2048-cap4m-r3a-v2-mf4090"), "run "),
        "step": (lambda res, fref: edit(res / "episode_000003.json", "policy_349999104", "policy_4096000"), "step "),
        "tier_dir": (lambda res, fref: edit(res / "episode_000003.json", "motion/tier3_hm/", "motion/tier3_v2/"), "tier_dir "),
        "sequence": (lambda res, fref: edit(res / "episode_000003.json", "sequence_id=seq_three", "sequence_id=seq_other"), "sequence "),
        "npz_sha": (lambda res, fref: (fref / "episode_000003.npz").write_bytes(b"changed"), "sha256"),
        "npz_missing": (lambda res, fref: (fref / "episode_000003.npz").unlink(), "没有 episode_000003.npz"),
        "dup": (lambda res, fref: edit(fref / "manifest_ml40.tsv", HEAD, HEAD + (fref / "manifest_mf4090.tsv").read_text().splitlines()[1] + "\n"), "有 2 行"),
        "none": (lambda res, fref: edit(fref / "manifest_mf4090.tsv", (fref / "manifest_mf4090.tsv").read_text(), HEAD), "有 0 行"),
        "json": (lambda res, fref: (res / "episode_000003.json").write_text(json.dumps({"checkpoint": "x", "parquet": "y"})), "认不出"),
        "no_json": (lambda res, fref: (res / "episode_000003.json").unlink(), "没有 metrics JSON"),
        "no_parquet": (lambda res, fref: (res / "episode_000003.parquet").unlink(), "没有 parquet"),
    }
    for name, (mutate, word) in cases.items():
        with tempfile.TemporaryDirectory() as t:
            res, fref = setup(Path(t))
            mutate(res, fref)
            r = match(res, fref)
            assert r.returncode == 1 and r.stdout.rstrip().endswith("FREFS_T3_MATCH_FAIL 1"), (name, r.stdout + r.stderr)
            assert "FAIL ep3" in r.stdout and word in r.stdout and "FAIL ep7" not in r.stdout, (name, r.stdout)


def test_file_failures() -> None:
    with tempfile.TemporaryDirectory() as t:   # 表头不对
        res, fref = setup(Path(t))
        edit(fref / "manifest_ml40.tsv", "source_parquet_sha256", "src")
        r = match(res, fref)
        assert r.returncode == 1 and "读不了帧参考的 manifest" in r.stdout and r.stdout.rstrip().endswith("FREFS_T3_MATCH_FAIL 1"), r.stdout
    with tempfile.TemporaryDirectory() as t:   # 结果目录里既没有 JSON 也没有 parquet
        res, fref = setup(Path(t))
        for j in list(res.glob("*.json")) + list(res.glob("*.parquet")):
            j.unlink()
        r = match(res, fref)
        assert r.returncode == 1 and "没有 episode_XXXXXX.json 与 .parquet" in r.stdout, r.stdout


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("ALL_OK")
