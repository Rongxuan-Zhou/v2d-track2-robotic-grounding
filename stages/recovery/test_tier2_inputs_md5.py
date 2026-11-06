"""tier2_inputs_md5.py 的测试（守计划 agent-doc/plan-t3-v2-retarget18-2026-10-10.md 第 2 节；暂无契约 ID）。只用标准库。
  1. 在假 Tier 2 目录里按 episodes_metadata.jsonl 取出给定几集的三个文件，打“<md5> <相对路径>”，共用的网格只打一次，
     输出能被 tier3_expect.py 的 Tier 2 记录正则取到
  2. 没有的集、缺文件、带 .. 的路径：报错
用法：python3 stages/recovery/test_tier2_inputs_md5.py
"""
import hashlib
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TIER2_LINE = re.compile(r"^([0-9a-f]{32}) ((?:mesh|ground_plane|support_surface)/\S+)$")   # 同 tier3_expect.py


def fake(root: Path, rows: list) -> None:
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "episodes_metadata.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    for r in rows:
        for k in ("mesh", "ground_plane", "support_surface"):
            f = root / r[k]
            if ".." not in r[k] and not f.exists():
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(f"{k} {r[k]}\n")


def row(ep: int, obj: str, seq: str) -> dict:
    return {"episode_index": ep, "sequence_id": seq, "mesh": f"mesh/{obj}/{obj}.glb",
            "ground_plane": f"ground_plane/{seq}.json", "support_surface": f"support_surface/{seq}_support.usda"}


def run(root: Path, *eps: str):
    return subprocess.run([sys.executable, str(HERE / "tier2_inputs_md5.py"), "--tier2", str(root), *eps],
                          capture_output=True, text=True)


def test_lists_md5_once():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t) / "tier2"
        fake(root, [row(0, "stool", "s0"), row(3, "mug", "s3"), row(21, "stool", "s21")])
        p = run(root, "0", "21")
        assert p.returncode == 0, p.stderr
        got = [m.groups() for m in map(TIER2_LINE.match, p.stdout.splitlines()) if m]
        want = {"mesh/stool/stool.glb", "ground_plane/s0.json", "support_surface/s0_support.usda",
                "ground_plane/s21.json", "support_surface/s21_support.usda"}
        assert sorted(rel for _, rel in got) == sorted(want), p.stdout
        for h, rel in got:
            assert h == hashlib.md5((root / rel).read_bytes()).hexdigest(), rel
        assert p.stdout.splitlines()[0].startswith("== ") and "TIER2_INPUTS_DONE（2 集，5 个文件）" in p.stdout, p.stdout


def test_missing_episode_refuses():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t) / "tier2"
        fake(root, [row(0, "stool", "s0")])
        p = run(root, "0", "5")
        assert p.returncode != 0 and "没有 ep[5]" in p.stderr, p.stdout + p.stderr


def test_missing_file_refuses():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t) / "tier2"
        fake(root, [row(0, "stool", "s0")])
        (root / "ground_plane" / "s0.json").unlink()
        p = run(root, "0")
        assert p.returncode != 0 and "ground_plane 在 Tier 2 目录里没有" in p.stderr, p.stdout + p.stderr


def test_unsafe_path_refuses():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t) / "tier2"
        r = row(0, "stool", "s0")
        r["mesh"] = "../outside.glb"
        fake(root, [r])
        p = run(root, "0")
        assert p.returncode != 0 and "路径不安全" in p.stderr, p.stdout + p.stderr


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("ALL_OK")
