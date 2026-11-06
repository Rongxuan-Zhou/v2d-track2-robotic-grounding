"""assemble_t3_results.py 的测试（只用标准库，Mac 上 python3 直接跑）。计划 agent-doc/plan-t3pack-2026-10-29.md 第 1 节第 3 条与复核 NIT。"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = HERE / "assemble_t3_results.py"


def ep(d: Path, e: int, run: str, step: int, payload: bytes = b"p"):
    d.mkdir(parents=True, exist_ok=True)
    (d / f"episode_{e:06d}.json").write_text(json.dumps(
        {"checkpoint": f"/repo/runs/flash_chord/{run}/policy_{step}.safetensors", "environment_steps": step}))
    (d / f"episode_{e:06d}.parquet").write_bytes(payload)


def run(*args):
    return subprocess.run([sys.executable, str(TOOL), *map(str, args)], capture_output=True, text=True)


def main():
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        old, new = t / "old", t / "new"
        ep(old, 0, "rA", 10); ep(old, 1, "rB", 10); ep(new, 1, "rC", 10, b"new")
        sel = t / "sel.tsv"
        sel.write_text("3\t0\trA\t10\tmf4090\n3\t1\trC\t10\tml40\n")
        r = run(sel, t / "out", old, new)
        assert r.returncode == 0 and "ASSEMBLE_T3_OK 2" in r.stdout, r.stderr
        assert (t / "out" / "episode_000001.parquet").read_bytes() == b"new"
        assert not (t / "out" / "ASSEMBLE.json").exists() and (t / "out.ASSEMBLE.json").exists()
        # 目标已存在就拒绝
        assert run(sel, t / "out", old, new).returncode != 0
        # 两个源都符合：拒绝
        ep(t / "dup", 1, "rC", 10)
        assert run(sel, t / "out2", old, new, t / "dup").returncode != 0
        # 步数不对：找不到源，拒绝
        sel.write_text("3\t0\trA\t11\tmf4090\n")
        assert run(sel, t / "out3", old).returncode != 0
        # 档位不是 3、episode 重复：拒绝
        sel.write_text("2\t0\trA\t10\tmf4090\n")
        assert run(sel, t / "out4", old).returncode != 0
        sel.write_text("3\t0\trA\t10\tmf4090\n3\t0\trA\t10\tmf4090\n")
        assert run(sel, t / "out5", old).returncode != 0
        # 缺 parquet：拒绝
        (old / "episode_000000.parquet").unlink()
        sel.write_text("3\t0\trA\t10\tmf4090\n")
        assert run(sel, t / "out6", old).returncode != 0
    print("TEST_ASSEMBLE_T3_OK")


if __name__ == "__main__":
    main()
