"""frefs_from_motion_t3.py 的测试（暂无契约 ID：运行规范待成文；计划 agent-doc/plan-t3b2-2026-10-28.md 第 1 步第 4 条）。要 numpy、pyarrow。
借 test_make_frame_refs_t3.py 的最小仓库（真的 make_frame_refs_t3.py 与 tier_guard.py，假的 motion_src、Tier 3 根与 run 记录）：
  1. 同一集、同一份 motion 源：frefs_from_motion_t3 出的 npz 与 make_frame_refs_t3 出的逐字节相同；清单的列同 make_frame_refs_t3，run 是 -、步数是 0；
  2. 逐条 FAIL：motion_src 里没有这一集；一集两个序列时整份拒绝；
  3. 拒绝：档位目录不是 tier3[_<名>]；episode 重复；输出目录已存在；
  4. 档位隔离：forbid_tier1 生效（序列名带 tier_1_multiview_caption 时读 v2d_meta.json 被拦下）。
用法：uv run --no-project --python 3.12 --with numpy --with pyarrow python stages/evaluate/test_frefs_from_motion_t3.py
"""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_make_frame_refs_t3 as base   # noqa: E402

REPO = base.REPO


def prep(root: Path, **kw) -> None:
    base.setup(root, **kw)
    shutil.copy(REPO / "stages/evaluate/frefs_from_motion_t3.py", root / "stages/evaluate/frefs_from_motion_t3.py")


def run(root: Path, tdir: str, eps: str, out: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "stages/evaluate/frefs_from_motion_t3.py", tdir, eps, "--out", out],
                          cwd=root, capture_output=True, text=True, timeout=120)


def test_same_bytes() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        prep(root)
        r0 = base.run(root)
        assert r0.returncode == 0, r0.stdout + r0.stderr
        r = run(root, "tier3_hm", "2,5", "runs/x")
        assert r.returncode == 0 and r.stdout.rstrip().endswith("FRAME_REFS_MOTION_OK 2"), r.stdout + r.stderr
        man = (root / "runs/x/manifest_motion.tsv").read_text().splitlines()
        assert man[0] == (root / base.OUT / "manifest_mf4090.tsv").read_text().splitlines()[0], man[0]
        for line in man[1:]:
            ep, run_, step, tdir, seq, n, psha, ssha, f, sha = line.split("\t")
            assert (run_, step, tdir, seq) == ("-", "0", "tier3_hm", f"seq{ep}"), line
            a = (root / "runs/x" / f).read_bytes()
            b = (root / base.OUT / f).read_bytes()
            assert a == b, f"ep{ep} 的 npz 与 make_frame_refs_t3 的不同"
            assert hashlib.sha256(a).hexdigest() == sha


def test_failures_and_refusals() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        prep(root)
        r = run(root, "tier3_hm", "2,9", "runs/y")
        assert r.returncode == 1 and "FAIL ep9" in r.stdout and "没有 episode 9" in r.stdout and r.stdout.rstrip().endswith("FRAME_REFS_MOTION_FAIL 1"), r.stdout
        assert (root / "runs/y/episode_000002.npz").exists() and not (root / "runs/y/episode_000009.npz").exists()
        for tdir, eps, why in (("tier1", "2", "不是 tier3"), ("tier3_HM", "2", "不是 tier3"), ("tier3_hm", "2,2", "重复"), ("tier3_hm", "2", "已存在")):
            r = run(root, tdir, eps, "runs/y")
            assert r.returncode != 0 and why in (r.stdout + r.stderr), (tdir, eps, r.stdout + r.stderr)
        dup = root / "runs/motion_src/tier3_hm/seq2b"
        shutil.copytree(root / "runs/motion_src/tier3_hm/seq2", dup)
        m = json.loads((dup / "v2d_meta.json").read_text())
        m["sequence_id"] = "seq2b"
        (dup / "v2d_meta.json").write_text(json.dumps(m))
        r = run(root, "tier3_hm", "5", "runs/z")
        assert r.returncode != 0 and "两个序列" in (r.stdout + r.stderr), r.stdout + r.stderr


def test_tier1_guard() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        prep(root, seq_fmt="tier_1_multiview_caption_{ep}")
        r = run(root, "tier3_hm", "2", "runs/w")
        assert r.returncode != 0 and "Tier1AccessError" in (r.stdout + r.stderr), r.stdout + r.stderr


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok {name}")
    print("TEST_FREFS_FROM_MOTION_T3_OK")
