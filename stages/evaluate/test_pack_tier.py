"""pack_tier.sh 的测试（暂无契约 ID：运行规范待成文）。只用标准库。计划 agent-doc/plan-submission-exports-2026-10-02.md 第 3.5 节。

临时目录里建一个 git 仓库（HEAD 有假的 origin 分支），放真的 pack_tier.sh 与 scripts/dev/record，再放一个假的 kit：假打包器按参数写 CSV、
打印 “locally computed: …”，把收到的参数记下来；假上传器把收到的参数记下来。V2D_PACK_PYTHON 用本机 python，不起 uv。
  1. Tier 2：结果文件与标准输出里都没有 “locally computed”；打包器收到 --frame-references 与这次 HEAD 的 commit URL；上传器收到 --dry-run；
  2. Tier 1：那一行照录；上传器同样只收到 --dry-run；
  3. 给了 episode（预演）：打包器收到 --config，里面这一档的 scored_episodes 是这几条；不调上传器；
  4. 打包器失败：退出码非零，不调上传器；反向对照（V2D_PACK_NEGCTL=1）：不给帧参考、打包器报刚体变换不对 → NEGCTL_OK；只许 Tier 2 预演用；
  5. 拒绝：工作树有未提交改动、HEAD 不在 origin 上、CSV 已存在、kit 的 zip 不对、缺 Tier 2 的帧参考、结果目录里有不合名字的 parquet；
  6. Tier 3（计划 agent-doc/plan-r3o-2026-10-15.md 2.4、2.4.1）：“locally computed” 同 Tier 2 滤掉；打包器收到 V2D_FRAME_REFS 给的 --frame-references
     （假打包器同真的：Tier 3 不给就报 policy input reference not found）；track 是 track_2_tier3；上传器只收到 --dry-run；档位 4 拒绝；
  7. 帧参考的拒绝（退出码 2、不调打包器）：Tier 3 没设 V2D_FRAME_REFS、目录里没有 manifest_*.tsv、缺某条 episode 的 npz、manifest 的 run 与导出 JSON
     不符（frefs_t3_match.py 逐集核；它自己的各种情形见 test_frefs_t3_match.py）；Tier 1、2 设了 V2D_FRAME_REFS。
用法：python3 stages/evaluate/test_pack_tier.py
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
KITREL = "kit/v2d_submission_kit"
FAKE_PACK = r'''import json, sys
from pathlib import Path
a = sys.argv[1:]
Path("runs/pack_argv.json").write_text(json.dumps(a))
if Path("runs/pack_fail").exists():
    print("error: the trajectories do not line up with the reference"); sys.exit(1)
if "track_2_tier2" in a and "--frame-references" not in a:
    print("error: the trajectories do not line up with the reference:\n  - episode_000002.parquet: the evaluator and released references do not differ by one rigid transform"); sys.exit(1)
if "track_2_tier3" in a and "--frame-references" not in a:
    print("policy input reference not found at data/challenge/track_2/cari4d_reconstruction. Download the selected tier's input or pass --frame-references."); sys.exit(1)
out = Path(a[a.index("-o") + 1]); out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("id,x\nr1,1\nr2,2\n")
print(f"wrote {out} (2 rows)")
print("locally computed: AUC=0.1234  SP-SR=0.5000  MP-SR=0.0000  MPPE=9.8765")
print(f"upload with: python submit/v2d_submit.py {out} --track x -m '<message>'")
'''
FAKE_SUBMIT = r'''import json, sys
from pathlib import Path
Path("runs/submit_argv.json").write_text(json.dumps(sys.argv[1:]))
print("looks valid"); print("--dry-run: nothing submitted." if "--dry-run" in sys.argv else "UPLOADED")
'''


def git(root: Path, *a) -> str:
    return subprocess.run(["git", *a], cwd=root, check=True, capture_output=True, text=True).stdout


def setup(root: Path) -> dict:
    for d in ("stages/evaluate", "scripts/dev", f"{KITREL}/tools", f"{KITREL}/submit", f"{KITREL}/config",
              "data/challenge/track_2/tier_1_multiview_caption/meta", "runs/submission/frame_refs_t2", "runs/submission/final/t1", "runs/submission/final/t2", "runs/submission/final/t3"):
        (root / d).mkdir(parents=True, exist_ok=True)
    for f in ("stages/evaluate/pack_tier.sh", "scripts/dev/record", "stages/evaluate/frefs_t3_match.py"):
        shutil.copy(REPO / f, root / f); (root / f).chmod(0o755)
    (root / f"{KITREL}/tools/pack_submission.py").write_text(FAKE_PACK)
    (root / f"{KITREL}/submit/v2d_submit.py").write_text(FAKE_SUBMIT)
    (root / f"{KITREL}/requirements.txt").write_text("numpy\n")
    (root / f"{KITREL}/config/tracks.json").write_text(json.dumps({"tracks": {"track_2_tier1": {"scored_episodes": "all"}, "track_2_tier2": {"scored_episodes": "all"}, "track_2_tier3": {"scored_episodes": "all"}}}))
    (root / "kit/v2d_submission_kit.zip").write_bytes(b"zip")
    (root / "runs/submission/frame_refs_t2/manifest.tsv").write_text("x\n")
    (root / "runs/submission/frame_refs_t3").mkdir(parents=True)
    (root / "runs/submission/frame_refs_t3/episode_000003.npz").write_bytes(b"npz")
    (root / "runs/submission/frame_refs_t3/manifest_mf4090.tsv").write_text(
        "episode\trun\tstep\ttier_dir\tsequence\tframes\tposes_sha256\tsource_parquet_sha256\tfile\tsha256\n"
        f"3\tt3-ep3-w2048-cap4m-r3a-hm-mf4090\t349999104\ttier3_hm\tseq3\t40\tp\ts\tepisode_000003.npz\t{hashlib.sha256(b'npz').hexdigest()}\n")
    for t in ("t1", "t2", "t3"):
        (root / f"runs/submission/final/{t}/episode_000003.parquet").write_text("p")
    (root / "runs/submission/final/t3/episode_000003.json").write_text(json.dumps({
        "checkpoint": "/repo/runs/flash_chord/t3-ep3-w2048-cap4m-r3a-hm-mf4090/policy_349999104.safetensors",
        "parquet": "/repo/runs/motion/tier3_hm/whole_body/soma/sequence_id=seq3/robot_name=g1/data.parquet"}))
    (root / ".gitignore").write_text("runs/\ndata/\nkit/\n")
    git(root, "init", "-q"); git(root, "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A")
    git(root, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "x")
    head = git(root, "rev-parse", "HEAD").strip()
    git(root, "update-ref", "refs/remotes/origin/main", head)
    return {"head": head}


def env(root: Path, **kw) -> dict:
    e = dict(os.environ, V2D_KIT_DIR=KITREL, V2D_KIT_ZIP_SHA=hashlib.sha256(b"zip").hexdigest(), V2D_PACK_PYTHON=sys.executable)
    e.update(kw)
    return {k: v for k, v in e.items() if v is not None}


def pack(root: Path, *args, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "stages/evaluate/pack_tier.sh", *args], cwd=root, env=env(root, **kw), capture_output=True, text=True, timeout=120)


def argv(root: Path, name: str):
    f = root / f"runs/{name}_argv.json"
    return json.loads(f.read_text()) if f.exists() else None


def test_tier2_filters_local_scores() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); info = setup(root)
        r = pack(root, "2", "runs/submission/final/t2", "x")
        rec = (root / "results/submission/pack/t2_x.txt").read_text()
        assert r.returncode == 0 and r.stdout.rstrip().endswith("PACK_OK"), r.stdout + r.stderr
        assert "locally computed" not in rec and "locally computed" not in r.stdout and "locally computed" not in r.stderr, rec
        assert "CSV runs/submission/csv/track2_tier2_x.csv：2 行" in rec, rec
        a = argv(root, "pack")
        assert a[a.index("--frame-references") + 1] == "runs/submission/frame_refs_t2" and a[a.index("--track") + 1] == "track_2_tier2", a
        assert a[a.index("--code-commit") + 1] == f"https://github.com/Manifold-ai/maniphys_lab_v2d_challenge/commit/{info['head']}", a
        s = argv(root, "submit")
        assert "--dry-run" in s and s[0] == "runs/submission/csv/track2_tier2_x.csv" and s[s.index("--track") + 1] == "track_2_tier2", s
        assert "nothing submitted" in (root / "results/submission/pack/t2_x_dryrun.txt").read_text()


def test_tier1_keeps_local_scores() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root)
        r = pack(root, "1", "runs/submission/final/t1", "x")
        rec = (root / "results/submission/pack/t1_x.txt").read_text()
        assert r.returncode == 0 and "locally computed: AUC=0.1234" in rec, rec
        assert "--frame-references" not in argv(root, "pack") and "--dry-run" in argv(root, "submit")


def test_rehearsal_config() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root)
        r = pack(root, "1", "runs/submission/final/t1", "rh", "6", "8")
        a = argv(root, "pack")
        cfg = json.loads((root / a[a.index("--config") + 1]).read_text())
        assert r.returncode == 0 and "预演" in r.stdout and cfg["tracks"]["track_2_tier1"]["scored_episodes"] == [6, 8], (r.stdout, cfg)
        assert argv(root, "submit") is None


def test_packer_failure() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root)
        (root / "runs/pack_fail").write_text("")
        r = pack(root, "2", "runs/submission/final/t2", "x")
        assert r.returncode != 0 and argv(root, "submit") is None and (root / "results/submission/pack/t2_x.txt.failed").exists(), r.stdout


def test_negctl() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root)
        r = pack(root, "2", "runs/submission/final/t2", "neg", "2", V2D_PACK_NEGCTL="1")
        assert r.returncode == 0 and "NEGCTL_OK" in r.stdout and "--frame-references" not in argv(root, "pack"), r.stdout + r.stderr
        assert argv(root, "submit") is None and not (root / "runs/submission/csv/track2_tier2_neg.csv").exists()
    for args in (("1", "runs/submission/final/t1", "neg", "2"), ("2", "runs/submission/final/t2", "neg")):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); setup(root)
            r = pack(root, *args, V2D_PACK_NEGCTL="1")
            assert r.returncode == 2 and argv(root, "pack") is None, (args, r.stderr)


def test_refusals() -> None:
    cases = {
        "dirty": lambda root: (root / "new.txt").write_text("x"),
        "unpushed": lambda root: git(root, "update-ref", "-d", "refs/remotes/origin/main"),
        "csv": lambda root: ((root / "runs/submission/csv").mkdir(parents=True), (root / "runs/submission/csv/track2_tier2_x.csv").write_text("old")),
        "zip": lambda root: (root / "kit/v2d_submission_kit.zip").write_bytes(b"other"),
        "refs": lambda root: (root / "runs/submission/frame_refs_t2/manifest.tsv").unlink(),
        "name": lambda root: (root / "runs/submission/final/t2/foo.parquet").write_text("p"),
    }
    for name, mutate in cases.items():
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); setup(root); mutate(root)
            r = pack(root, "2", "runs/submission/final/t2", "x")
            assert r.returncode == 2 and argv(root, "pack") is None and argv(root, "submit") is None, (name, r.stdout, r.stderr)


def test_tier3_filters_local_scores() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root)
        r = pack(root, "3", "runs/submission/final/t3", "x", V2D_FRAME_REFS="runs/submission/frame_refs_t3")
        rec = (root / "results/submission/pack/t3_x.txt").read_text()
        assert r.returncode == 0 and r.stdout.rstrip().endswith("PACK_OK"), r.stdout + r.stderr
        assert "locally computed" not in rec and "locally computed" not in r.stdout and "locally computed" not in r.stderr, rec
        assert "帧参考 runs/submission/frame_refs_t3：1 个 npz，manifest_mf4090.tsv sha256" in r.stdout and "逐集核对 FREFS_T3_MATCH_OK 1" in r.stdout, r.stdout
        a = argv(root, "pack")
        assert a[a.index("--frame-references") + 1] == "runs/submission/frame_refs_t3" and a[a.index("--track") + 1] == "track_2_tier3", a
        s = argv(root, "submit")
        assert "--dry-run" in s and s[s.index("--track") + 1] == "track_2_tier3", s
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root)
        r = pack(root, "4", "runs/submission/final/t3", "x")
        assert r.returncode == 2 and argv(root, "pack") is None, r.stdout + r.stderr


def test_frame_refs_refusals() -> None:
    F = "runs/submission/frame_refs_t3"
    cases = {
        "unset": ("3", "runs/submission/final/t3", None, lambda root: None, "V2D_FRAME_REFS"),
        "manifest": ("3", "runs/submission/final/t3", F, lambda root: (root / F / "manifest_mf4090.tsv").unlink(), "manifest_"),
        "npz": ("3", "runs/submission/final/t3", F, lambda root: (root / F / "episode_000003.npz").unlink(), "episode_000003.npz"),
        "run": ("3", "runs/submission/final/t3", F, lambda root: (root / F / "manifest_mf4090.tsv").write_text(
            (root / F / "manifest_mf4090.tsv").read_text().replace("-hm-mf4090", "-v2-mf4090")), "与导出不符"),
        "tier2": ("2", "runs/submission/final/t2", F, lambda root: None, "只给 Tier 3 用"),
        "tier1": ("1", "runs/submission/final/t1", F, lambda root: None, "只给 Tier 3 用"),
    }
    for name, (tier, res, fref, mutate, word) in cases.items():
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); setup(root); mutate(root)
            r = pack(root, tier, res, "x", V2D_FRAME_REFS=fref)
            assert r.returncode == 2 and word in r.stderr and argv(root, "pack") is None and argv(root, "submit") is None, (name, r.stdout, r.stderr)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"ok  {name}")
    print("ALL_OK")
