"""make_frame_references.py 的测试（暂无契约 ID：运行规范待成文）。要 numpy、pyarrow；装了 pandas 时再用 kit 打包器自己的函数读回来核。

临时目录里搭最小仓库：真的 make_frame_references.py 与 libs/v2d_infra/tier_guard.py，假的 v2d_meta.json、v5.4 npz、发布的 Tier 2 parquet。
  1. 正常：输出的 npz 键、类型、形状与 kit 的 tracks/track_2/README.md 一致，xyzw 换对，manifest 的 sha256 对；
     发布输入有一帧不可见、四元数为零（同真实的 ep11）也照样通过，并打出排除的帧数；
     有 pandas 时：kit 的 input_frame_reference 读得进来，convert_track2_frame 拿“刚体变换后的 v5.4”当评测器参考能对上（残差 < 1e-9）；
  2. v5.4 与发布输入的世界系差一个 diag(1,-1,-1)：FAIL；v5.4 的四元数按 xyzw 存（布局错）：FAIL；帧数不一致：FAIL；来源不是 v5.4：FAIL；
  3. 输出目录已存在：拒绝；
  4. 档位隔离：v2d_meta 的来源指向 Tier 1 目录时被 tier_guard 拦下（FAIL，不是读进来）。
用法：uv run --no-project --python 3.12 --with numpy --with pyarrow --with pandas python stages/evaluate/test_make_frame_references.py
     （没有 pandas 时只跑不依赖 kit 的部分）
"""
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
KIT = REPO / "third_party/v2d_submission_kit/update_0f2f9f4b/v2d_submission_kit"


def quat_wxyz(n, rng):
    q = rng.normal(size=(n, 4)); return q / np.linalg.norm(q, axis=1, keepdims=True)


def setup(root: Path, eps=(0, 1), src_dir="runs/recovery/tier2_v54", denoiser="v5.4", flip=(), short=(), swapq=()) -> dict:
    for f in ("stages/evaluate/make_frame_references.py", "libs/v2d_infra/tier_guard.py"):
        (root / f).parent.mkdir(parents=True, exist_ok=True); shutil.copy(REPO / f, root / f)
    rng = np.random.default_rng(0); truth = {}
    for ep in eps:
        T = 40 + ep
        pos = rng.normal(scale=0.5, size=(T, 3)); wxyz = quat_wxyz(T, rng)
        v54 = np.concatenate([pos, wxyz], axis=1)
        noisy = v54.copy(); noisy[:, :3] += 0.01
        vis = np.ones(T, bool)
        if ep == 1:   # 同真实的 ep11：一帧不可见，发布输入那一帧全零
            noisy[5] = 0.0; vis[5] = False
        if ep in flip:   # 世界系错了：位置 y、z 取反
            v54 = v54.copy(); v54[:, 1:3] *= -1; v54[:, 1:3] += 3.0
        if ep in swapq:   # 四元数布局错了：按 xyzw 存
            v54 = v54.copy(); v54[:, 3:7] = v54[:, [4, 5, 6, 3]]
        seq = f"seq{ep}"
        npz = root / src_dir / f"episode_{ep:06d}.npz"; npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(npz, object_pose=v54[:-1] if ep in short else v54, visible=np.ones(T, bool), meta=np.array("x"))
        m = root / f"runs/motion_src/tier2/{seq}/v2d_meta.json"; m.parent.mkdir(parents=True, exist_ok=True)
        m.write_text(json.dumps({"tier": 2, "episode": ep, "sequence_id": seq,
                                 "object_pose_source": {"kind": "recovery", "path": f"{src_dir}/episode_{ep:06d}.npz", "meta": {"denoiser": denoiser}}}))
        d = root / "data/challenge/track_2/tier_2_synthetic_noise/data/chunk-000"; d.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.table({"frame_index": pa.array(np.arange(T), pa.int64()), "episode_index": pa.array(np.full(T, ep), pa.int64()),
                                 "observation.object.pose": pa.array(list(noisy), pa.list_(pa.float64(), 7)),
                                 "observation.object.visible": pa.array(vis)}), d / f"episode_{ep:06d}.parquet")
        truth[ep] = v54
    return truth


def run(root: Path, *extra) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "stages/evaluate/make_frame_references.py", *extra], cwd=root, capture_output=True, text=True, timeout=120)


def test_normal() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); truth = setup(root)
        r = run(root, "--episodes", "0", "1")
        assert r.returncode == 0 and r.stdout.rstrip().endswith("FRAME_REFS_OK 2") and "粗核排除 1 帧" in r.stdout, r.stdout + r.stderr
        assert "RuntimeWarning" not in r.stderr, r.stderr
        out = root / "runs/submission/frame_refs_t2"
        man = (out / "manifest.tsv").read_text().splitlines()
        assert len(man) == 3 and man[0].startswith("episode\tsequence\tframes"), man
        for line in man[1:]:
            ep, seq, n, src, src_sha, f, sha = line.split("\t")
            assert hashlib.sha256((out / f).read_bytes()).hexdigest() == sha and hashlib.sha256((root / src).read_bytes()).hexdigest() == src_sha
            with np.load(out / f, allow_pickle=False) as z:
                assert sorted(z.files) == ["coordinate_frame", "episode_index", "frame_index", "pose_xyzw"], z.files
                assert str(z["coordinate_frame"].item()) == "released_track2_world" and int(z["episode_index"].item()) == int(ep)
                assert z["frame_index"].dtype == np.int64 and np.array_equal(z["frame_index"], np.arange(int(n)))
                p = z["pose_xyzw"]; v = truth[int(ep)]
                assert p.shape == (int(n), 1, 7) and p.dtype == np.float64
                assert np.array_equal(p[:, 0, :3], v[:, :3]) and np.array_equal(p[:, 0, 3:6], v[:, 4:7]) and np.array_equal(p[:, 0, 6], v[:, 3])
        kit_check(out, truth)


def kit_check(out: Path, truth: dict) -> None:
    try:
        import pandas  # noqa: F401
    except ImportError:
        print("  （没有 pandas：跳过用 kit 函数读回的核对）"); return
    sys.path.insert(0, str(KIT / "tools")); sys.path.insert(0, str(KIT))
    import pack_submission as ps
    for ep, v in truth.items():
        fi = np.arange(len(v))
        ref = ps.input_frame_reference(out, ep, fi, 1, ["object"])
        assert ref.shape == (len(v), 7)
        # 评测器参考 = 对 v5.4 做一个刚体变换（绕 z 转 30°、平移），kit 应恰好换回来
        c, s = np.cos(np.pi / 6), np.sin(np.pi / 6)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]]); rq = np.array([0, 0, np.sin(np.pi / 12), np.cos(np.pi / 12)])
        ev = ref.copy(); ev[:, :3] = ref[:, :3] @ R.T + np.array([1.0, 2.0, 0.5])
        ev[:, 3:] = ps.op._quaternion_multiply(np.broadcast_to(rq, ref[:, 3:].shape), ref[:, 3:])
        conv, dp, da, changed = ps.convert_track2_frame(ev, ev, ref)
        assert dp < 1e-9 and da < 1e-6 and changed, (dp, da)
    print("  kit 的 input_frame_reference、convert_track2_frame 读回核对通过")


def test_failures() -> None:
    for kw, word in (({"flip": (1,)}, "像是世界系或四元数布局不对"), ({"swapq": (1,)}, "像是世界系或四元数布局不对"),
                     ({"short": (1,)}, "v5.4 形状"), ({"denoiser": "v5.3"}, "来源不是 v5.4")):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); setup(root, **kw)
            r = run(root, "--episodes", "0", "1")
            assert r.returncode == 1 and word in r.stdout and "FRAME_REFS_FAIL" in r.stdout, (kw, r.stdout + r.stderr)


def test_refuse_existing() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root)
        (root / "runs/submission/frame_refs_t2").mkdir(parents=True)
        r = run(root, "--episodes", "0")
        assert r.returncode != 0 and "拒绝覆盖" in r.stderr, r.stderr


def test_tier1_source_blocked() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root, eps=(0,), src_dir="data/challenge/track_2/tier_1_multiview_caption/x")
        r = run(root, "--episodes", "0")
        assert r.returncode == 1 and "Tier1AccessError" in r.stdout, r.stdout + r.stderr


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"ok  {name}")
    print("ALL_OK")
