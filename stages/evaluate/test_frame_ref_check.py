"""frame_ref_check.py 的行为测试（暂无契约 ID：运行规范待成文）。要 numpy、pyarrow：
  Mac 上 uv run --no-project --with numpy --with pyarrow python stages/evaluate/test_frame_ref_check.py；训练机宿主机的 python3 也能直接跑。

临时目录里搭最小仓库：真的 frame_ref_check.py 与 libs/v2d_infra/tier_guard.py；发布数据按 LeRobot 的列写（observation.object.pose 为
wxyz、observation.object.visible），motion parquet 是一行多个 list 列，物体位姿 = 发布位姿左乘一个随机刚体变换（就像重定向的换系与地面找平）。
  1. Tier 1：刚体变换下 FRAME_REF_OK；有一帧不可见（位姿全 0）也 OK，因为不计分；某些计分帧再加 2 mm 平移就 FRAME_REF_FAIL；
  2. Tier 2：输入参考取 v2d_meta.json 指向的 v5.4 npz（OpenCV 世界系、wxyz），可见性取 Tier 2 数据：FRAME_REF_OK；
     motion 用的是带噪的发布位姿、npz 是另一条轨迹时 FRAME_REF_FAIL。Tier 2 模式下 tier_guard 生效（不读 Tier 1）。
用法见上。
"""
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
SEQ, T = "2026-01-01_00-00-00_box_test_01", 200


def rand_quat(rng, n):
    q = rng.normal(size=(n, 4)); return q / np.linalg.norm(q, axis=1, keepdims=True)   # wxyz


def qmul_wxyz(a, b):
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw], axis=-1)


def rot_wxyz(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def write_release(root: Path, tier_dir: str, pose_wxyz: np.ndarray, vis: np.ndarray) -> None:
    d = root / f"data/challenge/track_2/{tier_dir}/data/chunk-000"; d.mkdir(parents=True, exist_ok=True)
    pose = pose_wxyz.copy(); pose[~vis] = 0.0
    col = pa.FixedSizeListArray.from_arrays(pa.array(pose.astype(np.float32).ravel()), 7)
    pq.write_table(pa.table({"observation.object.pose": col, "observation.object.visible": pa.array(vis)}), d / "episode_000000.parquet")


def write_motion(root: Path, tier: int, pose_wxyz: np.ndarray, rng) -> None:
    qr = rand_quat(rng, 1)[0]; R = rot_wxyz(qr); t = rng.normal(size=3)
    pos = pose_wxyz[:, :3] @ R.T + t
    quat = qmul_wxyz(np.broadcast_to(qr, (len(pose_wxyz), 4)), pose_wxyz[:, 3:])
    d = root / f"runs/motion/tier{tier}/whole_body/soma/sequence_id={SEQ}/robot_name=g1"; d.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"object_body_position": [pos[:, None].tolist()], "object_body_wxyz": [quat[:, None].tolist()]}), d / "data.parquet")


def setup(root: Path, tier: int, motion_pose: np.ndarray, release_pose: np.ndarray, vis: np.ndarray, npz_pose=None, seed=0) -> None:
    (root / "stages/evaluate").mkdir(parents=True); shutil.copy(REPO / "stages/evaluate/frame_ref_check.py", root / "stages/evaluate")
    (root / "libs/v2d_infra").mkdir(parents=True); shutil.copy(REPO / "libs/v2d_infra/tier_guard.py", root / "libs/v2d_infra")
    rng = np.random.default_rng(seed + 100)
    write_release(root, {1: "tier_1_multiview_caption", 2: "tier_2_synthetic_noise"}[tier], release_pose, vis)
    write_motion(root, tier, motion_pose, rng)
    meta = {"tier": tier, "episode": 0, "sequence_id": SEQ}
    if tier == 2:
        (root / "runs/recovery/tier2_v54").mkdir(parents=True)
        np.savez(root / "runs/recovery/tier2_v54/episode_000000.npz", object_pose=npz_pose, visible=vis)
        meta["object_pose_source"] = {"path": "runs/recovery/tier2_v54/episode_000000.npz"}
    m = root / f"runs/motion_src/tier{tier}/{SEQ}"; m.mkdir(parents=True)
    (m / "v2d_meta.json").write_text(json.dumps(meta))


def run(root: Path, tier: int) -> tuple:
    r = subprocess.run([sys.executable, "stages/evaluate/frame_ref_check.py", "--tier", str(tier)], cwd=root, capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout + r.stderr


def traj(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.concatenate([np.cumsum(rng.normal(scale=0.01, size=(T, 3)), axis=0), rand_quat(rng, T)], axis=1)


def test_tier1() -> None:
    clean = traj(1); vis = np.ones(T, bool); vis[57] = False
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root, 1, clean, clean, vis)
        rc, out = run(root, 1)
        assert rc == 0 and "计分帧 199 / 200" in out and out.rstrip().endswith("FRAME_REF_OK"), out
    shifted = clean.copy(); shifted[120:, 0] += 0.002
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root, 1, shifted, clean, vis)
        rc, out = run(root, 1)
        assert rc == 1 and out.rstrip().endswith("FRAME_REF_FAIL 1"), out


def test_tier2() -> None:
    v54, noisy = traj(2), traj(3); vis = np.ones(T, bool)
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root, 2, v54, noisy, vis, npz_pose=v54)
        rc, out = run(root, 2)
        assert rc == 0 and out.rstrip().endswith("FRAME_REF_OK"), out
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root, 2, noisy, noisy, vis, npz_pose=v54)
        rc, out = run(root, 2)
        assert rc == 1 and out.rstrip().endswith("FRAME_REF_FAIL 1"), out


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("ALL_OK")
