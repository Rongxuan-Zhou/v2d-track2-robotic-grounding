"""rotate_frefs_t3.py 的测试（暂无契约 ID：运行规范待成文；用途见 agent-doc/experiments/2026-10-16-t3-world-frame.md 事实六）。要 numpy、pyarrow；
装了 pandas 时再用 kit 打包器自己的函数核“frame refs 整体转 R，打包出的轨迹也整体转 R”。

临时目录里放两份 manifest（mf4090：ep2、ep5；ml40：ep7），npz 按 make_frame_refs_t3.py 的格式写（四元数 w ≥ 0，含转角 179° 的帧）。
  1. 正常：R 取 results/t3world/k_budget.txt 里方格 G 的 R*（转角 4.69°）。新 npz 的键、frame_index 与源相同；位置等于 R p、朝向矩阵等于 R M
     （差 ≤ 1e-9），w ≥ 0；manifest 的表头与源相同、除 sha256 外逐列相同、sha256 等于新文件；frefs_t3_match.manifest_rows 能读；ROTATION.json
     记了 R、转角、label、源与新 npz 的 sha256；
  2. 单位阵：新位姿与源相同（位置差 ≤ 1e-12，四元数差 ≤ 1e-12）；
  3. 拒绝（退出码非 0、不建目标目录）：反射（det = −1）；转角 15°；源 npz 被改过（sha256 与 manifest 不符）；label 带空格；目标目录已存在（不动它）；
  4. 有 pandas 时：评测器参考 = 对源 frame refs 做一个刚体变换；rollout = 评测器参考加小扰动。kit 的 convert_track2_frame 用新、旧 frame refs
     各换算一次，新结果的位置等于 R × 旧结果的位置、朝向矩阵等于 R × 旧结果的朝向（差 ≤ 1e-9）。
用法：uv run --no-project --python 3.12 --with numpy --with pyarrow --with pandas python stages/evaluate/test_rotate_frefs_t3.py
最后一行 ALL_OK。
"""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "stages/evaluate"))
from frefs_t3_match import COLS, manifest_rows  # noqa: E402
from make_frame_refs_t3 import FRAME, mat_to_xyzw, xyzw_to_mat  # noqa: E402

KIT = REPO / "third_party/v2d_submission_kit/update_0f2f9f4b/v2d_submission_kit"
R_G = np.array([[0.99692, 0.02714, 0.07351], [-0.0254, 0.99938, -0.02452], [-0.07413, 0.02257, 0.99699]])   # k_budget.txt，方格 G
ROWS = {"mf4090": (2, 5), "ml40": (7,)}


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def rand_mats(T: int, rng) -> np.ndarray:
    q = rng.normal(size=(T, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 3] < 0] *= -1
    M = xyzw_to_mat(q)
    for i, axis in enumerate(np.eye(3)):   # 转角 179°：迹为负
        a = np.radians(179.0)
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        M[i] = np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K
    return M


def setup(root: Path, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    src = root / "src"
    src.mkdir()
    data = {}
    for mach, eps in ROWS.items():
        lines = ["\t".join(COLS)]
        for ep in eps:
            T = 20 + ep
            M = rand_mats(T, rng)
            p = rng.normal(size=(T, 3)) * [1.0, 0.5, 2.0] + [0, 0.7, 3.0]
            pose = np.concatenate([p, mat_to_xyzw(M)], axis=1)[:, None, :]
            f = src / f"episode_{ep:06d}.npz"
            np.savez(f, episode_index=np.int64(ep), coordinate_frame=np.array(FRAME), frame_index=np.arange(T), pose_xyzw=pose)
            data[ep] = (p, M)
            lines.append("\t".join([str(ep), f"t3-ep{ep}-run", "349999104", "tier3_hm", f"seq{ep}", str(T), "a" * 64, "b" * 64, f.name, sha256(f)]))
        (src / f"manifest_{mach}.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return data


def rot_json(root: Path, R, label="gridG", name="rot.json") -> Path:
    f = root / name
    f.write_text(json.dumps({"rotation": np.asarray(R).tolist(), "label": label, "source": "test"}), encoding="utf-8")
    return f


def run(root: Path, rot: Path, out: str = "out") -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(REPO / "stages/evaluate/rotate_frefs_t3.py"), str(root / "src"), str(rot), "--out", str(root / out)],
                          capture_output=True, text=True)


def read(out: Path, ep: int):
    with np.load(out / f"episode_{ep:06d}.npz", allow_pickle=False) as z:
        return sorted(z.files), z["frame_index"].copy(), z["pose_xyzw"][:, 0].copy()


def test_normal() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        data = setup(root)
        U, _, Vt = np.linalg.svd(R_G)
        R = U @ Vt
        r = run(root, rot_json(root, R_G))
        assert r.returncode == 0 and r.stdout.strip().endswith("ROTATE_FREFS_OK 3"), r.stdout + r.stderr
        out = root / "out"
        for ep, (p, M) in data.items():
            keys, fi, pose = read(out, ep)
            assert keys == ["coordinate_frame", "episode_index", "frame_index", "pose_xyzw"] and np.array_equal(fi, np.arange(len(p)))
            assert np.abs(pose[:, :3] - p @ R.T).max() <= 1e-9, ep
            assert np.abs(xyzw_to_mat(pose[:, 3:]) - np.einsum("ij,njk->nik", R, M)).max() <= 1e-9, ep
            assert (pose[:, 6] >= 0).all(), ep
        for mach in ROWS:
            a = (root / "src" / f"manifest_{mach}.tsv").read_text(encoding="utf-8").splitlines()
            b = (out / f"manifest_{mach}.tsv").read_text(encoding="utf-8").splitlines()
            assert a[0] == b[0] and len(a) == len(b)
            for la, lb in zip(a[1:], b[1:]):
                ra, rb = dict(zip(COLS, la.split("\t"))), dict(zip(COLS, lb.split("\t")))
                assert all(ra[c] == rb[c] for c in COLS if c != "sha256") and rb["sha256"] == sha256(out / rb["file"]), (ra, rb)
        rows = manifest_rows(out)
        assert sorted(rows) == [2, 5, 7]
        rec = json.loads((out / "ROTATION.json").read_text(encoding="utf-8"))
        assert rec["label"] == "gridG" and abs(rec["angle_deg"] - 4.69) < 0.01 and np.abs(np.array(rec["rotation"]) - R).max() < 1e-12
        assert all(rec["episodes"][str(ep)]["sha256"] == sha256(out / f"episode_{ep:06d}.npz") for ep in data)
    print("  1. 正常：位置 = R p、朝向 = R M、manifest 只换 sha256、ROTATION.json 齐全")


def test_identity() -> None:
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        data = setup(root, seed=1)
        r = run(root, rot_json(root, np.eye(3)))
        assert r.returncode == 0, r.stdout + r.stderr
        for ep in data:
            _, _, a = read(root / "src", ep)
            _, _, b = read(root / "out", ep)
            assert np.abs(a - b).max() <= 1e-12, ep
    print("  2. 单位阵：位姿不变")


def test_refuse() -> None:
    cases = {"反射": np.diag([1.0, 1.0, -1.0]),
             "转角 15°": np.array([[np.cos(np.radians(15)), -np.sin(np.radians(15)), 0], [np.sin(np.radians(15)), np.cos(np.radians(15)), 0], [0, 0, 1.0]])}
    for name, R in cases.items():
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            setup(root)
            r = run(root, rot_json(root, R))
            assert r.returncode != 0 and not (root / "out").exists(), (name, r.stdout)
    with tempfile.TemporaryDirectory() as d:   # 源 npz 被改过
        root = Path(d)
        setup(root)
        f = root / "src/episode_000005.npz"
        with np.load(f) as z:
            pose = z["pose_xyzw"].copy()
        pose[0, 0, 0] += 0.01
        np.savez(f, episode_index=np.int64(5), coordinate_frame=np.array(FRAME), frame_index=np.arange(len(pose)), pose_xyzw=pose)
        r = run(root, rot_json(root, R_G))
        assert r.returncode != 0 and not (root / "out").exists(), r.stdout
    with tempfile.TemporaryDirectory() as d:   # label 带空格
        root = Path(d)
        setup(root)
        r = run(root, rot_json(root, R_G, label="grid G"))
        assert r.returncode != 0 and not (root / "out").exists(), r.stdout
    with tempfile.TemporaryDirectory() as d:   # 目标目录已存在
        root = Path(d)
        setup(root)
        (root / "out").mkdir()
        (root / "out/keep").write_text("x")
        r = run(root, rot_json(root, R_G))
        assert r.returncode != 0 and (root / "out/keep").read_text() == "x" and len(list((root / "out").iterdir())) == 1, r.stdout
    print("  3. 拒绝：反射、转角 15°、源 npz 被改过、label 带空格、目标目录已存在")


def test_kit() -> None:
    try:
        import pandas  # noqa: F401
    except ImportError:
        print("  4. （没有 pandas：跳过 kit 的换算核对）")
        return
    sys.path.insert(0, str(KIT / "tools"))
    sys.path.insert(0, str(KIT))
    import pack_submission as ps
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        data = setup(root, seed=2)
        U, _, Vt = np.linalg.svd(R_G)
        R = U @ Vt
        assert run(root, rot_json(root, R_G)).returncode == 0
        rng = np.random.default_rng(3)
        a, b = np.radians(30.0), np.radians(20.0)
        Rg = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]]) @ \
            np.array([[1.0, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
        tg = np.array([1.0, 2.0, 0.5])
        for ep, (p, M) in data.items():
            n = len(p)
            ref0 = ps.input_frame_reference(root / "src", ep, np.arange(n), 1, ["object"])
            ref1 = ps.input_frame_reference(root / "out", ep, np.arange(n), 1, ["object"])
            ev = np.concatenate([p @ Rg.T + tg, mat_to_xyzw(np.einsum("ij,njk->nik", Rg, M))], axis=1)   # 评测器参考：源的刚体变换
            dR = np.stack([xyzw_to_mat(rng.normal(0, 0.02, (1, 4)) + [0, 0, 0, 1.0])[0] for _ in range(n)])
            Ma = np.einsum("nij,njk->nik", dR, xyzw_to_mat(ev[:, 3:]))
            ach = np.concatenate([ev[:, :3] + rng.normal(0, 0.03, (n, 3)), mat_to_xyzw(Ma)], axis=1)   # rollout：评测器参考加扰动
            c0 = ps.convert_track2_frame(ach, ev, ref0)[0]
            c1 = ps.convert_track2_frame(ach, ev, ref1)[0]
            assert np.abs(c1[:, :3] - c0[:, :3] @ R.T).max() <= 1e-9, ep
            assert np.abs(xyzw_to_mat(c1[:, 3:]) - np.einsum("ij,njk->nik", R, xyzw_to_mat(c0[:, 3:]))).max() <= 1e-9, ep
    print("  4. kit 的 convert_track2_frame：frame refs 转 R，换算出的 rollout 位置与朝向都恰好转 R")


if __name__ == "__main__":
    test_normal()
    test_identity()
    test_refuse()
    test_kit()
    print("ALL_OK")
