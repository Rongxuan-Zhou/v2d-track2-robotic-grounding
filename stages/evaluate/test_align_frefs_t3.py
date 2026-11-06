"""stages/evaluate/tier1_frame0.py 与 align_frefs_t3.py 的测试：守计划 agent-doc/plan-t3align-2026-10-21.md 判据 A0、A1 与复核改动第 1、3、5 条。
合成数据：已知的世界系误差 W（偏航 20°、倾斜 1°、平移），我们的物体轨迹 = W⁻¹ ∘ 真值，人体根同样；对齐后应逐帧还原真值。
用法：python3 stages/evaluate/test_align_frefs_t3.py（要 numpy、pyarrow；训练机上经 scripts/dev/in-flash-chord）。最后一行 ALIGN_FREFS_TESTS_OK。
"""
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import align_frefs_t3 as A  # noqa: E402
import tier1_frame0  # noqa: E402

UP = np.array([0.0, -1.0, 0.0])   # OpenCV 世界系里“上”大致是 −y
SEQ = "2026-01-01_00-00-00_box_test_01"


def _rot(axis, deg):
    return A.rotvec_to_R(np.asarray(axis, float) / np.linalg.norm(axis) * math.radians(deg))


def _truth_traj(T=60):
    """物体前 20 帧静止，之后平移加转动；人体根随便走。"""
    P = np.zeros((T, 7))
    for t in range(T):
        s = max(0, t - 20) / 40
        P[t, :3] = [0.3 + 0.5 * s, 0.9, 2.5 + 0.2 * s]
        P[t, 3:] = A.R_to_q(_rot([0, 1, 0], 30 + 40 * s) @ _rot([1, 0, 0], 5))
    return P


def _apply_inv(P, R, t):
    Ri = R.T
    return A.apply(P, Ri, -Ri @ t)


def _write_tier1(root: Path, P, hum_t, hum_R, dup_frame0=False):
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta").mkdir()
    (root / "meta" / "episodes_metadata.jsonl").write_text(json.dumps({"episode_index": 0, "sequence_id": SEQ}) + "\n")
    rows = []
    for t in range(len(P)):
        q = P[t, 3:]
        pose = [*P[t, :3], q[3], q[0], q[1], q[2]]   # Tier 1 存 [x y z qw qx qy qz]
        rv = np.zeros(231)
        th = math.acos(max(-1, min(1, (np.trace(hum_R[t]) - 1) / 2)))
        if th > 1e-9:
            w = np.array([hum_R[t][2, 1] - hum_R[t][1, 2], hum_R[t][0, 2] - hum_R[t][2, 0], hum_R[t][1, 0] - hum_R[t][0, 1]]) / (2 * math.sin(th))
            rv[:3] = w * th
        rows.append({"frame_index": t, "observation.object.pose": [float(x) for x in pose], "observation.object.visible": [True],
                     "observation.human.translation": [float(x) for x in hum_t[t]], "observation.human.pose": [float(x) for x in rv]})
    if dup_frame0:
        rows.append(dict(rows[0]))
    pq.write_table(pa.Table.from_pylist(rows), root / "data" / "chunk-000" / "episode_000000.parquet")


def _setup(d: Path, our_obj_yaw_err=0.0):
    W_R = _rot(UP, 20.0) @ _rot([1, 0, 0], 1.0)
    W_t = np.array([0.15, -0.02, 0.30])
    truth = _truth_traj()
    T = len(truth)
    hum_t = np.stack([[0.0 + 0.01 * t, 0.95, 2.0] for t in range(T)])
    hum_R = [_rot(UP, 10.0 + t) for t in range(T)]
    _write_tier1(d / "t1", truth, hum_t, hum_R)
    ours = _apply_inv(truth, W_R, W_t)
    if our_obj_yaw_err:   # 物体自己的朝向估错（对称物体），世界系误差不变
        Rerr = _rot(UP, our_obj_yaw_err)
        for t in range(T):
            ours[t, 3:] = A.R_to_q(A.q_to_R(ours[t, 3:]) @ Rerr)
    ours_hum_t = (hum_t - W_t) @ W_R
    ours_hum_R = [W_R.T @ R for R in hum_R]
    root = d / "tier3" / "root_x"
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta").mkdir()
    (root / "ground_plane").mkdir()
    plane = np.r_[W_R.T @ UP, 1.0]   # 我们世界系里的地面法向
    (root / "ground_plane" / f"{SEQ}.json").write_text(json.dumps({"plane": plane.tolist()}))
    (root / "meta" / "episodes_metadata.jsonl").write_text(json.dumps({"episode_index": 0, "sequence_id": SEQ, "ground_plane": f"ground_plane/{SEQ}.json"}) + "\n")
    rows = []
    for t in range(T):
        rv = np.zeros(231)
        R = ours_hum_R[t]
        th = math.acos(max(-1, min(1, (np.trace(R) - 1) / 2)))
        if th > 1e-9:
            rv[:3] = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / (2 * math.sin(th)) * th
        rows.append({"frame_index": t, "observation.object.visible": [True], "observation.human.translation": [float(x) for x in ours_hum_t[t]],
                     "observation.human.pose": [float(x) for x in rv]})
    f = root / "data" / "chunk-000" / "episode_000000.parquet"
    pq.write_table(pa.Table.from_pylist(rows), f)
    import hashlib
    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    fr = d / "frefs"
    fr.mkdir()
    np.savez(fr / "episode_000000.npz", episode_index=np.int64(0), coordinate_frame=np.array("released_track2_world"),
             frame_index=np.arange(T), pose_xyzw=ours[:, None, :].astype(np.float64))
    (fr / "manifest_m.tsv").write_text("\t".join(A.MCOLS) + "\n" + "\t".join(["0", "run", "1", "tier3_x", SEQ, str(T), "p", sha, "episode_000000.npz", "s"]) + "\n")
    return truth


def _run(d: Path, *extra):
    return subprocess.run([sys.executable, str(HERE / "align_frefs_t3.py"), str(d / "frefs"), str(d / "out"), "--tier1", str(d / "t1"), "--t3", str(d / "tier3"), *extra],
                          capture_output=True, text=True, check=False)


def test_reader_frame0_only_and_row_count():
    """A0：读取口只给第 0 帧（位姿顺序 wxyz → xyzw 转对）；第 0 帧有两行时报错。"""
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        truth = _truth_traj()
        hum = np.zeros((len(truth), 3))
        _write_tier1(d / "a", truth, hum, [np.eye(3)] * len(truth))
        r = tier1_frame0.read(d / "a", 0)
        assert np.allclose(r["p"], truth[0, :3]) and np.allclose(r["q_xyzw"], truth[0, 3:]), r
        _write_tier1(d / "b", truth, hum, [np.eye(3)] * len(truth), dup_frame0=True)
        try:
            tier1_frame0.read(d / "b", 0)
            raise AssertionError("两行第 0 帧应报错")
        except SystemExit:
            pass


def test_auto_obj_recovers_every_frame():
    """A1：物体与人体都只有世界系误差时选 obj，新帧参考逐帧还原真值（≤ 1e-6）。"""
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        truth = _setup(d)
        p = _run(d)
        assert p.returncode == 0, (p.stdout, p.stderr)
        rep = json.loads((d / "out" / "ALIGN.json").read_text())["episodes"]["0"]
        assert rep["choice"] == "obj", rep
        new = np.load(d / "out" / "episode_000000.npz")["pose_xyzw"][:, 0, :]
        assert np.abs(new[:, :3] - truth[:, :3]).max() < 1e-6
        assert max(A.angle_deg(A.q_to_R(a) @ A.q_to_R(b).T) for a, b in zip(new[:, 3:], truth[:, 3:])) < 1e-4
        assert (d / "out" / "manifest_aligned.tsv").read_text().count("\n") == 2


def test_symmetric_object_falls_back_to_human_yaw():
    """复核第 3 条：物体朝向自己错了 90°（对称），人体给的偏航一致于世界——选 yaw_hum，位置逐帧仍还原（物体不转的段）。"""
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        truth = _setup(d, our_obj_yaw_err=90.0)
        p = _run(d)
        assert p.returncode == 0, (p.stdout, p.stderr)
        rep = json.loads((d / "out" / "ALIGN.json").read_text())["episodes"]["0"]
        assert rep["choice"] == "yaw_hum", rep
        new = np.load(d / "out" / "episode_000000.npz")["pose_xyzw"][:, 0, :]
        assert np.linalg.norm(new[0, :3] - truth[0, :3]) < 1e-6
        assert np.abs(new[:, :3] - truth[:, :3]).max() < 0.02, np.abs(new[:, :3] - truth[:, :3]).max()   # 只剩 1° 倾斜的残差


def test_swapped_quaternion_is_caught():
    """复核第 1 条的负对照：按 xyzw 读错 Tier 1 四元数，obj 方案下新第 0 帧朝向与真值差得远（端到端核对会抓到）。"""
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        truth = _setup(d)
        p = _run(d, "--variant", "obj", "--swap-quat")
        assert p.returncode == 0, (p.stdout, p.stderr)
        new = np.load(d / "out" / "episode_000000.npz")["pose_xyzw"][0, 0, :]
        assert A.angle_deg(A.q_to_R(new[3:]) @ A.q_to_R(truth[0, 3:]).T) > 5.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name} 过")
    print("ALIGN_FREFS_TESTS_OK")
