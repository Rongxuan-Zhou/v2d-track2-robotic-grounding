"""diag_phase.py 的测试（暂无契约 ID：运行规范待成文）。要 numpy ≥ 2（np.trapezoid）；Mac 上用 uv，服务器上进 FlashCHORD 容器。

合成一条 120 帧的参考：球形物体（半径 5 cm）第 50–90 帧沿 x 移 0.3 m、绕 z 转 30°；参考右手第 40–100 帧接触，手掌离球面 2 cm。
12 个世界各造一种情形，期望分类：0、1 成功；2、3 A1（接触前物体被推开 5 cm）；4、5 A2（手停在 20 cm 外）；
6、7 G1（手到位后第 52 帧离开）；8、9 G2（手贴着物体，但物体只走参考的 30%）；10、11 K（物体跟着走、偏差慢慢长到 15 cm）。
  1. 分类逐个对上，失败世界里接近、抓取各 40%、跟踪 20%，结论“混合”；summary_json 的计数对
  2. 闸：同步差 1 cm、缺手掌数据、左右手对调、手掌对参考错一帧、同卡 chord_sr 差太多、JSON 的 MP-SR 被改 → 都退出码 3 且不分类
     （tracking_error 的 palm_position_m 按合成数据算，与评测器同义：两只手掌到参考手掌距离的较大者）
  3. 对照：成功世界没有标记 → CONTROL_OK（退出码 0）；把成功世界的手放到 20 cm 外 → CONTROL_FAIL（退出码 4）
  4. 试跑检查 diag2_sync.py：正常 → SYNC_OK；左右手对调、错一帧、同步差 1 cm、缺手掌数据 → SYNC_FAIL（退出码 1）
用法：uv run --no-project --python 3.12 --with numpy python stages/evaluate/test_diag_phase.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "stages/evaluate/diag_phase.py"
T, W = 120, 12
EXPECT = ["S", "S", "A1", "A1", "A2", "A2", "G1", "G1", "G2", "G2", "K", "K"]


def quat_z(theta):
    return np.stack([np.zeros_like(theta), np.zeros_like(theta), np.sin(theta / 2), np.cos(theta / 2)], -1)


def rotmat(q):
    x, y, z, w = (q[..., i] for i in range(4))
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1).reshape(*q.shape[:-1], 3, 3)


def geodesic(a, b):
    dot = np.clip(np.abs(np.sum(a * b, axis=-1)), 0.0, 1.0)
    return 2.0 * np.arctan2(np.sqrt(np.maximum(0.0, 1.0 - dot * dot)), dot)


def build(root: Path, far_success_hand=False, sync=0.0, drop_palms=False, mp_tamper=0.0, chord_same=0.5, auc_same_delta=0.0,
          swap_sides=False, palm_err_shift=0):
    rng = np.random.default_rng(0)
    v = rng.normal(size=(50, 3)); v = 0.05 * v / np.linalg.norm(v, axis=1, keepdims=True)
    t = np.arange(T)
    s = np.clip((t - 50) / 40, 0, 1)
    p0 = np.array([0.0, 0.0, 0.1])
    p_ref = p0 + np.outer(s, [0.3, 0.0, 0.0])
    q_ref = quat_z(np.radians(30.0) * s)
    refp = np.concatenate([p_ref, q_ref], -1)[:, None]                     # (T,1,7)
    contact = np.zeros((T, 2), bool); contact[40:101, 1] = True              # 左手没有接触，右手 40–100
    off = np.array([-0.07, 0.0, 0.0])
    r_palm = np.where((t < 40)[:, None], p0 + off + np.outer(np.clip((40 - t) / 40, 0, 1), [-0.5, 0, 0]), p_ref + off)
    r_palm = np.where((t > 100)[:, None], p_ref + off + [-0.2, 0, 0], r_palm)
    l_palm = np.broadcast_to([-1.0, 0.5, 0.5], (T, 3))
    ref_palm = np.stack([l_palm, r_palm], 1)                                 # (T,2,3)
    ach = np.repeat(np.concatenate([p_ref, q_ref], -1)[:, None], W, 1)       # (T,W,7)
    palm = np.repeat(ref_palm[:, None], W, 1).copy()                         # (T,W,2,3)
    if far_success_hand:
        palm[:, 0:2, 1] += [-0.2, 0, 0]
    ach[35:, 2:4, :3] += [0.05, 0, 0]                                        # A1
    ach[:, 4:6] = np.concatenate([p0, [0, 0, 0, 1]])                         # A2：物体不动
    palm[:, 4:6, 1] = r_palm[:, None] + [-0.2, 0, 0]
    ach[:, 6:8] = np.concatenate([p0, [0, 0, 0, 1]])                         # G1：物体不动，手第 52 帧离开
    palm[52:, 6:8, 1] = r_palm[52:, None] + [-0.2, 0, 0]
    p_g2 = p0 + 0.3 * (p_ref - p0)                                           # G2：物体只走 30%，手贴着它
    ach[:, 8:10, :3] = p_g2[:, None]; ach[:, 8:10, 3:] = quat_z(np.radians(30.0) * s * 0.3)[:, None]
    palm[40:101, 8:10, 1] = (p_g2 + off)[40:101, None]
    p_k = p_ref + np.outer(0.15 * np.clip((t - 50) / 70, 0, 1), [0, 1, 0])  # K：跟着走，y 向偏差长到 15 cm
    ach[:, 10:12, :3] = p_k[:, None]
    palm[40:101, 10:12, 1] = (p_k + off)[40:101, None]
    ach = ach[:, :, None].astype(np.float32)                                 # (T,W,1,7)，评测器记的是 float32
    a64 = ach.astype(np.float64)
    position = np.linalg.norm(a64[..., :3] - refp[:, None, :, :3], axis=-1).mean(0)
    orientation = geodesic(a64[..., 3:], refp[:, None, :, 3:]).mean(0)
    mp = ((position[:, 0] < 0.03) & (np.degrees(orientation[:, 0]) < 30)).mean()
    sp = ((position.mean(-1) <= 0.1) & (orientation.mean(-1) <= 0.5)).mean()
    vw_a = np.einsum("twij,vj->twvi", rotmat(a64[:, :, 0, 3:]), v) + a64[:, :, 0, None, :3]
    vw_r = np.einsum("tij,vj->tvi", rotmat(refp[:, 0, 3:]), v) + refp[:, 0, None, :3]
    add = np.linalg.norm(vw_a - vw_r[:, None], axis=-1).mean(-1)
    auc = float(np.trapezoid([np.mean(add < th) for th in np.arange(0.01, 0.10, 0.01)], x=np.linspace(0, 1, 9)))
    te = np.zeros((T, W, 6), np.float32)
    rp32, pl32 = ref_palm.astype(np.float32).astype(np.float64), palm.astype(np.float32).astype(np.float64)
    rp_shift = np.roll(rp32, -palm_err_shift, axis=0) if palm_err_shift else rp32
    te[..., 2] = np.linalg.norm(pl32 - rp_shift[:, None], axis=-1).max(-1)
    arrays = dict(achieved_pose_w=ach, reference_pose_w=refp.astype(np.float64), tracking_error=te,
                  tracking_error_names=np.array(["pelvis_position_m", "pelvis_orientation_rad", "palm_position_m", "palm_orientation_rad",
                                                 "object_position_m", "object_orientation_rad"]),
                  add=add, add_static=add[:, 0], ref_palm_pos=ref_palm.astype(np.float32), ref_palm_mesh_dist=np.zeros((T, 2), np.float32),
                  object_vertices_o=v[None].astype(np.float32), ref_palm_names=np.array(["left_hand_palm_link", "right_hand_palm_link"]),
                  contact_sides=np.array(["left", "right"]), contact_active=contact, non_finite_worlds=np.array([], np.int64),
                  policy=np.array("/repo/runs/flash_chord/x/policy_1.safetensors"))
    if not drop_palms:
        arrays.update(robot_palm_pos=palm.astype(np.float32), robot_palm_sides=np.array(["right", "left"] if swap_sides else ["left", "right"]),
                      palm_object_sync_max_abs_m=np.float64(sync))
    np.savez_compressed(root / "r.npz", **arrays)
    metrics = {"add_auc": auc, "chord_sr": 0.5, "maniptrans_sr": float(mp) + mp_tamper, "spider_sr_uncentered": float(sp), "mppe_cm": 1.0}
    (root / "d.json").write_text(json.dumps({"world_count": W, "step_count": T, "checkpoint_sha256": "abc", "non_finite_worlds": [], "metrics": metrics}))
    same = dict(metrics, maniptrans_sr=float(mp), chord_sr=chord_same, add_auc=auc + auc_same_delta)
    (root / "s.json").write_text(json.dumps({"world_count": 4096, "step_count": T, "checkpoint_sha256": "abc", "metrics": same}))


def run(root: Path, role="main") -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), str(root / "r.npz"), "--diag-json", str(root / "d.json"),
                           "--same-card-json", str(root / "s.json"), "--role", role], capture_output=True, text=True, timeout=300)


def summary(out: str) -> dict:
    line = next(l for l in out.splitlines() if l.startswith("summary_json "))
    return json.loads(line[len("summary_json "):])


def test_classification() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); build(root)
        r = run(root)
        assert r.returncode == 0, r.stdout + r.stderr
        s = summary(r.stdout)
        want = {c: EXPECT.count(c) for c in ("S", "A1", "A2", "G1", "G2", "K")}
        assert s["counts"] == want, (s["counts"], r.stdout)
        assert s["t_c"] == 40 and s["t_last"] == 100 and s["t_m"] == 53 and s["active"] == ["right"], s
        assert abs(s["group_shares"]["接近"] - 0.4) < 1e-9 and abs(s["group_shares"]["抓取"] - 0.4) < 1e-9 and abs(s["group_shares"]["跟踪"] - 0.2) < 1e-9, s
        assert s["verdict"].startswith("混合"), s["verdict"]
        for c in ("A1", "A2", "G1", "G2", "K"):
            assert set(s["picks"][c]) <= {i for i, e in enumerate(EXPECT) if e == c}, (c, s["picks"])
        assert "VALIDITY PASS" in r.stdout and "V1 同卡" in r.stdout and "PASS  V0" in r.stdout and "手掌坐标帧闸：与评测器 palm_position_m 最大差" in r.stdout, r.stdout
        assert "失败世界 10 个：只有位置不过" in r.stdout and "A2 世界 2 个" in r.stdout and "A1 世界 2 个" in r.stdout and "K 世界 2 个" in r.stdout, r.stdout


def test_gates() -> None:
    for kw, word in (({"sync": 0.01}, "V0"), ({"drop_palms": True}, "V0"), ({"swap_sides": True}, "V0"), ({"palm_err_shift": 1}, "V0"),
                     ({"chord_same": 0.9, "auc_same_delta": 0.1}, "V1"), ({"mp_tamper": 1 / 12}, "V2")):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build(root, **kw)
            r = run(root)
            assert r.returncode == 3 and f"FAIL  {word}" in r.stdout and "VALIDITY FAIL" in r.stdout and "summary_json" not in r.stdout, (kw, r.stdout + r.stderr)


def test_control() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); build(root)
        r = run(root, "control")
        assert r.returncode == 0 and "CONTROL_OK" in r.stdout and summary(r.stdout)["control_false_rate"] == 0.0, r.stdout + r.stderr
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); build(root, far_success_hand=True)
        r = run(root, "control")
        assert r.returncode == 4 and "CONTROL_FAIL" in r.stdout and summary(r.stdout)["control_false_rate"] == 1.0, r.stdout + r.stderr


def test_sync() -> None:
    sync = REPO / "stages/evaluate/diag2_sync.py"
    for kw, want in (({}, 0), ({"swap_sides": True}, 1), ({"palm_err_shift": 1}, 1), ({"sync": 0.01}, 1), ({"drop_palms": True}, 1)):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t); build(root, **kw)
            r = subprocess.run([sys.executable, str(sync), str(root / "r.npz")], capture_output=True, text=True, timeout=120)
            last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
            assert r.returncode == want and last.startswith("SYNC_OK" if want == 0 else "SYNC_FAIL"), (kw, r.stdout + r.stderr)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"ok  {name}")
    print("ALL_OK")
