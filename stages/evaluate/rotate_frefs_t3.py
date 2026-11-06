"""Tier 3 帧参考整体转动：把 frame refs 的位姿绕世界原点（= 主机位相机中心）转一个 R，另存一个目录（Mac 上跑，要 numpy、pyarrow；
只读帧参考目录与转动文件，不读 Tier 1、Tier 2；进程开头 forbid_tier1()）。

为什么：官方打包器用“评测器记录的参考 ↔ frame refs”之间的一个刚体变换，把仿真里的物体位姿转到发布世界系（kit 的
tracks/track_2/README.md “Input references for frame conversion”）。frame refs 整体转 R，打包出的轨迹也整体绕原点转 R，不用重训
（test_rotate_frefs_t3.py 用 kit 自己的 convert_track2_frame 核这条性质）。用途：Tier 3 的内参 A/B。重建流程的内参（fx 830、主点居中）
与标定的内参不同，二者之差的大部分是绕相机中心的刚体转动 R*（stages/recovery/t3_k_budget.py，results/t3world/k_budget.txt）。

做法：
  - 读 <源目录>/manifest_*.tsv（列同 frefs_t3_match.COLS）与其中列出的 episode_*.npz；每个 npz 的 sha256 先与 manifest 核对，不一致就拒绝；
    npz 的键必须正好是 coordinate_frame（= released_track2_world）、episode_index、frame_index、pose_xyzw [T,1,7]；
  - 转动文件是 JSON：rotation（3×3，按行）、label（短名，只含字母数字与 _-）、source（出处）。R 必须是转动（‖RᵀR − I‖ ≤ 1e-4、det > 0，
    再用 SVD 投到最近的转动），转角 ≤ 10°（防误用）；
  - 位置 p' = R p，朝向矩阵 M' = R M（四元数 xyzw 与矩阵的互换用 make_frame_refs_t3.py 的 xyzw_to_mat / mat_to_xyzw，w ≥ 0）；
  - 写 <目标目录>/episode_*.npz（键与源相同），读回核对：位置与 R p 差 ≤ 1e-9 m、朝向矩阵与 R M 差 ≤ 1e-9、frame_index 与源逐个相同；
  - 写 <目标目录>/manifest_<机器>.tsv：列与源相同（frefs_t3_match.py 要求表头恰好是那 10 列），只把 sha256 换成新 npz 的；
  - 写 <目标目录>/ROTATION.json：R、转角、label、source、源目录、源 manifest 的 sha256、每集源 npz 与新 npz 的 sha256。
目标目录已存在就拒绝。
用法：uv run --no-project --python 3.12 --with numpy --with pyarrow python stages/evaluate/rotate_frefs_t3.py <源目录> <转动 json> --out <目标目录>
最后一行 ROTATE_FREFS_OK n（n 是集数）。测试：stages/evaluate/test_rotate_frefs_t3.py。
"""
import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frefs_t3_match import COLS  # noqa: E402
from make_frame_refs_t3 import FRAME, mat_to_xyzw, xyzw_to_mat  # noqa: E402

MAX_DEG = 10.0
KEYS = ["coordinate_frame", "episode_index", "frame_index", "pose_xyzw"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rotation(path: Path) -> tuple:
    spec = json.loads(path.read_text(encoding="utf-8"))
    R = np.asarray(spec["rotation"], dtype=np.float64)
    label, source = str(spec["label"]), str(spec["source"])
    if R.shape != (3, 3) or not np.isfinite(R).all():
        raise SystemExit(f"{path}：rotation 不是有限的 3×3")
    if np.abs(R.T @ R - np.eye(3)).max() > 1e-4 or np.linalg.det(R) <= 0:   # 防反射、缩放；打印到 5 位小数的 R 偏 1e-5，之后投到最近的转动
        raise SystemExit(f"{path}：rotation 不是转动（RᵀR 偏离单位阵或 det ≤ 0）")
    U, _, Vt = np.linalg.svd(R)   # 投到最近的转动，去掉打印精度带来的 1e-6 级误差
    R = U @ Vt
    ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))
    if ang > MAX_DEG:
        raise SystemExit(f"{path}：转角 {ang:.2f}° 超过 {MAX_DEG}°")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise SystemExit(f"{path}：label 只能含字母、数字、_ 与 -")
    return R, ang, label, source


def manifests(src: Path) -> list:
    out = []
    for m in sorted(src.glob("manifest_*.tsv")):
        lines = m.read_text(encoding="utf-8").splitlines()
        if not lines or lines[0].split("\t") != COLS:
            raise SystemExit(f"{m}：表头不是 {COLS}")
        out.append((m, [dict(zip(COLS, ln.split("\t"))) for ln in lines[1:] if ln.strip()]))
    if not out:
        raise SystemExit(f"{src}：没有 manifest_*.tsv")
    return out


def rotate_one(src_npz: Path, dst_npz: Path, R: np.ndarray, ep: int) -> None:
    with np.load(src_npz, allow_pickle=False) as z:
        if sorted(z.files) != KEYS:
            raise SystemExit(f"{src_npz}：键 {sorted(z.files)} 不是 {KEYS}")
        if str(z["coordinate_frame"].item()) != FRAME or int(z["episode_index"].item()) != ep:
            raise SystemExit(f"{src_npz}：coordinate_frame 或 episode_index 不对")
        fi, pose = z["frame_index"].copy(), z["pose_xyzw"].copy()
    if pose.ndim != 3 or pose.shape[1:] != (1, 7) or len(fi) != len(pose) or not np.isfinite(pose).all():
        raise SystemExit(f"{src_npz}：pose_xyzw 形状 {pose.shape} 不是 [T,1,7] 或有非有限值")
    p, M = pose[:, 0, :3], xyzw_to_mat(pose[:, 0, 3:])
    p2, M2 = p @ R.T, np.einsum("ij,njk->nik", R, M)
    out = np.concatenate([p2, mat_to_xyzw(M2)], axis=1)[:, None, :]
    np.savez(dst_npz, episode_index=np.int64(ep), coordinate_frame=np.array(FRAME), frame_index=fi, pose_xyzw=out)
    with np.load(dst_npz, allow_pickle=False) as z:   # 读回核对
        ok = (sorted(z.files) == KEYS and np.array_equal(z["frame_index"], fi)
              and np.abs(z["pose_xyzw"][:, 0, :3] - p2).max() <= 1e-9
              and np.abs(xyzw_to_mat(z["pose_xyzw"][:, 0, 3:]) - M2).max() <= 1e-9)
    if not ok:
        dst_npz.unlink()
        raise SystemExit(f"{dst_npz}：读回核对不过")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("rotation", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "libs/v2d_infra"))
    from tier_guard import forbid_tier1
    forbid_tier1()   # 档位隔离：此后读 Tier 1 的任何文件都会抛错
    R, ang, label, source = load_rotation(a.rotation)
    if a.out.exists():
        raise SystemExit(f"{a.out} 已存在")
    mans = manifests(a.src)
    for m, rows in mans:   # 先全部核源，再动手
        for r in rows:
            f = a.src / r["file"]
            if r["file"] != f"episode_{int(r['episode']):06d}.npz" or not f.is_file() or sha256(f) != r["sha256"]:
                raise SystemExit(f"{m.name} ep{r['episode']}：源 npz 缺失、文件名不对或 sha256 与 manifest 不符")
    a.out.mkdir(parents=True)
    record = {"label": label, "source": source, "angle_deg": ang, "rotation": R.tolist(), "src_dir": str(a.src),
              "src_manifests": {}, "episodes": {}}
    n = 0
    for m, rows in mans:
        record["src_manifests"][m.name] = sha256(m)
        new_rows = []
        for r in rows:
            ep = int(r["episode"])
            dst = a.out / r["file"]
            rotate_one(a.src / r["file"], dst, R, ep)
            record["episodes"][str(ep)] = {"src_sha256": r["sha256"], "sha256": sha256(dst)}
            new_rows.append({**r, "sha256": record["episodes"][str(ep)]["sha256"]})
            n += 1
        (a.out / m.name).write_text("\t".join(COLS) + "\n" + "".join("\t".join(row[c] for c in COLS) + "\n" for row in new_rows),
                                    encoding="utf-8")
    (a.out / "ROTATION.json").write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{label}：绕世界原点转 {ang:.3f}°（{source}），{n} 集 → {a.out}")
    print(f"ROTATE_FREFS_OK {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
