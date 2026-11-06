"""上传前核对：转动 frame refs 后重新打包的 Tier 3 CSV，是不是恰好是原 CSV 绕世界原点转 R（Mac 上跑，要 numpy；只读两个 CSV 与 ROTATION.json）。

逐行（按 row_id 对齐）要求：
  - 两个 CSV 的 row_id 集合相同、行数相同；
  - 位置 p_新 = R p_原（CSV 只留 6 位小数，差 ≤ 5e-6 m）；朝向矩阵 M_新 = R M_原（四元数 xyzw，换成矩阵后差 ≤ 1e-5）；
  - 溯源列 eval_script_sha256、eval_code_sha256、checkpoint_sha256、packer_script_sha256、uploader_script_sha256 逐行相同（同一批 rollout、同一个 kit）；
    code_commit_url 可以不同（打包时的 HEAD）。
用法：uv run --no-project --python 3.12 --with numpy python stages/evaluate/csv_rotation_check.py <原 CSV> <新 CSV> <frame refs 目录里的 ROTATION.json>
最后一行 CSV_ROTATION_OK 行数 或 CSV_ROTATION_FAIL，退出码 0 / 1。
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

SAME = ["eval_script_sha256", "eval_code_sha256", "checkpoint_sha256", "packer_script_sha256", "uploader_script_sha256"]


def quat_mat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = (q / np.linalg.norm(q, axis=1, keepdims=True)).T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
        np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
        np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)


def load(path: Path) -> dict:
    with path.open(newline="") as f:
        return {r["row_id"]: r for r in csv.DictReader(f)}


def main() -> int:
    base, new, rot = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    R = np.asarray(json.loads(rot.read_text(encoding="utf-8"))["rotation"], float)
    a, b = load(base), load(new)
    if set(a) != set(b):
        print(f"row_id 不一致：只在原 CSV {len(set(a) - set(b))} 行，只在新 CSV {len(set(b) - set(a))} 行")
        print("CSV_ROTATION_FAIL")
        return 1
    ids = sorted(a)
    pa = np.array([[float(a[i][k]) for k in ("pos_x", "pos_y", "pos_z")] for i in ids])
    pb = np.array([[float(b[i][k]) for k in ("pos_x", "pos_y", "pos_z")] for i in ids])
    qa = np.array([[float(a[i][k]) for k in ("quat_x", "quat_y", "quat_z", "quat_w")] for i in ids])
    qb = np.array([[float(b[i][k]) for k in ("quat_x", "quat_y", "quat_z", "quat_w")] for i in ids])
    dp = float(np.abs(pb - pa @ R.T).max())
    dm = float(np.abs(quat_mat(qb) - np.einsum("ij,njk->nik", R, quat_mat(qa))).max())
    diff = [c for c in SAME if any(a[i][c] != b[i][c] for i in ids)]
    moved = np.linalg.norm(pb - pa, axis=1)
    print(f"{len(ids)} 行；位置与 R p_原 的最大差 {dp:.2e} m，朝向矩阵最大差 {dm:.2e}；溯源列不同的：{diff or '无'}；"
          f"转动带来的位置变化中位 {100 * np.median(moved):.1f} cm、90% 分位 {100 * np.percentile(moved, 90):.1f} cm")
    ok = dp <= 5e-6 and dm <= 1e-5 and not diff
    print(f"CSV_ROTATION_OK {len(ids)}" if ok else "CSV_ROTATION_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
