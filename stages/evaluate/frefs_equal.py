"""两份 Tier 3 帧参考目录逐集比内容（要 numpy）。计划 agent-doc/plan-t3b2-2026-10-28.md 判据 T0：npz 文件的 sha256 不同时，分清是数组不同还是只有 zip 封装不同。
每集：两边都有的 episode_XXXXXX.npz，键、dtype、形状、数值逐个比（np.array_equal，不容差）；另报两边 zip 条目的 date_time 与压缩方式。
内容不同且帧数相同时，再报 B 是不是 A 左乘一个刚体变换：位置按 Kabsch 拟合 R、t，报位置残差最大值、R 的转角，以及朝向 R·R_A 对 R_B 的最大夹角。
用法：python stages/evaluate/frefs_equal.py <目录 A> <目录 B> [episode,...]   最后一行 FREFS_EQUAL n（内容全同）或 FREFS_DIFFER k（k 集内容不同，退出码 1）。
"""
import sys
import zipfile
from pathlib import Path

import numpy as np


def qmat(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def main() -> int:
    A, B = Path(sys.argv[1]), Path(sys.argv[2])
    only = {int(x) for x in sys.argv[3].split(",")} if len(sys.argv) > 3 else None
    names = sorted(p.name for p in A.glob("episode_*.npz") if (B / p.name).exists() and (only is None or int(p.stem[-6:]) in only))
    bad = 0
    for n in names:
        with np.load(A / n, allow_pickle=False) as a, np.load(B / n, allow_pickle=False) as b:
            same = sorted(a.files) == sorted(b.files) and all(a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and np.array_equal(a[k], b[k]) for k in a.files)
            how = "" if same else "；".join(
                f"{k}: {a[k].dtype}{a[k].shape} 对 {b[k].dtype}{b[k].shape}" + (f"，最大差 {float(np.abs(a[k].astype(float) - b[k].astype(float)).max()):.3g}" if a[k].shape == b[k].shape and a[k].dtype.kind in 'fiu' else "")
                for k in sorted(set(a.files) & set(b.files)) if not (a[k].dtype == b[k].dtype and a[k].shape == b[k].shape and np.array_equal(a[k], b[k])))
        if not same:
            with np.load(A / n, allow_pickle=False) as a, np.load(B / n, allow_pickle=False) as b:
                pa_, pb = a["pose_xyzw"][:, 0], b["pose_xyzw"][:, 0]
            if pa_.shape == pb.shape:
                ca, cb = pa_[:, :3].mean(0), pb[:, :3].mean(0)
                U, _, Vt = np.linalg.svd((pa_[:, :3] - ca).T @ (pb[:, :3] - cb))
                D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
                R = Vt.T @ D @ U.T
                t = cb - R @ ca
                res = float(np.linalg.norm(pa_[:, :3] @ R.T + t - pb[:, :3], axis=1).max())
                ang = lambda M: float(np.degrees(np.arccos(np.clip((np.trace(M) - 1) / 2, -1, 1))))
                worst = max(ang((R @ qmat(x)).T @ qmat(y)) for x, y in zip(pa_[:, 3:], pb[:, 3:]))
                how += f"；刚体拟合：位置残差最大 {res * 100:.2f} cm，R 转角 {ang(R):.2f}°，t {np.linalg.norm(t) * 100:.1f} cm，朝向最大夹角 {worst:.2f}°"
        za = [(i.filename, i.date_time, i.compress_type) for i in zipfile.ZipFile(A / n).infolist()]
        zb = [(i.filename, i.date_time, i.compress_type) for i in zipfile.ZipFile(B / n).infolist()]
        bad += not same
        print(f"{n}\t内容{'相同' if same else '不同'}\tzip 条目{'相同' if za == zb else '不同'}\tA {za[0][1]} B {zb[0][1]}\t{how}")
    print(f"FREFS_EQUAL {len(names)}" if bad == 0 else f"FREFS_DIFFER {bad}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
