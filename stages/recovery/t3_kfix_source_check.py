"""k-fix 判据 0 的来源核对：convert 目录里的 CARI4D 输出是不是第一批策略训练用的那一份（Mac 上跑，要 numpy、pyarrow；只读 CARI4D 输出、
世界步参数与另一位用户 tier3_cari4d_f830_floor 的 parquet，不读真值，不读 Tier 1、Tier 2 轨迹）。计划 agent-doc/plan-t3-kfix-2026-10-16.md 判据 0（复核 N6）。

第一批 7 集的策略训在 root_v2 上，root_v2 是 `tier3_cari4d_f830_floor` 的拼接副本。这个版本是另一位用户 10/10 用焦距步（fx 830、深度乘 830/f_c）
与地面世界步（floor_fit_f830.json 第 0 组的 R、s）从 `runs/convert/<集>/object_cam.npz` 生成的（t3_world_all.sh）。把同样两步重新作用到现在的
object_cam.npz：X = s·R·(t ⊙ (1, 1, 830/f_c))，可见帧应与那个 parquet 的物体位置相同（parquet 存 float32，容差取 1e-6 m）。不同就说明 convert
目录在 10/10 之后被重跑过，k-fix 的新参考与旧参考不出自同一次 CARI4D，按判据 0 停下。
用法：uv run --no-project --python 3.12 --with numpy --with pyarrow python stages/recovery/t3_kfix_source_check.py [--root runs/t3world/budget_v2]
（每集目录里要有 object_cam.npz、focal_calib.json、world_calib.json 与 f830_floor 的 episode_*.parquet，见 results/t3world/pull_budget_v2.txt）
最后一行 KFIX_SOURCE_OK n 或 KFIX_SOURCE_FAIL。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
TOL_M = 1e-6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "runs/t3world/budget_v2")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "libs/v2d_infra"))
    from tier_guard import forbid_tier1
    forbid_tier1()
    ok, n = True, 0
    for d in sorted(p for p in a.root.iterdir() if p.is_dir()):
        oc = np.load(d / "object_cam.npz")
        t = oc["T_cam_glb"][:, :3, 3].astype(np.float64)
        fc = float(np.asarray(oc["intrinsics"])[0, 0])
        focal = json.loads((d / "focal_calib.json").read_text())
        wc = json.loads((d / "world_calib.json").read_text())
        k = float(focal["fx_new"]) / fc
        R, s = np.asarray(wc["rotation"], np.float64), float(wc["scale"])
        X = s * (t * np.array([1.0, 1.0, k])) @ R.T
        tab = pq.read_table(next(d.glob("episode_*.parquet")), columns=["observation.object.pose", "observation.object.visible"])
        pose = np.stack(tab.column("observation.object.pose").to_numpy(zero_copy_only=False)).astype(np.float64)
        vis = np.asarray(tab.column("observation.object.visible").to_numpy(zero_copy_only=False), bool)
        if len(pose) != len(X):
            print(f"  {d.name}：帧数不同（parquet {len(pose)}，convert {len(X)}）")
            ok = False
            continue
        diff = float(np.abs(pose[vis, :3] - X[vis]).max())
        good = diff <= TOL_M
        ok &= good
        n += 1
        print(f"  {'ok  ' if good else 'FAIL'} {d.name}：可见帧 {int(vis.sum())}，焦距比 {k:.6f}，世界步缩放 {s}，位置最大差 {diff:.2e} m")
    print(f"KFIX_SOURCE_OK {n}" if ok else "KFIX_SOURCE_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
