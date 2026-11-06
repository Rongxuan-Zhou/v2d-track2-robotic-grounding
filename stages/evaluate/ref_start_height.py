"""参考物体开头几帧的高度与这条 run 实际用的支撑面设置（容器里跑，要 numpy、pyarrow）。只读、不占 GPU。
用途：逐帧终止分析里起点挤在冻结期刚结束（第 51、69、72 帧）的几条 run 都设了 scene.support=false，核对物体开头是不是放在高处（台面上），
关了支撑面就会掉（计划 agent-doc/plan-frame-onset-2026-10-14.md 执行记录）。参考 parquet 与配置都取自 policy 内嵌的配置（同 fail_onset.py）。
一次只收同一档的 run；t2、t3 先装 Tier 1 禁读。
打印：每条 run 的 scene.support、物体网格半径、第 0、25、50、75、100 帧物体根位置的 z（米，地面已找平到 z=0）与第 0 到 100 帧 z 的最小值。
用法：python stages/evaluate/ref_start_height.py <run> [<run> ...] [--steps 349999104] [--root .]
"""

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import fail_onset as fon  # noqa: E402
from libs.v2d_infra import tier_guard  # noqa: E402

FRAMES = (0, 25, 50, 75, 100)


def config(root: Path, run: str, steps: int) -> dict:
    with open(root / "runs/flash_chord" / run / f"policy_{steps}.safetensors", "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(json.loads(f.read(n))["__metadata__"]["user.resolved_config_json"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--steps", type=int, default=349999104)
    ap.add_argument("--root", default=".")
    a = ap.parse_args(argv)
    tiers = {r.removeprefix("smoke-").split("-", 1)[0] for r in a.runs}   # 冒烟 run 也收（计划 agent-doc/plan-t3g5-2026-10-21.md：用冒烟的 policy 提前看参考开头）
    if len(tiers) != 1 or not tiers <= {"t1", "t2", "t3"}:
        print(f"ref_start_height: 一次只收同一档的 run，这次是 {sorted(tiers)}", file=sys.stderr)
        return 2
    if tiers != {"t1"}:
        tier_guard.forbid_tier1()
    import pyarrow.parquet as pq

    root = Path(a.root)
    for run in a.runs:
        cfg = config(root, run, a.steps)
        path = fon.ref_parquet(root, run, a.steps)
        t = pq.read_table(path, columns=["object_root_position", "object_mesh_radius"]).to_pydict()
        pos = np.asarray(t["object_root_position"][0], dtype=np.float64).reshape(-1, 3)
        rad = t["object_mesh_radius"][0]
        sup = (cfg.get("scene") or {}).get("support")
        zs = "，".join(f"第 {f} 帧 {pos[f, 2]:.3f}" for f in FRAMES if f < len(pos))
        print(f"== {run}：scene.support={sup}，物体网格半径 {rad}，{len(pos)} 帧；物体根 z：{zs}；第 0–100 帧最小 {pos[:101, 2].min():.3f}")
        print(f"   参考 {path}")
    print("REF_START_HEIGHT_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
