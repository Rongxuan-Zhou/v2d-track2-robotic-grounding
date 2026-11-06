"""Tier 2：物体一直不动（停在第 0 帧）时，相对 Tier 2 自己的物体轨迹，按官方公式各集的 MPPE / SP-SR / MP-SR 是多少（净室：挂 tier_guard）。

对照的是 Tier 2 数据集里带噪的物体轨迹（不是干净真值，也不是 v5.4 降噪后的参考），所以只能看量级：带噪轨迹与 v5.4 参考差 1–3 cm、
十几度，MPPE 的关键点半径 5 cm，旋转噪声折进来约 1–2 cm；“不动”的数本身是几十厘米的量级，这点差不改结论。
与 stages/evaluate/t1_static_floor.py 同一套公式（工具包 v2dlb/object_pose.py）。可选给一张“集 → 导出 MPPE”的表（相对 v5.4 参考，
例如 tools/t2_lost_swap.py 的结果）并排打印。

用法：python stages/evaluate/t2_static_floor.py <Tier 2 数据根> <工具包根> [ep=MPPE …]
"""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "libs" / "v2d_infra"))
from tier_guard import forbid_tier1  # noqa: E402

forbid_tier1()

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402


def load_kit(kit: Path):
    spec = importlib.util.spec_from_file_location("kit_object_pose", kit / "v2dlb" / "object_pose.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def traj_xyzw(parquet: Path) -> np.ndarray:
    t = pq.read_table(parquet, columns=["observation.object.pose", "observation.object.visible"])
    pose = np.asarray(t.column("observation.object.pose").to_pylist(), dtype=np.float64)
    vis = np.asarray(t.column("observation.object.visible").to_pylist(), dtype=bool)
    if not vis[0]:
        raise SystemExit(f"{parquet.name}：第 0 帧不可见")
    for i in range(1, len(pose)):
        if not vis[i]:
            pose[i] = pose[i - 1]
    return np.concatenate([pose[:, :3], pose[:, 4:7], pose[:, 3:4]], axis=1)


def main():
    root, kit = Path(sys.argv[1]), Path(sys.argv[2])
    if "tier_2" not in str(root):
        raise SystemExit(f"{root} 不是 Tier 2 的数据根")
    given = {int(k.removeprefix("ep")): float(v) for k, v in (a.split("=") for a in sys.argv[3:])}
    op = load_kit(kit)
    files = sorted((root / "data").glob("chunk-*/episode_*.parquet"))
    if len(files) != 30:
        raise SystemExit(f"数据 {len(files)} 集")
    print("集\t不动 MPPE（cm）\t不动 SP/MP\t给定的导出 MPPE\t不动减导出")
    for f in files:
        ep = int(f.stem.split("_")[-1])
        ref = traj_xyzw(f)[:, None, None, :]
        static = np.repeat(ref[:1], len(ref), axis=0)
        mppe = op._mean_per_point_error_cm(static, ref)
        sp = float(np.mean(op._spider_success(static, ref)))
        mp = float(np.mean(op._maniptrans_success(static, ref, np.array([0]))))
        g = given.get(ep)
        print(f"ep{ep}\t{mppe:.2f}\t{sp:.0f}/{mp:.0f}\t" + (f"{g:.2f}\t{mppe - g:+.2f}" if g is not None else "—\t—"))
    print("T2_STATIC_FLOOR_DONE")


if __name__ == "__main__":
    main()
