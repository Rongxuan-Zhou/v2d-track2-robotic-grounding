"""Tier 1：物体一直不动（停在第 0 帧的位姿）时，各集按官方公式会得多少 MPPE / SP-SR / MP-SR，和现在交的导出那一局并排。

只读 Tier 1 自己的数据（参考就是 Tier 1 的干净轨迹，用 Tier 1 的数据算 Tier 1 的分数，不涉及跨档）；公式直接调提交工具包
v2dlb/object_pose.py 里的函数（与 Kaggle 计分端同一段代码）。刚体变换不改距离和夹角，所以在数据集的 OpenCV 世界系里算，与计分端的世界系
算出来相同。不可见帧（位姿全零）沿用上一帧。

用途：MPPE 是逐帧距离的平均，策略把物体甩出去的集一集就有几十到上百厘米，压过其余 25 集的总和。“不动”的数是这类集的一个参照：
现在的导出比“不动”还差，说明策略在这一集上是在帮倒忙。这里只报数，不产生任何提交物（提交的轨迹必须来自学到的策略）。

用法：python stages/evaluate/t1_static_floor.py <Tier 1 数据根> <工具包根> <select_t1 的 scores.json> <清单.tsv>
"""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def load_kit(kit: Path):
    spec = importlib.util.spec_from_file_location("kit_object_pose", kit / "v2dlb" / "object_pose.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def truth_xyzw(parquet: Path) -> np.ndarray:
    """(T, 7) 位姿，四元数换成 xyzw（数据集是 [x, y, z, qw, qx, qy, qz]）；不可见帧沿用上一帧。"""
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
    root, kit, scores_p, sel = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4]
    op = load_kit(kit)
    scores = json.load(open(scores_p))
    current = {}
    for line in open(sel):
        f = line.rstrip("\n").split("\t")
        if len(f) >= 3 and f[0] == "1":
            current[int(f[1])] = f[2]
    files = sorted((root / "data").glob("chunk-*/episode_*.parquet"))
    if len(files) != 30 or len(current) != 30:
        raise SystemExit(f"数据 {len(files)} 集，清单 {len(current)} 集")
    print("集\t现选 run\t导出 MPPE\t导出 SP/MP\t不动 MPPE\t不动 SP/MP\t不动减导出（cm）")
    sum_cur = sum_floor = 0.0
    worse = []
    for f in files:
        ep = int(f.stem.split("_")[-1])
        ref = truth_xyzw(f)[:, None, None, :]                    # (T, 1, 1, 7)
        static = np.repeat(ref[:1], len(ref), axis=0)           # 物体停在第 0 帧
        mppe = op._mean_per_point_error_cm(static, ref)
        sp = float(np.mean(op._spider_success(static, ref)))
        mp = float(np.mean(op._maniptrans_success(static, ref, np.array([0]))))
        c = scores[current[ep]]
        sum_cur += c["mppe_cm"]
        sum_floor += min(c["mppe_cm"], mppe)
        if mppe < c["mppe_cm"]:
            worse.append(ep)
        print(f"ep{ep}\t{current[ep]}\t{c['mppe_cm']:.2f}\t{c['spider_sr']:.0f}/{c['maniptrans_sr']:.0f}\t"
              f"{mppe:.2f}\t{sp:.0f}/{mp:.0f}\t{mppe - c['mppe_cm']:+.2f}")
    print(f"导出比“不动”还差的集：{worse}")
    print(f"现在的 MPPE 合计 {sum_cur:.2f} cm（平均 {sum_cur / 30:.4f}）；这些集换成“不动”的数后合计 {sum_floor:.2f} cm（平均 {sum_floor / 30:.4f}）")
    print("T1_STATIC_FLOOR_DONE")


if __name__ == "__main__":
    main()
