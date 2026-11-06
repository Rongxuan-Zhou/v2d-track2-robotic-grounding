"""Tier 3 世界坐标系的事实之一：30 集发布的地面平面放在一起比（Mac 上跑，只用标准库；不读任何轨迹，不读 Tier 1）。

数据集 README（tier_2_synthetic_noise/README.md）：物体位姿与地面平面都在“source OpenCV world frame”里；导出没有相机内参与外参；
每集保留了源标定序列的编号（episodes_metadata.jsonl 的 calibration_sequence），但那不是标定矩阵。地面平面是对点云做 RANSAC 拟合的
[a, b, c, d]（ax + by + cz + d = 0，法向单位化）。地面平面与网格属场景信息，不是轨迹，Tier 3 链路一直在用（发布的 ground_plane）。

要回答的是：世界坐标系是不是 exo 主机位相机的坐标系。平面在相机坐标系里只反映相机的俯仰、横滚与离地高度，与偏航无关，所以只比平面
分不开“各相机自己的坐标系”与“固定的房间坐标系”；这一步只看三件事，判读规则在看数据之前写在这里：
  1. 主机位（另一位用户 10/10 交接按画面背景分的组：除 ep12、14、16、25、28 与 ep21 之外的 24 集）跨标定日是不是同一个平面：
     两两法向夹角最大值 ≤ 1.0° 且 d 的极差 ≤ 3 cm，记“主机位各集平面一致”；否则记“主机位各集平面不一致”，按标定序列分组再看。
  2. 第二机位（ep12、14、16、25、28）、侧机位（ep21）与主机位的平面比：各自与主机位平均法向的夹角、d 之差。夹角 ≤ 1.0° 且 |Δd| ≤ 3 cm
     记“与主机位同一平面”。同一平面时有两种可能（房间坐标系；或各相机碰巧同高、同俯仰横滚），要靠第 2 步的图像检验分开；
     不同平面时，世界坐标系至少对其中一组不是同一个，或者是各相机自己的坐标系。
  3. 法向离 −y 轴多远（OpenCV 约定 y 朝下，地面法向朝上）：只报，配合第 2 步。
输出逐集一行：episode、序列、标定序列、机位组、法向、d、与 −y 的夹角、与主机位平均法向的夹角、d − 主机位 d 中位；再按机位组与标定序列汇总。
用法：python3 stages/recovery/t3_world_frame_facts.py [--root data/challenge/track_2/tier_2_synthetic_noise]
最后一行 WORLD_FRAME_FACTS_DONE。
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SECOND = {12, 14, 16, 25, 28}   # 另一位用户 10/10 交接（t3_camera_groups.py）：第二机位
SIDE = {21}                     # 侧机位
ANG_TOL_DEG, D_TOL_M = 1.0, 0.03


def group(ep: int) -> str:
    return "second" if ep in SECOND else "side" if ep in SIDE else "main"


def unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def angle_deg(u, v) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, sum(a * b for a, b in zip(u, v))))))


def load(root: Path) -> list:
    rows = []
    for line in (root / "meta/episodes_metadata.jsonl").read_text().splitlines():
        m = json.loads(line)
        p = json.loads((root / m["ground_plane"]).read_text())["plane"]
        n = math.sqrt(p[0] ** 2 + p[1] ** 2 + p[2] ** 2)
        a, b, c, d = (x / n for x in p)
        if d < 0:   # 统一成 d > 0：原点在法向所指的一侧，法向指向原点那一侧
            a, b, c, d = -a, -b, -c, -d
        rows.append({"ep": int(m["episode_index"]), "seq": m["sequence_id"], "cal": m["calibration_sequence"],
                     "grp": group(int(m["episode_index"])), "n": [a, b, c], "d": d, "raw_norm": n})
    return sorted(rows, key=lambda r: r["ep"])


def mean_normal(rows) -> list:
    return unit([sum(r["n"][i] for r in rows) for i in range(3)])


def max_pair_angle(rows) -> float:
    return max((angle_deg(a["n"], b["n"]) for i, a in enumerate(rows) for b in rows[i + 1:]), default=0.0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "data/challenge/track_2/tier_2_synthetic_noise")
    a = ap.parse_args()
    rows = load(a.root)
    main_rows = [r for r in rows if r["grp"] == "main"]
    mn = mean_normal(main_rows)
    dmed = sorted(r["d"] for r in main_rows)[len(main_rows) // 2]
    print(f"== 30 集的地面平面（{a.root}）；单位化前的范数全部在 {min(r['raw_norm'] for r in rows):.6f}–{max(r['raw_norm'] for r in rows):.6f}")
    print("ep\t机位组\t标定序列\tn_x\tn_y\tn_z\td(m)\t与-y夹角°\t与主机位平均法向夹角°\td-主机位中位(cm)")
    for r in rows:
        print(f"{r['ep']}\t{r['grp']}\t{r['cal']}\t{r['n'][0]:+.5f}\t{r['n'][1]:+.5f}\t{r['n'][2]:+.5f}\t{r['d']:.4f}\t"
              f"{angle_deg(r['n'], [0, -1, 0]):.2f}\t{angle_deg(r['n'], mn):.2f}\t{100 * (r['d'] - dmed):+.1f}")
    print("== 按机位组")
    for g in ("main", "second", "side"):
        rs = [r for r in rows if r["grp"] == g]
        gm = mean_normal(rs)
        ds = [r["d"] for r in rs]
        print(f"{g}：{len(rs)} 集，{len({r['cal'] for r in rs})} 个标定序列；组内两两法向最大夹角 {max_pair_angle(rs):.2f}°；"
              f"d {min(ds):.4f}–{max(ds):.4f} m（极差 {100 * (max(ds) - min(ds)):.1f} cm）；组平均法向与主机位平均法向夹角 {angle_deg(gm, mn):.2f}°，"
              f"组 d 中位 − 主机位 d 中位 {100 * (sorted(ds)[len(ds) // 2] - dmed):+.1f} cm")
    print("== 主机位按标定序列")
    for cal in sorted({r["cal"] for r in main_rows}):
        rs = [r for r in main_rows if r["cal"] == cal]
        ds = [r["d"] for r in rs]
        print(f"{cal}：ep {','.join(str(r['ep']) for r in rs)}；组内最大夹角 {max_pair_angle(rs):.2f}°；d {min(ds):.4f}–{max(ds):.4f}；"
              f"与主机位平均法向夹角 {angle_deg(mean_normal(rs), mn):.2f}°")
    ok_main = max_pair_angle(main_rows) <= ANG_TOL_DEG and (max(r['d'] for r in main_rows) - min(r['d'] for r in main_rows)) <= D_TOL_M
    print(f"判读 1：{'主机位各集平面一致' if ok_main else '主机位各集平面不一致'}（规则：两两夹角 ≤ {ANG_TOL_DEG}° 且 d 极差 ≤ {100 * D_TOL_M:.0f} cm）")
    for g in ("second", "side"):
        rs = [r for r in rows if r["grp"] == g]
        same = all(angle_deg(r["n"], mn) <= ANG_TOL_DEG and abs(r["d"] - dmed) <= D_TOL_M for r in rs)
        print(f"判读 2（{g}）：{'与主机位同一平面' if same else '与主机位不是同一平面'}（规则：每集与主机位平均法向夹角 ≤ {ANG_TOL_DEG}° 且 |d − 主机位中位| ≤ {100 * D_TOL_M:.0f} cm）")
    print("WORLD_FRAME_FACTS_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
