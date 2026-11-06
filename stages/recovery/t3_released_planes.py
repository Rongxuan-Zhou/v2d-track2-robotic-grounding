"""发布地面按机位组的分布（Mac 上、仓库根执行，只用标准库；只读发布的 ground_plane.json，不读任何轨迹）。
地面文件取 data/challenge/track_2/tier_2_synthetic_noise/ground_plane/（与 t3_floor_tilt_k.py 相同的来源）；机位组取 t3_manhattan_calib.py 的
SECOND、SIDE（其余是主机位）。每集把平面换成朝上的单位法向（OpenCV 相机系：x 右、y 下、z 前），报俯仰（绕 x）、横滚（绕光轴 z）与离原点的距离 |d|。
用途：判断发布地面是在各集自己相机的坐标系里，还是在同一个世界系（主机位）里。三台朝向不同的相机，地面在各自坐标系里的俯仰、横滚、高度
不会碰巧一样；各组的分布落在同一范围，就说明发布地面在同一个世界系里，二机位、侧机位地面拟合里的“残差”是相机自身的姿态，不能拿来核内参。
用法：python3 stages/recovery/t3_released_planes.py；最后一行 RELEASED_PLANES_DONE。
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "data/challenge/track_2/tier_2_synthetic_noise"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from t3_manhattan_calib import SECOND, SIDE


def main() -> int:
    rows = []
    for line in (DS / "meta/episodes_metadata.jsonl").read_text().splitlines():
        m = json.loads(line)
        e = int(m["episode_index"])
        p = json.loads((DS / m["ground_plane"]).read_text())["plane"]
        k = math.sqrt(sum(v * v for v in p[:3]))
        n, d = [v / k for v in p[:3]], p[3] / k
        if n[1] > 0:
            n, d = [-v for v in n], -d
        pitch, roll = math.degrees(math.atan2(-n[2], -n[1])), math.degrees(math.atan2(n[0], -n[1]))
        rows.append(("二机位" if e in SECOND else "侧机位" if e in SIDE else "主机位", e, pitch, roll, abs(d)))
    for g in ("主机位", "二机位", "侧机位"):
        r = [x for x in rows if x[0] == g]
        lo_hi = [(min(x[i] for x in r), max(x[i] for x in r)) for i in (2, 3, 4)]
        print(f"{g}（{len(r)} 集：{'、'.join(f'ep{x[1]}' for x in r) if len(r) < 10 else '其余各集'}）：俯仰 {lo_hi[0][0]:.2f}–{lo_hi[0][1]:.2f}°，"
              f"横滚 {lo_hi[1][0]:.2f}–{lo_hi[1][1]:.2f}°，|d| {lo_hi[2][0]:.3f}–{lo_hi[2][1]:.3f} m")
    print("RELEASED_PLANES_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
