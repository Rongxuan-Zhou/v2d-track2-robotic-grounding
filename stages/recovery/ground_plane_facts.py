"""发布的地面平面在什么坐标系里：逐集列出法向与 OpenCV 相机“上方”（−y）的夹角、原点到平面的距离（只用标准库，只读）。

每个 ground_plane/<序列>.json 的 "plane" 是 [a, b, c, d]，平面 ax + by + cz + d = 0（数据 README：与 observation.object.pose 同在
“source OpenCV world frame”）。如果这个世界系就是 exo 相机的 OpenCV 系（y 朝下、z 朝前、原点在相机），法向应接近 −y，
|d| 就是相机离地的高度。脚本只读地面平面文件，不读任何轨迹。
用法：python3 stages/recovery/ground_plane_facts.py <ground_plane 目录>
  落盘：scripts/dev/record <名字> scripts/dev/pipe-py stages/recovery/ground_plane_facts.py data/challenge/track_2/tier_2_synthetic_noise/ground_plane
"""
import json
import math
import sys
from pathlib import Path


def main() -> int:
    d = Path(sys.argv[1])
    files = sorted(d.glob("*.json"))
    if not files:
        raise SystemExit(f"ground_plane_facts: {d} 里没有 json")
    print("序列\t法向与 -y 夹角(°)\t法向 x,y,z\t原点到平面 |d|(m)\t内点比例")
    angs, ds = [], []
    for f in files:
        j = json.loads(f.read_text())
        a, b, c, dd = j["plane"]
        n = math.sqrt(a * a + b * b + c * c)
        a, b, c, dd = a / n, b / n, c / n, dd / n
        if b > 0:   # 统一成法向朝 −y（相机的“上方”）
            a, b, c, dd = -a, -b, -c, -dd
        ang = math.degrees(math.acos(max(-1.0, min(1.0, -b))))
        frac = j.get("n_inliers", 0) / j["n_candidates"] if j.get("n_candidates") else float("nan")
        angs.append(ang)
        ds.append(abs(dd))
        print(f"{f.stem}\t{ang:.2f}\t{a:+.3f},{b:+.3f},{c:+.3f}\t{abs(dd):.3f}\t{frac:.3f}")
    angs.sort()
    ds.sort()
    print(f"共 {len(files)} 集：法向与 -y 夹角 {angs[0]:.2f}–{angs[-1]:.2f}°（中位 {angs[len(angs) // 2]:.2f}°），"
          f"|d| {ds[0]:.3f}–{ds[-1]:.3f} m（中位 {ds[len(ds) // 2]:.3f} m）")
    print(f"文件里的键：{sorted(json.loads(files[0].read_text()))}")
    print("GROUND_PLANE_FACTS_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
