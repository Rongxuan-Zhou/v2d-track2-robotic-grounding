"""各机位组的地面法向残差，拆成俯仰与横滚（Mac 上跑，只用标准库；只读 Tier 3 的地面拟合结果，不读真值）。
输入是另一位用户 runs/floor/ 下的 floor_fit_<版本>.json（经 ssh cat 从 ml40 读，脱敏后进记录）：每组有 n_ours（按接触或地平线从重建拟的地面法向，
相机系，OpenCV：x 右、y 下、z 前）与 n_released（发布的 ground_plane.json 法向）。tilt_deg 是两者夹角；这里把它拆开：
  俯仰 = 法向在 y-z 面里的角度差（绕相机 x 轴；主点 cy 偏了主要改这一项，Δcy 像素约改 atan(Δcy/f)），
  横滚 = 法向在 x-y 面里的角度差（绕光轴 z）。
用途：核内参（f、cx、cy）换成主机位标定值之后，二机位、侧机位的地面俯仰是否也对上（agent-doc/plan-t3-kfix-2026-10-16.md 的协作项）。
用法：python3 stages/recovery/t3_floor_group_residual.py <标签>=<floor_fit JSON 文件或 - 表示 stdin> ...；最后一行 FLOOR_GROUP_RESIDUAL_DONE。
"""
import json
import math
import sys
from pathlib import Path


def pitch_roll(n: list) -> tuple:
    # 法向朝上（y 为负），取向上的那一面算角度
    x, y, z = (-v for v in n) if n[1] > 0 else n
    return math.degrees(math.atan2(-z, -y)), math.degrees(math.atan2(x, -y))


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    for arg in sys.argv[1:]:
        tag, path = arg.split("=", 1)
        d = json.loads(sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8"))
        for g in d["groups"]:
            po, ro = pitch_roll(g["n_ours"])
            pr, rr = pitch_roll(g["n_released"])
            print(f"{tag} 组 {g['group']}（{len(g.get('members', []))} 集，{g.get('how')}）：tilt {g.get('tilt_deg')}°，"
                  f"俯仰差 {po - pr:+.2f}°（ours {po:.2f}°、发布 {pr:.2f}°），横滚差 {ro - rr:+.2f}°（ours {ro:.2f}°、发布 {rr:.2f}°），"
                  f"尺度 {g.get('scale')}")
    print("FLOOR_GROUP_RESIDUAL_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
