"""k-fix 判据 2：另一位用户的 t3_floor_fit.py 在新表达上重拟的地面校正，是否复现 t3_floor_tilt_k.py 的同一行（Mac 上跑，只用标准库；只读地面拟合的 JSON）。
计划 agent-doc/plan-t3-kfix-2026-10-16.md 判据 2（复核 N3：这是两份实现的复现核对，不是预测）。

登记的目标（口径 F，方格 GB 内参；`results/t3world/floor_tilt_k.txt` 的 GB1 行）：倾斜 1.22° ± 0.1°，|横滚分量| > |俯仰分量|；
s′ = d_released / d_ours，目标 1.38765 / 1.4102 = 0.98401 ± 0.003（1.38765 是主机位 24 集发布 d 的中位，1.4102 是 GB1 行拟出的离地高度）。
俯仰、横滚分量的算法同 t3_floor_tilt_k.py：把 n_ours 与 n_released 投到相机 y-z 平面（俯仰）与 x-y 平面（横滚）里比角度。
用法：python3 stages/recovery/t3_kfix_floor_check.py <floor_fit_kt.json>；最后一行 KFIX_FLOOR_REPRODUCED 或 KFIX_FLOOR_NOT_REPRODUCED。
"""
import json
import math
import sys
from pathlib import Path

TILT_REF, TILT_TOL = 1.22, 0.10
S_REF, S_TOL = 1.38765 / 1.4102, 0.003


def main() -> int:
    d = json.loads(Path(sys.argv[1]).read_text())
    g = next(x for x in d["groups"] if x.get("group") == 0)
    n, r = g["n_ours"], g["n_released"]
    pitch = math.degrees(math.atan2(n[2], -n[1]) - math.atan2(r[2], -r[1]))
    roll = math.degrees(math.atan2(n[0], -n[1]) - math.atan2(r[0], -r[1]))
    tilt, s = float(g["tilt_deg"]), float(g["scale"])
    ok_t = abs(tilt - TILT_REF) <= TILT_TOL and abs(roll) > abs(pitch)
    ok_s = abs(s - S_REF) <= S_TOL
    print(f"主机位组：{g['events']} 个事件，方式 {g['how']}；倾斜 {tilt:.3f}°（俯仰分量 {pitch:+.3f}°、横滚分量 {roll:+.3f}°），"
          f"离地 d_ours {g['d_ours_m']} m、d_released {g['d_released_m']} m，s′ {s:.5f}")
    print(f"  倾斜：目标 {TILT_REF} ± {TILT_TOL}° 且 |横滚| > |俯仰| → {'复现' if ok_t else '不复现'}")
    print(f"  s′：目标 {S_REF:.5f} ± {S_TOL} → {'复现' if ok_s else '不复现'}")
    if "validation_summary" in g:
        v = g["validation_summary"]
        print(f"  留一验证（她的脚本自带）：{v['episodes']} 集，接触点离发布地面的高度中位绝对值 校正前 {100 * v['before_abs_median_m']:.1f} cm、"
              f"校正后 {100 * v['after_abs_median_m']:.1f} cm（最大 {100 * v['after_abs_max_m']:.1f} cm）")
    print("KFIX_FLOOR_REPRODUCED" if ok_t and ok_s else "KFIX_FLOOR_NOT_REPRODUCED")
    return 0 if ok_t and ok_s else 1


if __name__ == "__main__":
    sys.exit(main())
