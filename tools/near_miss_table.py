#!/usr/bin/env python3
"""“SP 过、MP 不过”的集：4096 世界评测记录里的成功率与物体平均位置、朝向误差，并列阈值（Mac 上、仓库根执行，只用标准库，只读评测记录）。
SP-SR：时间平均位置 ≤ 0.10 m 且朝向 ≤ 0.5 rad（28.6°）；MP-SR：位置 < 0.03 m 且朝向 < 30°（工具包 v2dlb/object_pose.py）。SP 的朝向门槛比 MP 还严，
所以 SP 过、MP 不过的世界，多半卡在位置（3–10 cm）。这里报的是 4096 个世界的平均误差，不是逐世界分布。
评测记录在 results/day2/eval_<run>_s<步数>.txt（mf4090）或 results/ml40/day2/eval_<run>_s<步数>.txt（ml40），读 summary_json。
用法：python3 tools/near_miss_table.py <run>[@步数] …   最后一行 NEAR_MISS_TABLE_DONE
"""

import json
import sys
from pathlib import Path


def summary(run: str, step: str):
    for d in ("results/day2", "results/ml40/day2"):
        p = Path(d) / f"eval_{run}_s{step}.txt"
        if p.is_file():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("summary_json "):
                    return json.loads(line[len("summary_json "):])
    return None


def main():
    print("run\t步数\tSP-SR\tMP-SR\t平均位置误差 m（MP 门槛 0.03，SP 0.10）\t平均朝向误差 °（MP 门槛 30，SP 28.6）\tAUC\tMPPE cm\t失败原因比例")
    for a in sys.argv[1:]:
        run, _, step = a.partition("@")
        step = step or "349999104"
        j = summary(run, step)
        if j is None:
            print(f"{run}\t{step}\t（没有评测记录）")
            continue
        fc = (j.get("reference_tracking") or {}).get("failure_cause_fraction")
        print(f"{run}\t{step}\t{j['spider_sr_uncentered']:.3f}\t{j['maniptrans_sr']:.3f}\t{j['object_position_error_m']:.4f}\t"
              f"{j['object_orientation_error_deg']:.2f}\t{j['add_auc']:.3f}\t{j['mppe_cm']:.2f}\t{fc}")
    print("NEAR_MISS_TABLE_DONE")


if __name__ == "__main__":
    main()
