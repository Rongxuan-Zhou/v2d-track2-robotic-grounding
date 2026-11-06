"""从 4096 世界评测 JSON 的汇总量看误差的分布形状：均值、标准差与各失败原因的世界占比（只用标准库）。

评测 JSON（fc_eval.sh 写的 eval_<步数>_w<世界数>.json）没有逐世界数组，只有均值与标准差；std / mean 大，说明少数世界误差极大、
拉高了均值（例如物体掉了或机器人摔了）。failure_cause_fraction 是按 checkpoint 自带的终止阈值事后判的首个失败原因占比：
骨盆与手掌的阈值变体与基线相同，可以比；物体那一项的朝向阈值不同（0.6 对 1.5 rad），不可比。

用法：python3 stages/evaluate/eval_spread.py <trial 清单 json> [--steps 349999104] [--worlds 4096]（在训练机的仓库根目录跑）
"""

import argparse
import json
from pathlib import Path


def row(run, steps, worlds):
    p = Path("runs/flash_chord") / run / f"eval_{steps}_w{worlds}.json"
    d = json.loads(p.read_text())
    m = d["metrics"]
    fc = d.get("reference_tracking", {}).get("failure_cause_fraction", {})
    return {
        "run": run,
        "mp": m["maniptrans_sr"],
        "sp": m["spider_sr_uncentered"],
        "mppe": m["mppe_cm"],
        "pos": m["object_position_error_m"] * 100,
        "pos_sd": m["object_position_error_std_m"] * 100,
        "ori": m["object_orientation_error_deg"],
        "ori_sd": m["object_orientation_error_std_deg"],
        "add": m["mean_add_m"] * 100,
        "add_sd": m["add_std_m"] * 100,
        "f_pelvis": fc.get("pelvis", float("nan")),
        "f_palm": fc.get("palm", float("nan")),
        "f_object": fc.get("object", float("nan")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--steps", type=int, default=349999104)
    ap.add_argument("--worlds", type=int, default=4096)
    a = ap.parse_args()
    spec = json.loads(Path(a.spec).read_text())
    print("run | MP-SR | SP-SR | MPPE cm | 位置误差 均值±sd cm | 朝向误差 均值±sd ° | ADD 均值±sd cm | 失败:骨盆 | 失败:手掌 | 失败:物体(不可比)")
    for e in spec["episodes"]:
        for k in ("base42", "base43", "var42", "var43"):
            r = row(e[k], a.steps, a.worlds)
            print(f"{r['run']} | {r['mp']:.3f} | {r['sp']:.3f} | {r['mppe']:.2f} | {r['pos']:.2f}±{r['pos_sd']:.2f} | "
                  f"{r['ori']:.1f}±{r['ori_sd']:.1f} | {r['add']:.2f}±{r['add_sd']:.2f} | "
                  f"{r['f_pelvis']:.3f} | {r['f_palm']:.3f} | {r['f_object']:.3f}")
        print()


if __name__ == "__main__":
    main()
