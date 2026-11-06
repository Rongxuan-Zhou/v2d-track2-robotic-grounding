"""Tier 3 对齐的端到端核对：打包好的 CSV 逐集第 0 帧对 Tier 1 第 0 帧（Mac 上跑，要 numpy、pyarrow）。计划 agent-doc/plan-t3align-2026-10-21.md 复核改动第 1 条。
Tier 1 只经 stages/evaluate/tier1_frame0.py 读第 0 帧。CSV 的行 track_2/eXXXXXX/w000/f000000/o0 是这一集第 0 个计分帧的物体位姿（pos_*、quat_xyzw）。
判据（登记）：ALIGN.json 里对齐了的每集（方案不是 none）——新 CSV 第 0 帧位置差 ≤ 2 cm，且小于旧 CSV 的；选了 obj 的集，新 CSV 第 0 帧朝向差 ≤ 5°。
负对照：用 --swap-quat 生成的帧参考打出来的 CSV，这条核对必须报不过。
用法：python csv_frame0_check.py <旧 CSV> <新 CSV> <新帧参考目录的 ALIGN.json> --tier1 <Tier 1 数据根>   最后一行 CSV_FRAME0_OK 或 CSV_FRAME0_FAIL（退出码 1）
--episodes 5,10,…：只核 ALIGN.json 里这几集（每一集都必须在 ALIGN.json 里）；清单只换了一部分集时用（计划 agent-doc/plan-t3pack-2026-10-29.md 复核第 2 条）。
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

import tier1_frame0


def frame0_rows(path: Path) -> dict:
    out = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            parts = r["row_id"].split("/")
            if parts[3] == "f000000" and parts[4] == "o0":
                ep = int(parts[1][1:])
                out[ep] = (np.array([float(r["pos_x"]), float(r["pos_y"]), float(r["pos_z"])]),
                           np.array([float(r["quat_x"]), float(r["quat_y"]), float(r["quat_z"]), float(r["quat_w"])]))
    return out


def rot_err_deg(qa, qb) -> float:
    qa, qb = qa / np.linalg.norm(qa), qb / np.linalg.norm(qb)
    return math.degrees(2 * math.acos(min(1.0, abs(float(np.dot(qa, qb))))))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("align_json")
    ap.add_argument("--tier1", required=True)
    ap.add_argument("--episodes", default="")
    a = ap.parse_args()
    old, new = frame0_rows(Path(a.old)), frame0_rows(Path(a.new))
    eps = json.loads(Path(a.align_json).read_text())["episodes"]
    if a.episodes:
        want = [str(int(x)) for x in a.episodes.split(",") if x.strip()]
        missing = [k for k in want if k not in eps]
        if missing:
            raise SystemExit(f"--episodes 里的集不在 ALIGN.json 里：{','.join(missing)}")
        eps = {k: eps[k] for k in want}
    bad = 0
    print("ep\t方案\t旧位置差cm\t新位置差cm\t旧朝向差°\t新朝向差°\t结论")
    for k in sorted(eps, key=int):
        ep, choice = int(k), eps[k]["choice"]
        t = tier1_frame0.read(Path(a.tier1), ep)
        if ep not in old or ep not in new:
            print(f"{ep}\t{choice}\t缺第 0 帧行（第 0 帧不计分？）\t\t\t\t只报")
            continue
        dpo, dpn = (float(np.linalg.norm(x[0] - t["p"]) * 100) for x in (old[ep], new[ep]))
        dro, drn = (rot_err_deg(x[1], t["q_xyzw"]) for x in (old[ep], new[ep]))
        ok = True
        if choice != "none":
            ok = dpn <= 2.0 and dpn < dpo and (choice != "obj" or drn <= 5.0)
        bad += not ok
        print(f"{ep}\t{choice}\t{dpo:.2f}\t{dpn:.2f}\t{dro:.1f}\t{drn:.1f}\t{'过' if ok else '不过'}")
    print("CSV_FRAME0_OK" if bad == 0 else f"CSV_FRAME0_FAIL {bad} 集不过")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
