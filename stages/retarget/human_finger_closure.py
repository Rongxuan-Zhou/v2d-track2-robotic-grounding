"""Tier 2 里人手的手指在接触段弯没弯（训练机上经 scripts/dev/in-flash-chord 在容器里跑，只用 numpy 与 pyarrow，只读；只报不判）。
起因：agent-doc/plan-t2dg-2026-10-20.md 事实 2：30 集参考（重定向出来的 motion parquet）里 Dex3 的手指关节全程是 0。
若人手在接触段明显弯曲，“把人手的手指角重定向到 Dex3”才有依据。这里只读 Tier 2 自己的人体姿态（observation.human.pose，各关节局部旋转向量）
与同一集的 Tier 2 参考（只取 hand_contact_active），进程开头调用 tier_guard.forbid_tier1()。
每只手的弯曲量 = 该手 Thumb/Index/Middle/Ring/Pinky 各节（不含 *End）局部旋转角之和（弧度）；第 0 帧不一定是张开的手，所以报绝对值，
另报相对本集该手最小值的增量。每集一行：接触帧与非接触帧的弯曲量中位数与差。--detail 时另按 --every 帧列出这一集的逐帧值。
用法（容器内）：python /repo/stages/retarget/human_finger_closure.py [--detail] [--every 20] <ep>:<motion parquet>...   最后一行 HUMAN_FINGER_DONE
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "libs"))
from v2d_infra import tier_guard
from v2d_infra.paths import TIER2

DIGITS = ("Thumb", "Index", "Middle", "Ring", "Pinky")


def finger_cols(names: list[str], side: str) -> list[list[int]]:
    joints = []
    for i in range(0, len(names), 3):
        j = names[i].rsplit(".", 1)[0]
        if j.startswith(f"{side}Hand") and any(d in j for d in DIGITS) and not j.endswith("End"):
            joints.append([i, i + 1, i + 2])
    return joints


def main() -> int:
    tier_guard.forbid_tier1()
    ap = argparse.ArgumentParser()
    ap.add_argument("pairs", nargs="+")
    ap.add_argument("--detail", action="store_true")
    ap.add_argument("--every", type=int, default=20)
    a = ap.parse_args()
    names = json.loads((TIER2 / "meta/info.json").read_text())["features"]["observation.human.pose"]["names"]
    cols = {s: finger_cols(names, s) for s in ("Left", "Right")}
    print(f"人手手指节数：左 {len(cols['Left'])}、右 {len(cols['Right'])}（Thumb/Index/Middle/Ring/Pinky，不含 End）")
    print("ep\t帧\t手\t接触帧\t接触时弯曲中位 rad\t不接触时弯曲中位 rad\t差 rad\t全集 10%/90% rad")
    for pair in a.pairs:
        ep_s, motion = pair.split(":", 1)
        ep = int(ep_s)
        t = pq.read_table(TIER2 / f"data/chunk-000/episode_{ep:06d}.parquet", columns=["frame_index", "observation.human.pose"])
        order = np.argsort(np.asarray(t.column("frame_index").to_pylist()))
        pose = np.asarray(t.column("observation.human.pose").to_pylist(), dtype=float)[order]
        row = pq.read_table(motion, columns=["hand_sides", "hand_contact_active"]).to_pylist()[0]
        contact = {s: np.asarray(c, dtype=bool) for s, c in zip(row["hand_sides"], row["hand_contact_active"])}
        T = pose.shape[0]
        flex = {}
        for side, key in (("Left", "left"), ("Right", "right")):
            ang = np.stack([np.linalg.norm(pose[:, c], axis=1) for c in cols[side]], axis=1)
            flex[key] = ang.sum(axis=1)
            c = contact.get(key, np.zeros(T, bool))
            if c.shape[0] != T:
                print(f"{ep}\t{T}\t{key}\t参考帧数 {c.shape[0]} 与人体帧数不同，跳过")
                continue
            fin = flex[key][c]
            fout = flex[key][~c]
            mi = np.median(fin) if fin.size else float("nan")
            mo = np.median(fout) if fout.size else float("nan")
            print(f"{ep}\t{T}\t{key}\t{int(c.sum())}\t{mi:.3f}\t{mo:.3f}\t{mi - mo:+.3f}\t"
                  f"{np.percentile(flex[key], 10):.3f}/{np.percentile(flex[key], 90):.3f}")
        if a.detail:
            print(f"  ep{ep} 逐帧：帧\t左弯曲 rad\t左接触\t右弯曲 rad\t右接触")
            for f in list(range(0, T, a.every)) + [T - 1]:
                print(f"  {f}\t{flex['left'][f]:.3f}\t{int(contact.get('left', np.zeros(T, bool))[f])}\t"
                      f"{flex['right'][f]:.3f}\t{int(contact.get('right', np.zeros(T, bool))[f])}")
    print("HUMAN_FINGER_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
