"""一个档位全部参考在第一次接触之前各关节转得多快，某一集在哪些关节上比所有“好集”都快（训练机上经 scripts/dev/in-flash-chord 在容器里跑，只用 numpy 与 pyarrow；只读，只报不判）。
起因：agent-doc/plan-t12ctl-2026-10-20.md：速度、偏航角速度、向心加速度都分不开 ep4 与成功集，逐关节找 ep4 特有的量。
按回放速率（--rate，默认 50 帧/秒）算每个关节 |dq/dt|，--win 帧滑动平均，取接触前最大值。--summary 给 ep 编号，--ok 给“好集”的 ep，--target 给要查的 ep。
输出：每个关节一行：target 的值、好集里的最大值与是哪一集、target 在全部集里的名次；按 target / 好集最大值 从大到小排。
用法（容器内）：python /repo/stages/retarget/ref_joint_speed.py <档位 motion 根> --summary <batch_summary.tsv> --tier 2 --ok 0,3,5 --target 4 [--rate 50] [--win 5]   最后一行 REF_JOINT_SPEED_DONE
"""
import argparse
import csv
import glob
import sys

import numpy as np
import pyarrow.parquet as pq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--summary", required=True)
    ap.add_argument("--tier", default="2")
    ap.add_argument("--ok", required=True)
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--win", type=int, default=5)
    a = ap.parse_args()
    with open(a.summary) as f:
        ep = {r["sequence_id"]: int(r["ep"]) for r in csv.DictReader(f, delimiter="\t") if r["tier"] == a.tier}
    ok = {int(x) for x in a.ok.split(",")}
    vals, names = {}, None
    for f in sorted(glob.glob(f"{a.root}/whole_body/soma/sequence_id=*/robot_name=g1/data.parquet")):
        row = pq.read_table(f, columns=["sequence_id", "robot_joint_names", "robot_joint_positions", "hand_contact_active"]).to_pylist()[0]
        n = list(row["robot_joint_names"])
        if names is None:
            names = n
        elif n != names:
            print(f"{row['sequence_id']} 的关节名与第一集不同")
            return 2
        q = np.asarray(row["robot_joint_positions"], dtype=float)
        c = np.asarray(row["hand_contact_active"], dtype=bool)
        tc = min((int(np.flatnonzero(x)[0]) for x in c if x.any()), default=q.shape[0] - 1)
        dq = np.abs(np.diff(q[:max(tc, 2)], axis=0)) * a.rate
        k = np.ones(a.win) / a.win
        sm = np.stack([np.convolve(dq[:, j], k, mode="same") for j in range(dq.shape[1])], 1) if dq.shape[0] > a.win else dq
        vals[ep[row["sequence_id"]]] = sm.max(axis=0)
    if a.target not in vals or not ok <= set(vals):
        print(f"缺集：target {a.target}、ok {sorted(ok - set(vals))}")
        return 2
    V = np.stack([vals[e] for e in sorted(vals)])
    eps = sorted(vals)
    okl = sorted(ok)
    O = np.stack([vals[e] for e in okl])
    t = vals[a.target]
    ratio = t / np.maximum(O.max(axis=0), 1e-9)
    print(f"{len(eps)} 集；好集 {okl}；target ep{a.target}；速率 {a.rate:g} 帧/秒，窗口 {a.win} 帧；单位 rad/s")
    print("关节\ttarget\t好集最大\t是哪一集\ttarget/好集最大\ttarget 在全部集里的名次（1 = 最快）")
    for j in np.argsort(-ratio):
        rank = int((V[:, j] > t[j]).sum()) + 1
        print(f"{names[j]}\t{t[j]:.2f}\t{O[:, j].max():.2f}\tep{okl[int(O[:, j].argmax())]}\t{ratio[j]:.2f}\t{rank}/{len(eps)}")
    print("REF_JOINT_SPEED_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
