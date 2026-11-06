"""一个档位全部参考在第一次接触之前“边走边转”有多猛（训练机上经 scripts/dev/in-flash-chord 在容器里跑，只用 numpy 与 pyarrow；只读，只报不判）。
起因：agent-doc/plan-t12ref-2026-10-20.md 执行记录：Tier 2 ep4 的零残差回放从第 20 帧起落在参考后面，第 55–70 帧参考以约 0.8 m/s 走、同时偏航 1.7–2.5 rad/s。
单看速度（stages/retarget/ref_root_speed.py）分不开摔与不摔，这里看两者同时出现：向心加速度 v·|ω|。按回放速率（--rate，默认 50 帧/秒，逐帧回放）算，
--win 帧滑动平均。每集一行：ep（--summary 给了时从 batch_summary 的 tier/ep/sequence_id 查）、序列、第一次接触帧、接触前 v·|ω| 最大值（m/s²）与出现的帧、
那一帧的 v 与 |ω|、接触前水平加速度 |dv/dt| 最大值（m/s²）、接触前 v·|ω| > --thr 的帧数。用法（容器内）：
python /repo/stages/retarget/ref_turn_load.py <档位 motion 根> [--summary /repo/results/batch_summary_v040.tsv --tier 2] [--rate 50] [--win 10] [--thr 1.0]   最后一行 REF_TURN_LOAD_DONE
"""
import argparse
import csv
import glob
import sys

import numpy as np
import pyarrow.parquet as pq


def yaw(wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = (wxyz[:, i] for i in range(4))
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def smooth(x: np.ndarray, w: int) -> np.ndarray:
    return np.convolve(x, np.ones(w) / w, mode="same") if w > 1 and len(x) > w else x


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--summary")
    ap.add_argument("--tier", default="2")
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--win", type=int, default=10)
    ap.add_argument("--thr", type=float, default=1.0)
    a = ap.parse_args()
    ep = {}
    if a.summary:
        with open(a.summary) as f:
            ep = {r["sequence_id"]: int(r["ep"]) for r in csv.DictReader(f, delimiter="\t") if r["tier"] == a.tier}
    files = sorted(glob.glob(f"{a.root}/whole_body/soma/sequence_id=*/robot_name=g1/data.parquet"))
    print(f"回放速率 {a.rate:g} 帧/秒，窗口 {a.win} 帧，门槛 {a.thr} m/s²")
    print("ep\t序列\t第一次接触帧\t接触前 v·|ω| 最大 m/s²\t出现在第几帧\t那一帧 v m/s\t那一帧 |ω| rad/s\t接触前 |dv/dt| 最大 m/s²\t接触前 v·|ω| > 门槛的帧数")
    rows = []
    for f in files:
        row = pq.read_table(f, columns=["sequence_id", "robot_root_position", "robot_root_wxyz", "hand_contact_active"]).to_pylist()[0]
        p = np.asarray(row["robot_root_position"], dtype=float)
        q = np.asarray(row["robot_root_wxyz"], dtype=float)
        T = p.shape[0]
        c = np.asarray(row["hand_contact_active"], dtype=bool)
        tc = min((int(np.flatnonzero(x)[0]) for x in c if x.any()), default=T - 1)
        v = smooth(np.r_[0.0, np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1)] * a.rate, a.win)
        w = smooth(np.abs(np.r_[0.0, np.angle(np.exp(1j * np.diff(np.unwrap(yaw(q)))))]) * a.rate, a.win)
        acc = np.abs(np.gradient(v) * a.rate)
        load = (v * w)[:max(tc, 1)]
        k = int(load.argmax())
        rows.append((ep.get(row["sequence_id"], -1), row["sequence_id"], tc, float(load.max()), k, float(v[k]), float(w[k]),
                     float(acc[:max(tc, 1)].max()), int((load > a.thr).sum())))
    for r in sorted(rows):
        print(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]:.2f}\t{r[4]}\t{r[5]:.2f}\t{r[6]:.2f}\t{r[7]:.2f}\t{r[8]}")
    print("REF_TURN_LOAD_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
