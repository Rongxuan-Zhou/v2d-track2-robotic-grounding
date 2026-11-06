"""一个档位全部参考（重定向出来的 motion parquet）的手部盘点（训练机上经 scripts/dev/in-flash-chord 在容器里跑，只用 numpy 与 pyarrow，只读；只报不判）。
起因：stages/evaluate/ep_drop_facts.py 的 Tier 2 ep25 记录（results/t2dg/ep25_facts_mf4090.txt）里，参考的 Dex3 手指关节全程是 0（手张开），
而参考要把瓶子单手举高 0.67 m。这里看同一档位的每一集是不是都这样，以及各集“单手托着物体悬空”的帧有多少，供与各集评测结果对照。
每集一行（制表符分隔）：序列、帧数、网格半径、各手手指关节绝对值之和的最大值、各手接触帧数、两手同时接触的帧数、
物体最大抬高（相对第 0 帧）、“物体比第 0 帧高 --lift-m 以上且只有一只手在接触”的帧数、“物体高 --lift-m 以上且没有手接触”的帧数。
用法（容器内）：python /repo/stages/evaluate/ref_hand_survey.py <档位 motion 根，如 /repo/runs/motion/tier2> [--lift-m 0.05]   最后一行 REF_HAND_SURVEY_DONE
"""
import argparse
import glob
import sys

import numpy as np
import pyarrow.parquet as pq


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--lift-m", type=float, default=0.05)
    a = ap.parse_args()
    files = sorted(glob.glob(f"{a.root}/whole_body/soma/sequence_id=*/robot_name=g1/data.parquet"))
    if not files:
        print(f"{a.root} 下没有 motion parquet")
        return 2
    cols = ["sequence_id", "object_body_position", "hand_sides", "hand_contact_active", "robot_joint_names",
            "robot_joint_positions", "object_mesh_radius"]
    print("序列\t帧\t网格半径 m\t左手指最大 rad\t右手指最大 rad\t左接触帧\t右接触帧\t双手接触帧\t最大抬高 m\t单手托举帧\t无手悬空帧")
    for f in files:
        row = pq.read_table(f, columns=cols).to_pylist()[0]
        obj = np.asarray(row["object_body_position"], dtype=float)[:, 0, :]
        T = obj.shape[0]
        sides = row["hand_sides"]
        c = np.asarray(row["hand_contact_active"], dtype=bool)
        jn = list(row["robot_joint_names"])
        q = np.asarray(row["robot_joint_positions"], dtype=float)
        fmax = {}
        for s in ("left", "right"):
            ix = [i for i, n in enumerate(jn) if n.startswith(f"{s}_hand_")]
            fmax[s] = float(np.abs(q[:, ix]).sum(axis=1).max()) if ix else float("nan")
        cs = {s: c[i] for i, s in enumerate(sides)}
        left = cs.get("left", np.zeros(T, bool))
        right = cs.get("right", np.zeros(T, bool))
        dz = obj[:, 2] - obj[0, 2]
        up = dz > a.lift_m
        one = up & (left ^ right)
        none = up & ~left & ~right
        print(f"{row['sequence_id']}\t{T}\t{row['object_mesh_radius'][0]:.3f}\t{fmax['left']:.3f}\t{fmax['right']:.3f}\t"
              f"{int(left.sum())}\t{int(right.sum())}\t{int((left & right).sum())}\t{dz.max():+.3f}\t{int(one.sum())}\t{int(none.sum())}")
    print("REF_HAND_SURVEY_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
