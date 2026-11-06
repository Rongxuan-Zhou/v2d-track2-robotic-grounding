"""Tier 1 重定向参考（motion_v1 parquet）的逐序列概况，给训练试点挑序列用（在 FlashCHORD 容器里执行，只用 CPU）。

每条序列：帧数、fps 列、物体净位移 / 离起点最大位移 / 路径长、运动帧占比（物体速度 > 5 cm/s 或转速 > 10°/s，按 fps 换算）、
累计转角、离初始朝向最大角、网格半径、左右手接触帧数。一条 parquet 一行，list 列里是整条序列。
用法：python stages/policy/sequence_stats.py /repo/runs/motion/tier1/whole_body/soma
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

COLS = ["fps", "object_body_position", "object_body_wxyz", "object_mesh_radius", "hand_sides", "hand_contact_active"]


def angle_deg(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    return np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(qa * qb, -1)), 0, 1)))


def main() -> int:
    root = Path(sys.argv[1])
    files = sorted(root.glob("sequence_id=*/robot_name=g1/data.parquet"))
    print(f"{len(files)} 条序列（{root}）")
    print("sequence_id\tframes\tfps\tnet_m\tmaxdisp_m\tpath_m\tmoving\trot_total_deg\trot_max_deg\tradius_m\tcontact_L\tcontact_R")
    for f in files:
        row = pq.read_table(f, columns=COLS).to_pylist()[0]
        fps = float(row["fps"])
        p = np.asarray(row["object_body_position"], dtype=np.float64)[:, 0]      # (T,3)，只看第一个刚体
        q = np.asarray(row["object_body_wxyz"], dtype=np.float64)[:, 0]          # (T,4)
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        step = np.linalg.norm(np.diff(p, axis=0), axis=1)
        rot = angle_deg(q[1:], q[:-1])
        moving = np.mean((step * fps > 0.05) | (rot * fps > 10.0))
        sides = list(row["hand_sides"] or [])
        raw = row["hand_contact_active"]
        act = np.asarray(raw, dtype=bool) if raw is not None else np.zeros((len(p), 0), bool)
        if act.ndim == 2 and act.shape[0] == len(sides) and act.shape[1] == len(p):   # 存成（手, 帧）时转成（帧, 手）
            act = act.T
        contact = {s: int(act[:, i].sum()) for i, s in enumerate(sides)} if act.ndim == 2 and act.shape == (len(p), len(sides)) else {}
        if not contact:
            print(f"# {f.parents[1].name}: hand_contact_active 形状 {act.shape}、hand_sides {sides}，接触帧数不报", file=sys.stderr)
        seq = f.parents[1].name.removeprefix("sequence_id=")
        print(f"{seq}\t{len(p)}\t{fps:g}\t{np.linalg.norm(p[-1] - p[0]):.2f}\t{np.linalg.norm(p - p[0], axis=1).max():.2f}\t{step.sum():.2f}\t"
              f"{moving:.0%}\t{rot.sum():.0f}\t{angle_deg(q, q[:1]).max():.0f}\t{float(np.max(row['object_mesh_radius'])):.2f}\t"
              f"{contact.get('left', '-')}\t{contact.get('right', '-')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
