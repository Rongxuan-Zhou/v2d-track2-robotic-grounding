"""Tier 3 里唯一碰 Tier 1 的地方：只读一集 Tier 1 的第 0 帧（要 pyarrow）。计划 agent-doc/plan-t3align-2026-10-21.md 第 1 节第 1 条、判据 A0。
官方答复（10/21）：“Using the first frame of Tier 1 data to compute alignment is fine, provided the Tier 1 caption isn't used in Tier 3 otherwise”——只算对齐、只用第 0 帧。
读法：parquet 按 frame_index == 0 过滤，只取物体位姿、物体可见、人体平移、人体姿态（只留根节点 Hips 的 3 维）；行数必须恰好 1。
另读 meta/episodes_metadata.jsonl 里这一集的 sequence_id，给调用方核“集没配错”。每次读往 stderr 打一行 TIER1_FRAME0（集号、文件 sha256、读了哪几列）。
Tier 1 的物体位姿存成 [x, y, z, qw, qx, qy, qz]（meta/info.json 的 names），这里转成位置 + xyzw 四元数（归一）。swap_quat=True 是负对照：故意按 xyzw 读错。
测试：stages/evaluate/test_align_frefs_t3.py。
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

COLS = ["frame_index", "observation.object.pose", "observation.object.visible", "observation.human.translation", "observation.human.pose"]


def read(tier1_root: Path, episode: int, swap_quat: bool = False) -> dict:
    tier1_root = Path(tier1_root)
    f = tier1_root / "data" / "chunk-000" / f"episode_{episode:06d}.parquet"
    t = pq.read_table(f, columns=COLS, filters=[("frame_index", "==", 0)])
    if t.num_rows != 1:
        raise SystemExit(f"tier1_frame0：ep{episode} 第 0 帧读到 {t.num_rows} 行，不是 1 行")
    r = t.to_pylist()[0]
    if int(r["frame_index"]) != 0:
        raise SystemExit(f"tier1_frame0：ep{episode} 读到的不是第 0 帧")
    pose = np.asarray(r["observation.object.pose"], dtype=np.float64)
    q = pose[3:7].copy() if swap_quat else np.r_[pose[4:7], pose[3]]
    q /= np.linalg.norm(q)
    vis = r["observation.object.visible"]
    vis = bool(vis[0] if isinstance(vis, (list, tuple)) else vis)
    seq = None
    for line in (tier1_root / "meta" / "episodes_metadata.jsonl").read_text().splitlines():
        m = json.loads(line)
        if int(m["episode_index"]) == episode:
            seq = m["sequence_id"]
            break
    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    print(f"TIER1_FRAME0 ep{episode} {f.name} sha256 {sha[:16]} 只读第 0 帧：{', '.join(COLS[1:])}（人体姿态只留 Hips）", file=sys.stderr)
    return {"episode": episode, "sequence_id": seq, "p": pose[:3], "q_xyzw": q, "visible": vis,
            "human_t": np.asarray(r["observation.human.translation"], dtype=np.float64),
            "human_root_rotvec": np.asarray(r["observation.human.pose"][:3], dtype=np.float64), "file_sha256": sha}
