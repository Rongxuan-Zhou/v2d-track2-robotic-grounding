"""Tier 1 参考里首次接触后一小段的转动拆分（训练机上经 scripts/dev/in-flash-chord 在容器里跑，只用 CPU；只读 Tier 1 的重定向参考）。
接 t1_grasp_consistency.py（results/t1diag/grasp_consistency_mf4090.txt：接触且物体在动的帧里，物体相对掌心的朝向漂移跟丢组中位 102.5°、其余 27.4°）。
跟丢集的物体多在首次接触后 10–25 帧越线（plan-r3o-2026-10-15.md 第 8 节），所以这里只看每只手首次接触后的 W 帧（默认 30），报三个转角
（各帧相对窗口第一帧的转角，取窗口内最大值）：物体在世界里转了多少、掌心在世界里转了多少、物体相对掌心转了多少；另报窗口内物体平移与掌心平移。
物体相对掌心转得多、而物体自己在世界里转得少，说明是掌心在扭（重定向的腕部朝向与物体脱节）；物体自己转得多，说明参考要的是手里转动或滑动。
掌心位姿按官方 schema 的 _POSE7 = [x, y, z, qw, qx, qy, qz]；物体朝向取 object_body_wxyz 的第 0 个部件；episode 号取 runs/motion_src/<目录>/<序列>/v2d_meta.json。
用法（容器内）：python /repo/stages/evaluate/t1_contact_window.py --lost 1,2,4,9,11,25,28 [--window 30] <parquet>...；最后一行 T1_CONTACT_WINDOW_DONE。
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = (q / np.linalg.norm(q, axis=-1, keepdims=True)).T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def max_angle_from_first(R: np.ndarray) -> float:
    tr = np.einsum("nij,ij->n", R, R[0])
    return float(np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))).max())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lost", required=True)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("parquets", nargs="+")
    a = ap.parse_args()
    lost = {int(x) for x in a.lost.split(",")}
    groups = {"跟丢": [], "其余": []}
    for p in a.parquets:
        r = pq.read_table(p, columns=["ee_pose_w", "object_root_position", "object_body_wxyz", "hand_contact_active"]).to_pylist()[0]
        ee = np.asarray(r["ee_pose_w"], float)
        obj = np.asarray(r["object_root_position"], float)
        Ro = quat_to_mat(np.asarray([b[0] for b in r["object_body_wxyz"]], float))
        c = np.asarray(r["hand_contact_active"], float)
        c = c.T if c.shape[0] != len(obj) else c
        m = re.search(r"/runs/motion/([^/]+)/whole_body/soma/sequence_id=([^/]+)/", p)
        meta = Path(p.split("/runs/motion/")[0]) / "runs" / "motion_src" / m.group(1) / m.group(2) / "v2d_meta.json"
        ep = int(json.loads(meta.read_text())["episode"])
        parts = []
        best = None
        for h, side in enumerate(("左", "右")):
            on = np.flatnonzero(c[:, h] > 0.5)
            if not len(on):
                parts.append(f"{side}手没有接触")
                continue
            f0 = int(on[0]); f1 = min(f0 + a.window, len(obj))
            Rp = quat_to_mat(ee[f0:f1, h, 3:7])
            Rrel = np.einsum("nji,njk->nik", Rp, Ro[f0:f1])
            o_rot, p_rot, rel_rot = max_angle_from_first(Ro[f0:f1]), max_angle_from_first(Rp), max_angle_from_first(Rrel)
            o_mv = float(np.linalg.norm(obj[f0:f1] - obj[f0], axis=1).max() * 100)
            p_mv = float(np.linalg.norm(ee[f0:f1, h, :3] - ee[f0, h, :3], axis=1).max() * 100)
            parts.append(f"{side}手首次接触第 {f0} 帧，窗口 {f1 - f0} 帧：物体自转 {o_rot:.1f}°、掌心自转 {p_rot:.1f}°、物体相对掌心 {rel_rot:.1f}°；"
                         f"物体平移 {o_mv:.1f} cm、掌心平移 {p_mv:.1f} cm")
            if best is None or f0 < best[0]:
                best = (f0, o_rot, p_rot, rel_rot)
        print(f"ep{ep}{'（跟丢）' if ep in lost else ''}  " + "；".join(parts))
        if best:
            groups["跟丢" if ep in lost else "其余"].append(best[1:])
    for g, v in groups.items():
        if v:
            arr = np.asarray(v)
            print(f"{g} {len(v)} 集（取先接触的那只手）：物体自转中位 {np.median(arr[:, 0]):.1f}°、掌心自转中位 {np.median(arr[:, 1]):.1f}°、"
                  f"物体相对掌心中位 {np.median(arr[:, 2]):.1f}°")
    print("T1_CONTACT_WINDOW_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
