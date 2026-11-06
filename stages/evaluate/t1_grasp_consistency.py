"""Tier 1 参考里“抓着的物体在手里动了多少”（训练机上经 scripts/dev/in-flash-chord 在容器里跑，只用 CPU；只读 Tier 1 的重定向参考）。
用途：r3o 判“没用”之后的 Tier 1 诊断（agent-doc/plan-r3o-2026-10-15.md 第 3 节“没用 → 另写计划”）。7 个跟丢集里 5 个是物体位置先越线，
越线中位在第 117–129 步，刚接触不久。假设：重定向后的参考里，接触期间物体相对机器人掌心并不固定（人手比例与 G1 不同），
策略要同时跟住掌心与物体就做不到，于是一接触就丢。
量法：每集、每只手，取“这只手标记为接触（hand_contact_active > 0.5）且参考里物体在动（物体根速度 > 0.05 m/s）”的帧，
把物体根位置与朝向换到掌心坐标系（ee_pose_w 的掌心位姿），报这些帧里物体在掌心系中的位置离其中位的距离（中位、p90，cm）
与朝向离其中位的角度（中位、p90，度）；另报这类帧数、第一个接触帧。跟丢集与其余集分组给中位。
参考的列：robot_root_position、ee_pose_w（掌心 [x y z qw qx qy qz]，world）、object_root_position、object_body_wxyz（第 0 个部件）、
hand_contact_active（手 × 帧）、fps。掌心位姿按官方 schema 的 _POSE7 = [x, y, z, qw, qx, qy, qz]（motion_schema/schema.py 第 47 行）。episode 号取 runs/motion_src/<目录>/<序列>/v2d_meta.json。
用法（容器内）：python /repo/stages/evaluate/t1_grasp_consistency.py --lost 1,2,4,9,11,25,28 <parquet>...；最后一行 T1_GRASP_DONE。
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


def angle(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    tr = np.einsum("nij,ij->n", Ra, Rb)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def episode(p: str) -> dict:
    r = pq.read_table(p, columns=["fps", "ee_pose_w", "object_root_position", "object_body_wxyz", "hand_contact_active",
                                  "sequence_id"]).to_pylist()[0]
    fps = float(r["fps"])
    ee = np.asarray(r["ee_pose_w"], float)                      # 帧 × 手 × 7
    obj = np.asarray(r["object_root_position"], float)          # 帧 × 3
    oq = np.asarray([b[0] for b in r["object_body_wxyz"]], float)  # 帧 × 4（第 0 个部件）
    c = np.asarray(r["hand_contact_active"], float)
    c = c.T if c.shape[0] != len(obj) else c                    # 帧 × 手
    speed = np.r_[0.0, np.linalg.norm(np.diff(obj, axis=0), axis=1) * fps]
    Ro = quat_to_mat(oq)
    out = {"seq": str(r["sequence_id"]), "frames": len(obj), "hands": []}
    for h in range(ee.shape[1]):
        m = (c[:, h] > 0.5) & (speed > 0.05)
        first = int(np.argmax(c[:, h] > 0.5)) if (c[:, h] > 0.5).any() else None
        if m.sum() < 10:
            out["hands"].append({"n": int(m.sum()), "first": first})
            continue
        Rp = quat_to_mat(ee[:, h, 3:7])
        rel = np.einsum("nji,nj->ni", Rp[m], obj[m] - ee[m, h, :3])     # 物体在掌心系里的位置
        Rrel = np.einsum("nji,njk->nik", Rp[m], Ro[m])                   # 物体在掌心系里的朝向
        dpos = np.linalg.norm(rel - np.median(rel, axis=0), axis=1) * 100
        k = int(np.argmin(np.abs(dpos - np.median(dpos))))               # 取接近中位的一帧作朝向参照
        drot = angle(Rrel, Rrel[k])
        out["hands"].append({"n": int(m.sum()), "first": first, "pos50": float(np.median(dpos)), "pos90": float(np.percentile(dpos, 90)),
                             "rot50": float(np.median(drot)), "rot90": float(np.percentile(drot, 90))})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lost", required=True)
    ap.add_argument("parquets", nargs="+")
    a = ap.parse_args()
    lost = {int(x) for x in a.lost.split(",")}
    groups = {"跟丢": [], "其余": []}
    for p in a.parquets:
        e = episode(p)
        m = re.search(r"/runs/motion/([^/]+)/whole_body/soma/sequence_id=([^/]+)/", p)
        meta = Path(p.split("/runs/motion/")[0]) / "runs" / "motion_src" / m.group(1) / m.group(2) / "v2d_meta.json" if m else None
        ep = int(json.loads(meta.read_text())["episode"]) if meta and meta.exists() else None
        name = f"ep{ep} {e['seq']}" + ("（跟丢）" if ep in lost else "")
        tag = ""
        for h, side in zip(e["hands"], ("左", "右")):
            if "pos50" in h:
                tag += f"  {side}手 {h['n']} 帧，首次接触第 {h['first']} 帧，位置漂 中位 {h['pos50']:.1f} / p90 {h['pos90']:.1f} cm，朝向漂 中位 {h['rot50']:.1f} / p90 {h['rot90']:.1f}°"
            else:
                tag += f"  {side}手 动着接触 {h['n']} 帧（不足 10，不算）"
        print(f"{name}{tag}")
        best = max((h for h in e["hands"] if "pos50" in h), key=lambda h: h["n"], default=None)
        if best is not None:
            groups["跟丢" if ep in lost else "其余"].append((best["pos50"], best["rot50"]))
    for g, v in groups.items():
        if v:
            arr = np.asarray(v)
            print(f"{g} {len(v)} 集（取动着接触帧最多的那只手）：位置漂中位的中位 {np.median(arr[:, 0]):.1f} cm，朝向漂中位的中位 {np.median(arr[:, 1]):.1f}°")
    print("T1_GRASP_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
