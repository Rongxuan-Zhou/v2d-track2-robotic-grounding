"""参考里被拿着的物体与手掌的朝向是不是一起转（训练机上经 scripts/dev/in-flash-chord 在容器里跑，只用 CPU；只读参考 parquet，只报不判）。
起因：agent-doc/plan-t3fix2-2026-10-28.md 事实 2：ep15 拿着那段第 497–502 帧物体朝向切换 168°，撤销切换后段尾与放下后歇着的位姿差 119°，
即这次转动多半是真的、只是重建把它压成了几帧。要把它摊开慢慢转，先看手掌在哪几帧转：被拿着的刚体应随手一起转。
逐帧算（每只手）：手掌每帧转角；物体每帧转角；物体在手掌系里的相对朝向 Q_rel = R_palmᵀ R_obj 相对 --ref 帧的夹角（rel 角）。
报：--range a:b:s 里每 s 帧一行；另报 --range 内手掌相对 --ref 帧累计转角最大的帧与角度（两只手各一）。
读 ee_pose_w（手掌位姿，四元数 wxyz）、ee_link_names、object_body_wxyz、hand_contact_active、hand_sides。
用法（容器内）：python /repo/stages/retarget/ref_hand_object_turn.py --ref 480 --range 400:620:5 <motion parquet>...   最后一行 REF_TURN_DONE
"""
import argparse
import sys

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from ref_hand_object_slip import quat_mat  # noqa: E402


def ang(A, B):
    """逐帧两个旋转矩阵的夹角（度）。"""
    tr = np.einsum("nij,nij->n", A, B)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", type=int, required=True, help="相对朝向的参照帧")
    ap.add_argument("--range", required=True, help="a:b:s")
    ap.add_argument("parquet", nargs="+")
    a = ap.parse_args()
    s0, s1, st = (int(x) for x in a.range.split(":"))
    for path in a.parquet:
        row = pq.read_table(path).to_pylist()[0]
        sides = list(row["hand_sides"])
        contact = {s: np.asarray(c, dtype=bool) for s, c in zip(sides, row["hand_contact_active"])}
        ee = np.asarray(row["ee_pose_w"], dtype=float)
        names = list(row["ee_link_names"])
        oR = quat_mat(np.asarray(row["object_body_wxyz"], dtype=float)[:, 0], "wxyz")
        T = len(oR)
        r = a.ref
        o_turn = np.r_[0.0, ang(oR[1:], oR[:-1])]
        o_from = ang(oR, np.repeat(oR[r:r + 1], T, axis=0))
        cols, per = [], {}
        for s in sides:
            pR = quat_mat(ee[:, names.index(f"{s}_hand_palm_link"), 3:], "wxyz")
            rel = np.einsum("nji,njk->nik", pR, oR)
            per[s] = (np.r_[0.0, ang(pR[1:], pR[:-1])], ang(pR, np.repeat(pR[r:r + 1], T, axis=0)), ang(rel, np.repeat(rel[r:r + 1], T, axis=0)))
            cols += [f"{s}接触", f"{s}掌每帧°", f"{s}掌相对参照°", f"{s} rel 相对参照°"]
        print(f"{path}：{T} 帧；参照帧 {r}")
        for s in sides:
            seg = per[s][1][s0:min(s1, T)]
            k = int(np.argmax(seg))
            print(f"  {s} 手掌在第 {s0}–{min(s1, T)} 帧相对参照帧累计转角最大 {seg[k]:.1f}°（第 {s0 + k} 帧）；物体同一帧相对参照 {o_from[s0 + k]:.1f}°")
        print("  帧\t物体每帧°\t物体相对参照°\t" + "\t".join(cols))
        for f in range(s0, min(s1, T), st):
            vals = []
            for s in sides:
                tp, fp, fr = per[s]
                vals += [str(int(contact[s][f])), f"{tp[f]:.1f}", f"{fp[f]:.1f}", f"{fr[f]:.1f}"]
            print(f"  {f}\t{o_turn[f]:.1f}\t{o_from[f]:.1f}\t" + "\t".join(vals))
    print("REF_TURN_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
