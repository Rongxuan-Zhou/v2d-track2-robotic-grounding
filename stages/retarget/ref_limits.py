"""一个档位全部参考里关节顶在限位上的情况（训练机上经 scripts/dev/in-flash-chord 在容器里跑，只用 CPU；只读，只报不判）。
起因：stages/evaluate/fall_timeline.py 的 Tier 2 ep4 记录：参考的 waist_pitch 从第 90 帧起一直是 +0.52 rad（看来是 G1 腰部俯仰的限位），骨盆同时降到 0.66 m，
机器人第 99 帧摔倒。重定向器在人体姿态超出机器人关节范围时会把关节压在限位上；压在限位上的段，机器人在仿真里也只能贴着限位动，可能难以保持平衡。
每集一行：序列、帧、参考第一次接触 t_c；腿、腰的关节（hip/knee/ankle/waist，共 15 个）里，[0, t_c + --after) 这段“任一关节离限位 < --tol rad”的帧数与比例，
整段的同一数；这段里离限位最近、出现最多的 3 个关节与各自的帧数。手臂与手指不算（只看平衡相关的关节）。
限位取 URDF（FlashCHORD 资产里那份，stages/retarget/finger_close.py 的 Robot）。
用法（容器内）：python /repo/stages/retarget/ref_limits.py <档位 motion 根> [--tol 0.02] [--after 60]   最后一行 REF_LIMITS_DONE
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finger_close import URDF, Robot

KEYS = ("hip", "knee", "ankle", "waist")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--after", type=int, default=60)
    ap.add_argument("--urdf", default=URDF)
    a = ap.parse_args()
    rob = Robot(a.urdf)
    files = sorted(glob.glob(f"{a.root}/whole_body/soma/sequence_id=*/robot_name=g1/data.parquet"))
    print(f"离限位 < {a.tol} rad 算顶住；只看腿、腰的关节；前段 = [0, t_c + {a.after})")
    print("序列\t帧\tt_c\t前段顶住帧\t前段比例\t整段顶住帧\t整段比例\t前段顶住最多的关节（帧数）")
    for f in files:
        row = pq.read_table(f, columns=["sequence_id", "robot_joint_names", "robot_joint_positions", "hand_contact_active"]).to_pylist()[0]
        names = list(row["robot_joint_names"])
        q = np.asarray(row["robot_joint_positions"], dtype=float)
        T = q.shape[0]
        c = np.asarray(row["hand_contact_active"], dtype=bool)
        t_c = min((int(np.flatnonzero(x)[0]) for x in c if x.any()), default=T)
        hi = min(T, t_c + a.after)
        sel = [i for i, n in enumerate(names) if any(k in n for k in KEYS)]
        at = np.zeros((T, len(sel)), dtype=bool)
        for j, i in enumerate(sel):
            jt = rob.joints[names[i]]
            if jt["lower"] is None or jt["upper"] is None:
                continue
            at[:, j] = (q[:, i] - jt["lower"] < a.tol) | (jt["upper"] - q[:, i] < a.tol)
        anyf = at.any(axis=1)
        top = sorted(((int(at[:hi, j].sum()), names[i]) for j, i in enumerate(sel)), reverse=True)[:3]
        tops = "，".join(f"{n.removesuffix('_joint')} {k}" for k, n in top if k > 0) or "—"
        print(f"{row['sequence_id']}\t{T}\t{t_c}\t{int(anyf[:hi].sum())}\t{anyf[:hi].mean():.3f}\t{int(anyf.sum())}\t{anyf.mean():.3f}\t{tops}")
    print("REF_LIMITS_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
