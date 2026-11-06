"""接触前后机器人跟得多准、什么时候摔（训练机上经 scripts/dev/in-flash-chord 在容器里跑，只用 numpy 与 pyarrow；只读已有文件，只报不判）。
起因：agent-doc/plan-t2dg-2026-10-20.md 执行记录：Tier 2 ep4 的骨盆朝向误差第 80–120 帧从 0.19 升到 1.51 rad（机器人在参考接触之前就摔了），
ep25 抬起时手掌离参考手掌 11–13 cm。要与同一档位成功的集比，才知道这些数大不大。
输入：同一集的参考（motion parquet，只取 hand_contact_active）与一份逐帧逐世界的跟踪误差（fc_eval_dump.sh 的 eval_dump/*.npz：tracking_error + error_names；
或 diag_rollout.py 的 rollout npz：tracking_error + tracking_error_names）。
每份一行：t_c（参考两只手第一次接触的帧）；[t_c − 30, t_c)、[t_c, t_c + 30) 两段里手掌位置误差、骨盆位置误差的世界中位（先对帧取中位、再报世界的 10/50/90 分位）；
“摔倒帧”= 骨盆朝向误差的世界中位第一次 > --fall-rad（默认 0.5）的帧（没有就记 —），以及它相对 t_c 的帧数；
骨盆朝向误差世界中位的最大值。--label 给这一行加个名字。用法（容器内）：
  python /repo/stages/evaluate/contact_window_errors.py --label <名> --motion <parquet> --npz <npz> [--fall-rad 0.5] [--win 30]   最后一行 CONTACT_WINDOW_DONE
可重复给多组 --label/--motion/--npz（按顺序配对）。
"""
import argparse
import sys

import numpy as np
import pyarrow.parquet as pq


def q3(x: np.ndarray) -> str:
    return " / ".join(f"{np.percentile(x, p):.3f}" for p in (10, 50, 90)) if x.size else "（无）"


def one(label: str, motion: str, npz: str, fall_rad: float, win: int) -> None:
    c = np.asarray(pq.read_table(motion, columns=["hand_contact_active"]).to_pylist()[0]["hand_contact_active"], dtype=bool)
    z = np.load(npz)
    names = [str(n) for n in (z["error_names"] if "error_names" in z.files else z["tracking_error_names"])]
    err = z["tracking_error"].astype(float)            # (T,W,6)
    T = err.shape[0]
    t_c = min((int(np.flatnonzero(x)[0]) for x in c if x.any()), default=-1)
    ip, iv, io = names.index("palm_position_m"), names.index("pelvis_position_m"), names.index("pelvis_orientation_rad")
    out = [label, str(T), str(t_c)]
    for lo, hi in ((t_c - win, t_c), (t_c, t_c + win)):
        lo, hi = max(lo, 0), min(hi, T)
        for i in (ip, iv):
            out.append(q3(np.median(err[lo:hi, :, i], axis=0)) if t_c >= 0 and hi > lo else "（无）")
    ori = np.median(err[:, :, io], axis=1)
    fall = np.flatnonzero(ori > fall_rad)
    f0 = int(fall[0]) if fall.size else -1
    out += [str(f0) if f0 >= 0 else "—", f"{f0 - t_c:+d}" if f0 >= 0 and t_c >= 0 else "—", f"{ori.max():.3f}"]
    print("\t".join(out))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", action="append", required=True)
    ap.add_argument("--motion", action="append", required=True)
    ap.add_argument("--npz", action="append", required=True)
    ap.add_argument("--fall-rad", type=float, default=0.5)
    ap.add_argument("--win", type=int, default=30)
    a = ap.parse_args()
    if not len(a.label) == len(a.motion) == len(a.npz):
        print("--label、--motion、--npz 的个数要一样")
        return 2
    print(f"窗口 {a.win} 帧，摔倒门槛 {a.fall_rad} rad（骨盆朝向误差的世界中位）")
    print("名\t帧\tt_c\t接触前 手掌误差 10/50/90 m\t接触前 骨盆误差\t接触后 手掌误差\t接触后 骨盆误差\t摔倒帧\t摔倒帧 − t_c\t骨盆朝向误差中位最大 rad")
    for lab, m, n in zip(a.label, a.motion, a.npz):
        one(lab, m, n, a.fall_rad, a.win)
    print("CONTACT_WINDOW_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
