"""参考里支撑脚有没有在地上滑（训练机上经 scripts/dev/in-flash-chord 在容器里跑，只用 CPU；只读，只报不判）。
起因：agent-doc/plan-t12lift-2026-10-20.md 执行记录：Tier 2 全失败的集都摔了，参考的速度、偏航、骨盆高度、腰部限位都分不开成败。
一只脚踩在地上却在水平方向滑，机器人照着跟就得用别的办法补，可能失衡。这里用 URDF 正向运动学算两只脚（<side>_ankle_roll_link 的碰撞网格顶点里最低的那些点）每帧的位置：
某帧某只脚的最低点离这一集这只脚的最低点 < --stance-m（默认 2 cm）就算这只脚在支撑；支撑脚的水平速度按 --rate 帧/秒回放算（相邻两帧都支撑才算）。
每集一行：序列、帧、参考第一次接触 t_c；[0, t_c + --after) 与整段里：支撑脚水平滑速的中位、95% 分位（m/s）、滑速 > --fast（默认 0.3 m/s）的支撑帧比例、两只脚都不支撑的帧比例。
用法（容器内）：python /repo/stages/retarget/ref_foot_slide.py <档位 motion 根> [--stance-m 0.02] [--rate 50] [--after 60] [--fast 0.3]   最后一行 REF_FOOT_SLIDE_DONE
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finger_close import URDF, Robot, quat_mat


def stats(v: np.ndarray, stance_pairs: np.ndarray, both_off: np.ndarray, fast: float) -> str:
    s = v[stance_pairs]
    if s.size == 0:
        return "—\t—\t—\t" + f"{both_off.mean():.3f}"
    return f"{np.median(s):.3f}\t{np.percentile(s, 95):.3f}\t{(s > fast).mean():.3f}\t{both_off.mean():.3f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--stance-m", type=float, default=0.02)
    ap.add_argument("--rate", type=float, default=50.0)
    ap.add_argument("--after", type=int, default=60)
    ap.add_argument("--fast", type=float, default=0.3)
    ap.add_argument("--urdf", default=URDF)
    a = ap.parse_args()
    rob = Robot(a.urdf)
    feet = {s: f"{s}_ankle_roll_link" for s in ("left", "right")}
    fpts = {}
    for s, ln in feet.items():
        try:
            v = rob.link_points(ln, 400)
        except (OSError, ValueError, AttributeError):
            v = np.zeros((1, 3))
        fpts[s] = v
    files = sorted(glob.glob(f"{a.root}/whole_body/soma/sequence_id=*/robot_name=g1/data.parquet"))
    print(f"支撑：脚的最低点离本集这只脚的最低点 < {a.stance_m} m；回放 {a.rate} 帧/秒；前段 = [0, t_c + {a.after})；快滑 > {a.fast} m/s")
    print("序列\t帧\tt_c\t前段 滑速中位\t前段 95%\t前段 快滑比例\t前段 双脚离地比例\t整段 滑速中位\t整段 95%\t整段 快滑比例\t整段 双脚离地比例")
    for f in files:
        row = pq.read_table(f, columns=["sequence_id", "robot_joint_names", "robot_joint_positions", "robot_root_position", "robot_root_wxyz", "hand_contact_active"]).to_pylist()[0]
        names = list(row["robot_joint_names"])
        q = np.asarray(row["robot_joint_positions"], dtype=float)
        T = q.shape[0]
        L = rob.fk(np.asarray(row["robot_root_position"], dtype=float), quat_mat(np.asarray(row["robot_root_wxyz"], dtype=float), "wxyz"),
                   {n: q[:, i] for i, n in enumerate(names)}, set(feet.values()))
        low, xy = {}, {}
        for s, ln in feet.items():
            p, R = L[ln]
            pw = p[:, None, :] + np.einsum("tij,vj->tvi", R, fpts[s])        # (T,V,3)
            k = np.argsort(pw[:, :, 2], axis=1)[:, :10]                         # 每帧最低的 10 个点
            sole = np.take_along_axis(pw, k[:, :, None], axis=1).mean(1)       # (T,3)
            low[s], xy[s] = sole[:, 2], sole[:, :2]
        stance = {s: low[s] - low[s].min() < a.stance_m for s in feet}
        vel = {s: np.r_[0.0, np.linalg.norm(np.diff(xy[s], axis=0), axis=1)] * a.rate for s in feet}
        pair = {s: stance[s] & np.r_[False, stance[s][:-1]] for s in feet}
        v_all = np.concatenate([vel[s] for s in feet])
        p_all = np.concatenate([pair[s] for s in feet])
        off = ~(stance["left"] | stance["right"])
        c = np.asarray(row["hand_contact_active"], dtype=bool)
        t_c = min((int(np.flatnonzero(x)[0]) for x in c if x.any()), default=T)
        hi = min(T, t_c + a.after)
        v_pre = np.concatenate([vel[s][:hi] for s in feet])
        p_pre = np.concatenate([pair[s][:hi] for s in feet])
        print(f"{row['sequence_id']}\t{T}\t{t_c}\t{stats(v_pre, p_pre, off[:hi], a.fast)}\t{stats(v_all, p_all, off, a.fast)}")
    print("REF_FOOT_SLIDE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
