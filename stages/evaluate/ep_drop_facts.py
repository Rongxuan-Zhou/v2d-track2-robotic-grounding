"""一集“物体被碰掉 / 掉下去”的事实盘点（训练机上经 scripts/dev/in-flash-chord 在容器里跑，只用 numpy 与 pyarrow，只读已有文件；只报不判）。
起因：agent-doc/strategy-2026-10-19.md 第 2 节第 1 条：Tier 2 ep25 训好的策略在 4096 个世界里几乎都把瓶子弄到地上，下一步先诊断它为什么掉。
三份输入，各报各的：
  1. 参考（重定向出来的 motion parquet，训练与评测跟的就是它）：物体高度与相对第 0 帧的位移、两只手的接触标记、手掌到物体中心的距离、
     各手手指关节的张合（各关节角绝对值之和）、IK 误差，按 --every 帧列表，另报事件帧（物体第一次抬高 > 2 cm、接触段的起止、物体最高的帧）。
  2. 评测器的 4096 世界 dump（fc_eval 存的 eval_dump/*.npz）：tracking_error 各项按帧的世界分位数；每个世界物体位置误差第一次超过 --pos-m 的帧
     的分布（相对参考事件帧）；completion_step 的分布。
  3. 可选：diag_rollout.py 存的 rollout npz（含 achieved_pose_w）：各帧物体低于第 0 帧高度 --drop-m 以上的世界比例，以及第一次低于它的帧的分布。
门槛只用于描述，不进任何方法。不读别的档位的文件（参考与 dump 都是同一档位、同一集的）。
用法（容器内）：python /repo/stages/evaluate/ep_drop_facts.py --motion <data.parquet> --dump <eval_dump.npz> [--rollout <rollout.npz>]
  [--every 25] [--pos-m 0.10] [--drop-m 0.30]   最后一行 EP_DROP_FACTS_DONE
"""
import argparse
import sys

import numpy as np
import pyarrow.parquet as pq


def q(x: np.ndarray, qs=(10, 50, 90)) -> str:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return "（无）"
    return " / ".join(f"{np.percentile(x, p):.3f}" for p in qs)


def segments(mask: np.ndarray) -> list[tuple[int, int]]:
    out, start = [], None
    for t, m in enumerate(mask):
        if m and start is None:
            start = t
        if not m and start is not None:
            out.append((start, t - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def reference(path: str, every: int) -> dict:
    row = pq.read_table(path).to_pylist()[0]
    obj = np.asarray(row["object_body_position"], dtype=float)[:, 0, :]          # (T,3)
    T = obj.shape[0]
    sides = row["hand_sides"]
    contact = np.asarray(row["hand_contact_active"], dtype=bool)                  # (H,T)
    ee = np.asarray(row["ee_pose_w"], dtype=float)                                # (T,E,7)
    ee_names = row["ee_link_names"]
    # 手指张合取机器人关节里的 Dex3 手指（<side>_hand_*_joint）；hand_finger_joints 列在 G1 的 parquet 里是空的
    jn = list(row["robot_joint_names"])
    q_rob = np.asarray(row["robot_joint_positions"], dtype=float)                 # (T,J)
    fidx = [[i for i, n in enumerate(jn) if n.startswith(f"{s}_hand_")] for s in sides]
    fingers = [np.abs(q_rob[:, ix]).sum(axis=1) if ix else np.full(T, np.nan) for ix in fidx]  # 每手 (T,)
    ik = np.asarray(row["ik_error_per_frame"], dtype=float)
    dz = obj[:, 2] - obj[0, 2]
    disp = np.linalg.norm(obj - obj[0], axis=1)
    palm = {}
    for i, n in enumerate(ee_names):
        palm[n] = np.linalg.norm(ee[:, i, :3] - obj, axis=1)
    print(f"参考 {path}")
    print(f"  {T} 帧、fps {row['fps']}、物体 {row['object_name']}、网格半径 {row['object_mesh_radius']}；第 0 帧物体中心 {np.round(obj[0], 3).tolist()}")
    lift = np.flatnonzero(dz > 0.02)
    ev = {"T": T, "lift": int(lift[0]) if lift.size else -1, "top": int(np.argmax(dz))}
    print(f"  手指关节：{[[jn[i] for i in ix] for ix in fidx]}")
    print(f"  物体第一次抬高 > 2 cm：第 {ev['lift']} 帧；最高在第 {ev['top']} 帧（+{dz.max():.3f} m）；最后一帧高度变化 {dz[-1]:+.3f} m、位移 {disp[-1]:.3f} m")
    for h, s in enumerate(sides):
        segs = segments(contact[h])
        print(f"  {s} 手接触 {int(contact[h].sum())}/{T} 帧，段：{segs[:12]}{' …' if len(segs) > 12 else ''}")
        if segs:
            ev[f"c_{s}"] = segs[0][0]
    print(f"  IK 误差 分位 10/50/90：{q(ik)}")
    head = "  帧\t高度变化 m\t位移 m\t" + "\t".join(f"{s}接触" for s in sides) + "\t" + "\t".join(f"{n}→物体 m" for n in ee_names) \
        + "\t" + "\t".join(f"{s}手指张合 rad" for s in sides) + "\tIK"
    print(head)
    marks = sorted(set(list(range(0, T, every)) + [v for k, v in ev.items() if k != "T" and v >= 0] + [T - 1]))
    for t in marks:
        cells = [f"{dz[t]:+.3f}", f"{disp[t]:.3f}"] + [str(int(contact[h, t])) for h in range(len(sides))] \
            + [f"{palm[n][t]:.3f}" for n in ee_names] + [f"{fingers[h][t]:.3f}" for h in range(len(sides))] + [f"{ik[t]:.3f}"]
        print(f"  {t}\t" + "\t".join(cells))
    return ev


def dump(path: str, ev: dict, every: int, pos_m: float) -> None:
    z = np.load(path)
    err = z["tracking_error"].astype(float)          # (T,W,6)
    names = [str(n) for n in z["error_names"]]
    T, W, _ = err.shape
    print(f"评测 dump {path}：{T} 帧、{W} 个世界；误差项 {names}；门槛 {z['thresholds'].tolist()}；生效帧 {z['active_after_step'].tolist()}")
    print(f"  MP 成功 {int(z['mp_success'].sum())}/{W}、SP 成功 {int(z['sp_success'].sum())}/{W}；terminated {int(z['terminated'].sum())}、"
          f"truncated {int(z['truncated'].sum())}；completion_step 分位 10/50/90 {q(z['completion_step'].astype(float))}")
    ip = names.index("object_position_m") if "object_position_m" in names else 0
    pe = err[:, :, ip]
    over = pe > pos_m
    first = np.where(over.any(axis=0), over.argmax(axis=0), -1)
    hit = first[first >= 0]
    print(f"  物体位置误差第一次 > {pos_m} m 的帧：{hit.size}/{W} 个世界有；分位 10/25/50/75/90 {q(hit.astype(float), (10, 25, 50, 75, 90))}")
    if ev.get("lift", -1) >= 0 and hit.size:
        rel = hit - ev["lift"]
        print(f"  相对参考抬起帧（第 {ev['lift']} 帧）：早于抬起 {int((rel < 0).sum())}、抬起后 0–29 帧 {int(((rel >= 0) & (rel < 30)).sum())}、"
              f"30–89 帧 {int(((rel >= 30) & (rel < 90)).sum())}、90 帧以后 {int((rel >= 90).sum())}")
    print("  帧\t" + "\t".join(f"{n} 10/50/90" for n in names))
    for t in sorted(set(list(range(0, T, every)) + [T - 1])):
        print(f"  {t}\t" + "\t".join(q(err[t, :, k]) for k in range(len(names))))


def rollout(path: str, ev: dict, every: int, drop_m: float) -> None:
    z = np.load(path)
    ach = z["achieved_pose_w"][:, :, 0, :3].astype(float)    # (T,W,3)
    ref = z["reference_pose_w"][:, 0, :3].astype(float)      # (T,3)
    T, W, _ = ach.shape
    dz = ach[:, :, 2] - ach[0:1, :, 2]
    low = dz < -drop_m
    first = np.where(low.any(axis=0), low.argmax(axis=0), -1)
    hit = first[first >= 0]
    print(f"rollout {path}：{T} 帧、{W} 个世界、策略 {z['policy']}")
    disp_all = np.linalg.norm(ach - ach[0:1], axis=-1)        # (T,W) 相对各自第 0 帧
    print(f"  仿真物体相对第 0 帧的最大位移（全部 {T} 帧、全部 {W} 个世界）：{disp_all.max():.4f} m；最大高度变化 {dz.max():+.4f} / 最小 {dz.min():+.4f} m")
    print(f"  物体比第 0 帧低 {drop_m} m 以上：{hit.size}/{W} 个世界出现过；第一次出现的帧 分位 10/25/50/75/90 {q(hit.astype(float), (10, 25, 50, 75, 90))}")
    moved = disp_all > 0.05
    first_mv = np.where(moved.any(axis=0), moved.argmax(axis=0), -1)
    hm = first_mv[first_mv >= 0]
    print(f"  仿真物体离开起点 > 0.05 m：{hm.size}/{W} 个世界出现过；第一次出现的帧 分位 10/25/50/75/90 {q(hm.astype(float), (10, 25, 50, 75, 90))}")
    print("  帧\t参考高度变化 m\t仿真高度变化 10/50/90 m\t低于门槛的世界比例\t仿真离起点 10/50/90 m\t离起点 > 0.05 m 的世界比例\t仿真与参考水平距离 10/50/90 m")
    for t in sorted(set(list(range(0, T, every)) + [T - 1])):
        hd = np.linalg.norm(ach[t, :, :2] - ref[t, :2], axis=1)
        print(f"  {t}\t{ref[t, 2] - ref[0, 2]:+.3f}\t{q(dz[t])}\t{low[t].mean():.3f}\t{q(disp_all[t])}\t{moved[t].mean():.3f}\t{q(hd)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True)
    ap.add_argument("--dump")
    ap.add_argument("--rollout", nargs="*", default=[])
    ap.add_argument("--every", type=int, default=25)
    ap.add_argument("--pos-m", type=float, default=0.10)
    ap.add_argument("--drop-m", type=float, default=0.30)
    a = ap.parse_args()
    ev = reference(a.motion, a.every)
    if a.dump:
        dump(a.dump, ev, a.every, a.pos_m)
    for r in a.rollout:
        rollout(r, ev, a.every, a.drop_m)
    print("EP_DROP_FACTS_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
