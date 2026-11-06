"""训好的策略在回放里残差顶没顶到头（训练机上经 scripts/dev/in-flash-chord 在容器里跑，只用 CPU；读 diag_rollout.py --actions 存的 npz；只报不判）。
计划 agent-doc/plan-t12sat-2026-10-20.md。FlashSAC 的确定性动作 = action_scale · tanh(均值)（training.network.action_scale，G1 配方 10），
身体关节目标 = (SONIC 解码 + residual_scale · 动作) · sonic_scale + 默认位姿（runtime/actions/sonic.py 的 process_sonic_joint_targets，use_tanh False）。
所以 |动作| / action_scale 就是 tanh(均值) 的绝对值，接近 1 即顶到头。前 len(residual_joint_names) 个动作是身体残差，其后是手指。
每个 --window a:b（参考帧，含 a 不含 b）报：身体残差 |动作|/action_scale > --sat（默认 0.95）的比例、中位数、p90，手指的同样三项，
以及身体残差里顶到头比例最高的 --top 个关节。action_scale 与关节名从 checkpoint 内嵌的配置读。
--fall-npz（计划 agent-doc/plan-trajopt-2026-10-26.md 第 5 节第 3 版，复核 MAJOR 1–3）：读 fall_mechanism.py --save 的 t_f、t_03，按每个摔了的世界取窗口 pre30 = [t_f−30, t_f)、
pre60 = [t_f−60, t_f−30)、prewhole = [0, t_f)，按腿（hip/knee/ankle）、腰（waist）、手臂（shoulder/elbow/wrist）、手指四组报顶到头比例；
另报每个世界腿与腰第一次顶到头的步与 t_03 的先后（中位与“先于 t_03”的世界比例），以及窗口里落在训练回合长度 --horizon 步以内的帧占比。
用法（容器内）：python /repo/stages/evaluate/residual_sat.py --npz <npz> --policy <policy.safetensors> --window 0:40 --window 40:94 [--sat 0.95] [--top 8] [--label 名]
        [--fall-npz <npz> --horizon <训练回合平均长度>]
最后一行 RESIDUAL_SAT_DONE
"""
import argparse
import json
import struct
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--window", action="append", required=True)
    ap.add_argument("--sat", type=float, default=0.95)
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--label", default="")
    ap.add_argument("--fall-npz")
    ap.add_argument("--horizon", type=float, default=0.0)
    a = ap.parse_args()
    with open(a.policy, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        cfg = json.loads(json.loads(f.read(n))["__metadata__"]["user.resolved_config_json"])
    scale = float(cfg["training"]["network"]["action_scale"])
    names = list(cfg["action"]["residual_joint_names"])
    z = np.load(a.npz)
    if "policy_action" not in z.files:
        print("npz 里没有 policy_action（要 diag_rollout.py --actions）")
        return 2
    act = np.abs(z["policy_action"].astype(float)) / scale        # (N,W',A)
    s0 = int(z["start_frame"])
    nb = len(names)
    print(f"{a.label} {a.npz}：动作 {act.shape}（步 × 世界 × 维），参考起始帧 {s0}；action_scale {scale:g}，身体残差 {nb} 维、手指 {act.shape[2] - nb} 维；"
          f"residual_scale {cfg['action']['residual_scale']}；门槛 |tanh| > {a.sat}")
    print("窗口（参考帧）\t身体：顶到头比例\t中位\tp90\t手指：顶到头比例\t中位\tp90\t身体里顶到头最多的关节（比例）")
    for w in a.window:
        lo, hi = (int(x) for x in w.split(":"))
        i0, i1 = max(lo - s0, 0), min(hi - s0, act.shape[0])
        if i1 <= i0:
            print(f"{lo}:{hi}\t（不在已抄的 {act.shape[0]} 步里）")
            continue
        b, h = act[i0:i1, :, :nb], act[i0:i1, :, nb:]
        per = (b > a.sat).mean(axis=(0, 1))
        top = np.argsort(-per)[: a.top]
        print(f"{lo}:{hi}\t{(b > a.sat).mean():.3f}\t{np.median(b):.3f}\t{np.percentile(b, 90):.3f}\t"
              f"{(h > a.sat).mean():.3f}\t{np.median(h):.3f}\t{np.percentile(h, 90):.3f}\t"
              + "，".join(f"{names[j]} {per[j]:.2f}" for j in top))
    groups = {"腿": ("hip", "knee", "ankle"), "腰": ("waist",), "手臂": ("shoulder", "elbow", "wrist")}
    gidx = {g: [j for j, n in enumerate(names) if any(k in n for k in keys)] for g, keys in groups.items()}
    gidx["手指"] = list(range(nb, act.shape[2]))
    print("全程分组（参考帧 " + f"{s0}:{s0 + act.shape[0]}）：" + "，".join(f"{g} {(act[:, :, ix] > a.sat).mean():.3f}" for g, ix in gidx.items() if ix))
    if a.fall_npz:
        fz = np.load(a.fall_npz)
        aw = list(np.asarray(z["policy_action_worlds"], dtype=int))
        col = {w: i for i, w in enumerate(aw)}
        rows = [(col[w], int(tf), int(t3)) for w, tf, t3 in zip(fz["worlds"], fz["t_f"], fz["t_03"]) if w in col and tf >= 0]
        print(f"摔了且抄了动作的世界 {len(rows)}/{len(aw)}；训练回合长度 {a.horizon:g} 步")
        if rows:
            N = act.shape[0]
            for wname, (lo_off, hi_off) in (("pre30", (-30, 0)), ("pre60", (-60, -30)), ("prewhole", (None, 0))):
                cells, inh, tot = {g: [] for g in gidx}, 0, 0
                for c, tf, _ in rows:
                    lo = 0 if lo_off is None else max(0, tf + lo_off)
                    hi = min(N, tf + hi_off)
                    if hi <= lo:
                        continue
                    for g, ix in gidx.items():
                        if ix:
                            cells[g].append(act[lo:hi, c][:, ix] > a.sat)
                    tot += hi - lo
                    inh += max(0, min(hi, int(a.horizon)) - lo) if a.horizon > 0 else 0
                print(f"{wname}\t" + "\t".join(f"{g} {np.concatenate(v).mean():.3f}" for g, v in cells.items() if v)
                      + (f"\t训练长度以内 {inh / tot:.2f}" if tot and a.horizon > 0 else ""))
            lw = gidx["腿"] + gidx["腰"]
            first = []
            for c, tf, t3 in rows:
                hit = np.flatnonzero((act[: max(tf, 1), c][:, lw] > a.sat).any(axis=1))
                first.append((hit[0] if len(hit) else -1, t3))
            fs = np.array([f for f, _ in first]); t3s = np.array([t for _, t in first])
            have = fs >= 0
            print(f"腿/腰第一次顶到头（摔前）：有 {int(have.sum())}/{len(rows)} 个世界；步中位 {int(np.median(fs[have])) if have.any() else '—'}；"
                  f"t_03 中位 {int(np.median(t3s[t3s >= 0])) if (t3s >= 0).any() else '—'}；先于 t_03 的世界比例 {np.mean((fs >= 0) & ((t3s < 0) | (fs < t3s))):.2f}")
    print("RESIDUAL_SAT_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
