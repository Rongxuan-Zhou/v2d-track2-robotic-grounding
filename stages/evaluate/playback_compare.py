"""回放速度对照的判定（只读诊断报告里的 summary_json 行，只用标准库，任何机器都能跑）。

判据见 agent-doc/plan-playback-2026-09-28.md 第 2 版“判据（事先登记）”，跑之前定下：
  L = 关注段末（逐帧回放第 420 帧 / 真实时长第 700 帧）绕参考主轴、从第 0 帧起算的落后量，世界中位（诊断 Q2b 的 lag_deg.s1）；
      每臂取最后三个 checkpoint 的平均。S = 各臂三个 checkpoint 之间最大差的最大值；E = max(15°, 2S)。
  Δ = L(逐帧回放) − L(真实时长)，只看满配（10m）这一对，正数表示真实时长落后更少：
      Δ ≥ E → 真实时长更好；Δ ≤ −E → 真实时长更差；|Δ| < E/2 且真实时长一臂健康、已走平 → 预算内没有差别；其余 → 无结论。
  健康：真实时长满配操作段 palm 误差中位（三个 checkpoint 平均）≤ 7.5 cm。
  走平：它最后一个 checkpoint 的 L 比第一个低得不超过 max(10°, S_对照)，S_对照 是满配逐帧回放一臂三个 checkpoint 的最大差
  （登记后改动 2：原写 max(10°, S)，而 S 含这一臂自己的波动，条件永远成立）。
  机理（只在“更好”时给）：起转段与快转段新增落后量的变化各占 Δ 的比例，≥ 2/3 分别记“起步要时间”“速度上限”，其余记“混合”。
  100 万（1m）这一对只作描述，方向与满配相反且 |Δ| ≥ E 时注明。满配这一对缺失时不判定。
有效性（任何一条不满足就只列数字、退出码 3）：全部报告 VALIDITY PASS；逐帧回放的报告 553 帧、关注段 209–420、阶段 209 269 314 420，
真实时长的 921 帧、348–700、348 448 523 700；所有报告的参考净转相差 ≤ 1°；每臂正好三份报告。

用法：python3 stages/evaluate/playback_compare.py --arm 10m frame <330M 报告> <340M 报告> <350M 报告> --arm 10m real <…×3>
        [--arm 1m frame <…×3> --arm 1m real <…×3>]
"""
import argparse
import json
import sys
from pathlib import Path

MIN_E, CONVERGE_MIN, PALM_HEALTH_CM = 15.0, 10.0, 7.5
IDENTITY = {"frame": (553, [209, 420], [209, 269, 314, 420]), "real": (921, [348, 700], [348, 448, 523, 700])}
PHASES = ("起转", "停顿", "快转")


def summary(path: str) -> dict:
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if s.startswith("summary_json "):
            return json.loads(s[len("summary_json "):])
    raise SystemExit(f"{path}: 没有 summary_json 行")


def mean(xs):
    return sum(xs) / len(xs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs="+", action="append", required=True, metavar="BUFFER PLAYBACK REPORT")
    a = ap.parse_args()
    arms, problems = {}, []
    for spec in a.arm:
        if len(spec) != 5 or spec[0] not in ("10m", "1m") or spec[1] not in IDENTITY:
            raise SystemExit(f"--arm 要写成 <10m|1m> <frame|real> 三份报告（按 checkpoint 先后），收到 {spec}")
        key = (spec[0], spec[1])
        if key in arms:
            raise SystemExit(f"{key} 给了两次")
        arms[key] = [(p, summary(p)) for p in spec[2:]]
    ref_net = []
    for (buf, pb), reps in arms.items():
        frames, seg, bounds = IDENTITY[pb]
        for path, s in reps:
            if not s.get("valid"):
                problems.append(f"{path}：VALIDITY 不是 PASS")
            if s.get("frames") != frames or s.get("segments") != seg or s.get("phase_bounds") != bounds:
                problems.append(f"{path}：帧数 / 关注段 / 阶段 {s.get('frames')} / {s.get('segments')} / {s.get('phase_bounds')}，"
                                f"{pb} 一臂应为 {frames} / {seg} / {bounds}")
            if "ref_net_deg" in s:
                ref_net.append(s["ref_net_deg"])
    if ref_net and max(ref_net) - min(ref_net) > 1.0:
        problems.append(f"各报告的参考净转相差 {max(ref_net) - min(ref_net):.2f}°（> 1°）")

    stats = {}
    for (buf, pb), reps in arms.items():
        ls = [s["lag_deg"]["s1"] for _, s in reps]
        ph = [mean([s["phase_lag_deg"][i] for _, s in reps]) for i in range(3)] if all(len(s.get("phase_lag_deg", [])) == 3 for _, s in reps) else None
        palm = mean([s["palm_error_median_cm"][1] for _, s in reps])
        half = [s.get("half_rotated_lag_s1_deg") for _, s in reps]
        stats[(buf, pb)] = {"L": mean(ls), "Ls": ls, "spread": max(ls) - min(ls), "phase": ph, "palm": palm,
                            "success": [s["success"] for _, s in reps], "worlds": reps[-1][1]["worlds"],
                            "half": mean(half) if all(h is not None for h in half) else None,
                            "causes": reps[-1][1].get("first_fail_causes"), "lag_s0": mean([s["lag_deg"]["s0"] for _, s in reps])}
        st = stats[(buf, pb)]
        print(f"== {buf} {'逐帧回放（1.67 倍速）' if pb == 'frame' else '真实时长'}：L 三个 checkpoint {', '.join(f'{x:.1f}' for x in ls)} → 平均 {st['L']:.1f}°，"
              f"最大差 {st['spread']:.1f}°；关注段起点落后量 {st['lag_s0']:+.1f}°")
        if ph:
            print("   新增落后量（三个 checkpoint 平均）：" + "；".join(f"{n} {v:+.1f}°" for n, v in zip(PHASES, ph)))
        print(f"   操作段 palm 误差中位 {palm:.1f} cm；成功世界 {st['success']}（共 {st['worlds']}）；"
              f"净转过半那部分的 L {('%.1f°' % st['half']) if st['half'] is not None else '—'}；最后一个 checkpoint 首个越线原因 {st['causes']}")
    if problems:
        print("有效性没过，只列数字、不判定：\n  " + "\n  ".join(problems))
        return 3

    S = max(st["spread"] for st in stats.values())
    E = max(MIN_E, 2 * S)
    print(f"S = {S:.1f}°；E = max({MIN_E:.0f}°, 2S) = {E:.1f}°")
    deltas = {}
    for buf in ("10m", "1m"):
        if (buf, "frame") in stats and (buf, "real") in stats:
            deltas[buf] = stats[(buf, "frame")]["L"] - stats[(buf, "real")]["L"]
            print(f"Δ({buf}) = {stats[(buf, 'frame')]['L']:.1f} − {stats[(buf, 'real')]['L']:.1f} = {deltas[buf]:+.1f}°")
    if "10m" not in deltas:
        print("判定：不给（满配这一对缺失，只列数字）")
        return 0
    d = deltas["10m"]
    real = stats[("10m", "real")]
    healthy = real["palm"] <= PALM_HEALTH_CM
    s_ctrl = stats[("10m", "frame")]["spread"] if ("10m", "frame") in stats else S
    converged = real["Ls"][0] - real["Ls"][-1] <= max(CONVERGE_MIN, s_ctrl)
    print(f"健康（真实时长满配操作段 palm 误差 {real['palm']:.1f} ≤ {PALM_HEALTH_CM} cm）：{'是' if healthy else '否'}；"
          f"走平（第一个 {real['Ls'][0]:.1f} − 最后 {real['Ls'][-1]:.1f} ≤ max({CONVERGE_MIN:.0f}, S_对照 {s_ctrl:.1f})）：{'是' if converged else '否'}")
    if d >= E:
        verdict = "真实时长更好"
        fr = stats[("10m", "frame")]
        if fr["phase"] and real["phase"]:
            share = [(fr["phase"][i] - real["phase"][i]) / d for i in range(3)]
            mech = "起步要时间" if share[0] >= 2 / 3 else "速度上限" if share[2] >= 2 / 3 else "混合"
            verdict += f"（机理：{mech}；Δ 中起转段占 {share[0]:.0%}、停顿段 {share[1]:.0%}、快转段 {share[2]:.0%}）"
    elif d <= -E:
        verdict = "真实时长更差"
    elif abs(d) < E / 2 and healthy and converged:
        verdict = "预算内没有差别"
    else:
        why = []
        if abs(d) >= E / 2:
            why.append(f"|Δ| 在 E/2 与 E 之间")
        if not healthy:
            why.append("真实时长一臂不健康")
        if not converged:
            why.append("真实时长一臂可能没训够")
        verdict = "无结论（" + "；".join(why) + "）"
    print(f"判定：{verdict}")
    if "1m" in deltas:
        d1 = deltas["1m"]
        note = "（与满配方向相反且 |Δ| ≥ E，注明）" if d1 * d < 0 and abs(d1) >= E else ""
        print(f"100 万一对（只作描述）：Δ(1m) {d1:+.1f}°{note}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
