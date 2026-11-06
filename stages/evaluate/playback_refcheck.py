"""回放速度对照的参考核对（只用标准库）：比较同一条序列逐帧回放与按真实时长回放的两份 static_baseline.py 输出，
确认后者是同一段动作按时间重采样，并给出真实时长参考的 G4 已知答案与按时间换算的诊断帧。

判据（agent-doc/plan-playback-2026-09-28.md 执行链第 2 步，跑之前定下；s = 控制频率 / 源帧率 = 50 / 30）：
  R1 帧数 = round((源帧数 − 1) × s) + 1（553 → 921）；逐帧回放一份 source_frame_playback 为 True，另一份为 False
  R2 物体不动的 ADD-AUC 相差 ≤ 0.010
  R3 首次越过 1.5 rad 的帧与“源帧 × s”相差 ≤ 2
  R4 两份都打印了的“离初始朝向的角度”（每 20 帧一个，取整到 1°）里，真实时长第 k 帧与源第 k / s 帧相差 ≤ 2°
只读 [evaluator definition] 那一段（--tier1-episode 另算的那段不读）。全过：打印 G4 已知答案与换算后的帧，
给 --row 时另打一行 `diag_row …`（stages/evaluate/diag_day1b.sh 运行表的格式），退出码 0；任何一条不过退出码 1。

用法：python3 stages/evaluate/playback_refcheck.py <逐帧回放的输出> <真实时长的输出> [--segments 209 420] [--g6 359]
        [--row <run> <步数> <参照评测 sha256 前 16 位或 ->]
"""
import argparse
import json
import re
import sys
from pathlib import Path

SOURCE_FPS, CONTROL_FPS = 30.0, 50.0


def parse(path: str) -> dict:
    text = Path(path).read_text().split("[original 2026-09-27 method]")[0]

    def one(pattern: str) -> re.Match:
        m = re.search(pattern, text)
        if not m:
            raise SystemExit(f"{path}: 找不到 /{pattern}/")
        return m

    m = one(r"frames (\d+), bodies (\d+), source_frame_playback (True|False), playback_fps ([0-9.]+)")
    out = {"frames": int(m.group(1)), "frame_playback": m.group(3) == "True", "playback_fps": float(m.group(4))}
    out["add_auc"] = json.loads(one(r"summary_json (\{.*\})").group(1))["add_auc"]
    m = one(r"frames with increment > 0\.5 deg: (\d+) \(first (\d+), last (\d+)\)")
    out["first_move"], out["last_move"] = int(m.group(2)), int(m.group(3))
    out["first_gate"] = int(one(r"gate at frame (\d+)").group(1))
    pairs = one(r"angle vs frame 0 every 20 frames: (.*)").group(1).split()
    out["angle"] = {int(k): float(v) for k, v in (p.split(":") for p in pairs)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("frame_playback_txt")
    ap.add_argument("real_time_txt")
    ap.add_argument("--segments", type=int, nargs=2, default=(209, 420))
    ap.add_argument("--g6", type=int, default=359)
    ap.add_argument("--row", nargs=3, metavar=("RUN", "STEP", "REFSHA"))
    a = ap.parse_args()
    src, rt = parse(a.frame_playback_txt), parse(a.real_time_txt)
    s = CONTROL_FPS / SOURCE_FPS
    checks = []

    want = int(round((src["frames"] - 1) * s)) + 1
    checks.append(("R1 帧数与回放方式", rt["frames"] == want and src["frame_playback"] and not rt["frame_playback"],
                   f"真实时长 {rt['frames']} 帧（应为 {want}）；source_frame_playback 逐帧 {src['frame_playback']} / 真实时长 {rt['frame_playback']}"))
    d = rt["add_auc"] - src["add_auc"]
    checks.append(("R2 不动 ADD-AUC", abs(d) <= 0.010, f"{src['add_auc']:.4f} → {rt['add_auc']:.4f}（差 {d:+.4f}，容差 0.010）"))
    g = src["first_gate"] * s
    checks.append(("R3 首次越 1.5 rad", abs(rt["first_gate"] - g) <= 2, f"源第 {src['first_gate']} 帧 × {s:.4f} = {g:.1f}；真实时长第 {rt['first_gate']} 帧"))
    common = [(k, round(k / s)) for k in sorted(rt["angle"]) if abs(k / s - round(k / s)) < 1e-9 and round(k / s) in src["angle"]]
    worst = max(((abs(rt["angle"][k] - src["angle"][j]), k, j) for k, j in common), default=(float("inf"), -1, -1))
    checks.append(("R4 离初始朝向的角度", len(common) >= 5 and worst[0] <= 2.0,
                   f"{len(common)} 对可比帧，最大差 {worst[0]:.0f}°（真实时长第 {worst[1]} 帧 vs 源第 {worst[2]} 帧）"))

    print(f"参考核对（逐帧回放 {a.frame_playback_txt} vs 真实时长 {a.real_time_txt}；换算倍数 {s:.6f}）")
    for name, ok, detail in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}：{detail}")
    if not all(ok for _, ok, _ in checks):
        print("有核对没过：真实时长参考不能当作同一段动作按时间重采样，停下查原因")
        return 1
    seg = [int(round(f * s)) for f in a.segments]
    g6 = int(round(a.g6 * s))
    print(f"G4 已知答案（真实时长参考）：增量 > 0.5° 首帧 {rt['first_move']}、末帧 {rt['last_move']}；首次越 1.5 rad 第 {rt['first_gate']} 帧")
    print(f"按时间换算：关注段 {a.segments[0]}–{a.segments[1]} → {seg[0]}–{seg[1]}；G6 帧 {a.g6} → {g6}")
    if a.row:
        run, step, refsha = a.row
        print(f"diag_row {run} {step} {seg[0]} {seg[1]} {rt['first_move']} {rt['last_move']} {rt['first_gate']} {g6} {refsha} {s:.7f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
