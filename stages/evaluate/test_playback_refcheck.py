"""playback_refcheck.py 的测试（只用标准库，Mac 上直接跑；暂无契约 ID：诊断工具不在交付链上）。

逐帧回放一份用 Day 1 真实输出 results/day1/static_baseline_ep20.txt；真实时长一份按同一段动作换算后合成
（帧数 921、源第 60j 帧的角度放到第 100j 帧、首次越线第 448 帧），再各改坏一处，看 R1–R4 分别拦下。
另查：只读 [evaluator definition] 那段（后面追加的 --tier1-episode 段不影响结果）、缺行时报错、--row 打出的运行表行。
用法：python3 stages/evaluate/test_playback_refcheck.py
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "results" / "day1" / "static_baseline_ep20.txt"


def src_angles() -> dict:
    text = SRC.read_text().split("[original 2026-09-27 method]")[0]
    pairs = re.search(r"angle vs frame 0 every 20 frames: (.*)", text).group(1).split()
    return {int(k): v for k, v in (p.split(":") for p in pairs)}


def real_time_text(frames=921, playback="False", auc=0.4712, gate=448, bump=None, drop=None, tail="") -> str:
    ang = src_angles()
    items = []
    for k in range(0, frames, 20):
        j = 0.6 * k
        near = min(ang, key=lambda x: abs(x - j))  # 不在源打印里的帧随便取个近邻，R4 不比它们
        v = float(ang[int(j)] if j.is_integer() and int(j) in ang else ang[near])
        if bump is not None and k == bump:
            v += 5
        items.append(f"{k}:{v:.0f}")
    lines = [
        "[evaluator definition] reference /repo/runs/motion/tier1/x/data.parquet",
        f"  frames {frames}, bodies 1, source_frame_playback {playback}, playback_fps 50.0, vertices 500 per body (seed 0, trimesh)",
        f"  static-object ADD-AUC {auc:.4f} (per body {auc:.4f}); mean ADD 6.79 cm; max position drift 4.0 cm",
        '  summary_json {"add_auc": %r, "maniptrans_sr": 0.0}' % auc,
        "  angle vs frame 0: mean 49.0 deg, max 179.9 deg (frame 598), final 50.6 deg",
        "  per-frame increment: median 0.058 deg, p99 2.22, max 2.57 (frame 658); > 5 deg: 0 frames",
        "  cumulative rotation 330.1 deg; frames with increment > 0.5 deg: 250 (first 352, last 698)",
        f"  a static object first exceeds the 1.5 rad (85.9 deg) gate at frame {gate}; frames above the gate: 230",
        "  angle vs frame 0 every 20 frames: " + " ".join(items),
    ]
    if drop:
        lines = [ln for ln in lines if drop not in ln]
    return "\n".join(lines) + "\n" + tail


def run(tmp: Path, text: str, row=True):
    (tmp / "rt.txt").write_text(text)
    cmd = [sys.executable, str(HERE / "playback_refcheck.py"), str(SRC), str(tmp / "rt.txt")]
    if row:
        cmd += ["--row", "t1-ep20-test-rt", "349999104", "-"]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def main() -> int:
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"  | {detail[:600]}"))
        fails += not ok

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        p = run(tmp, real_time_text())
        row = next((ln for ln in p.stdout.splitlines() if ln.startswith("diag_row ")), "")
        check("合成的真实时长参考全部通过、退出码 0", p.returncode == 0 and p.stdout.count("  PASS  ") == 4, p.stdout + p.stderr)
        check("运行表行：关注段 348–700、G4 352 / 698 / 448、G6 598、换算 1.6666667",
              row == "diag_row t1-ep20-test-rt 349999104 348 700 352 698 448 598 - 1.6666667", row)
        check("R4 可比帧 10 对（真实时长第 0、100 … 900 帧）", "10 对可比帧" in p.stdout, p.stdout)
        for label, kw, gate in (("R1 帧数不对", {"frames": 553}, "R1"), ("R1 仍是逐帧回放", {"playback": "True"}, "R1"),
                                ("R2 不动 ADD-AUC 差 0.02", {"auc": 0.4908}, "R2"), ("R3 越线帧差 12", {"gate": 460}, "R3"),
                                ("R4 第 600 帧角度差 5°", {"bump": 600}, "R4")):
            p = run(tmp, real_time_text(**kw))
            check(f"{label} → 被 {gate} 拦下、退出码 1、不打运行表行",
                  p.returncode == 1 and f"FAIL  {gate}" in p.stdout and "diag_row" not in p.stdout, p.stdout + p.stderr)
        other = "[original 2026-09-27 method] Tier 1 episode 20\n  frames 553; static-object ADD-AUC 0.4708\n" \
                "  angle vs frame 0 every 20 frames: 0:90 20:90\n"
        p = run(tmp, real_time_text(tail=other))
        check("后面追加的 --tier1-episode 段不影响结果", p.returncode == 0, p.stdout + p.stderr)
        p = run(tmp, real_time_text(drop="gate at frame"))
        check("缺“首次越线”一行时报错退出", p.returncode != 0 and "找不到" in p.stderr, p.stdout + p.stderr)
    print(f"结果：{'全部通过' if fails == 0 else f'{fails} 项失败'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
