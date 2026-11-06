"""playback_compare.py 的合成测试（只用标准库，Mac 上直接跑；暂无契约 ID：诊断工具不在交付链上）。

每臂三份只含 summary_json 的假报告。查：四档判定各一例（更好且机理为起步、更差、预算内没有差别、|Δ| 在 E/2 与 E 之间）、
不健康与没走平时降为无结论、E 随波动变大、有效性（VALIDITY、帧数与关注段、参考净转）不过时退出码 3、
只有 100 万一对时不判定、100 万方向相反时注明。
用法：python3 stages/evaluate/test_playback_compare.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ID = {"frame": (553, [209, 420], [209, 269, 314, 420]), "real": (921, [348, 700], [348, 448, 523, 700])}


def report(tmp: Path, name: str, pb: str, lag, phase=(75.0, 5.0, 10.0), palm=5.0, valid=True, frames=None, ref=307.1) -> str:
    f, seg, bounds = ID[pb]
    s = {"worlds": 4096, "valid": valid, "add_auc": 0.43, "static_add_auc": 0.47, "segment_gap": [-0.02, 0.03, -0.04],
         "palm_error_median_cm": [4.7, palm, 9.9], "success": 3000, "frames": frames or f, "segments": seg, "ref_net_deg": ref,
         "lag_deg": {"s0": -20.0, "s1": lag, "last": lag + 5}, "phase_bounds": bounds, "phase_lag_deg": list(phase),
         "window_cross_share": 0.19, "half_rotated": 3300, "half_rotated_lag_s1_deg": lag - 5, "first_fail_causes": {"object": 700}}
    p = tmp / f"{name}.txt"
    p.write_text(f"VALIDITY {'PASS' if valid else 'FAIL'}\nsummary_json {json.dumps(s)}\n")
    return str(p)


COUNTER = [0]


def arm(tmp, buf, pb, lags, **kw):
    COUNTER[0] += 1   # 每次调用写新文件，用例之间不互相覆盖
    return ["--arm", buf, pb] + [report(tmp, f"{COUNTER[0]}-{buf}-{pb}-{i}", pb, x, **kw) for i, x in enumerate(lags)]


def run(args):
    p = subprocess.run([sys.executable, str(HERE / "playback_compare.py")] + args, capture_output=True, text=True, check=False)
    return p.returncode, p.stdout + p.stderr


def main() -> int:
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"  | {detail[-700:]}"))
        fails += not ok

    with tempfile.TemporaryDirectory() as td:
        t = Path(td)
        ctrl = arm(t, "10m", "frame", [70, 72, 71], phase=(75, 5, 10))
        one_m = arm(t, "1m", "frame", [75, 76, 74]) + arm(t, "1m", "real", [60, 61, 62])

        rc, out = run(ctrl + arm(t, "10m", "real", [40, 41, 42], phase=(45, 5, 10)) + one_m)
        check("更好：Δ = 71 − 41 = 30 ≥ E = 15，机理为起步（起转段占 100%）", rc == 0 and "判定：真实时长更好（机理：起步要时间" in out
              and "E = max(15°, 2S) = 15.0°" in out and "Δ(10m) = 71.0 − 41.0 = +30.0°" in out, out)
        rc, out = run(ctrl + arm(t, "10m", "real", [40, 41, 42], phase=(75, 5, -20)))
        check("更好但落后量的减少出在快转段 → 机理记速度上限", rc == 0 and "机理：速度上限" in out, out)
        rc, out = run(ctrl + arm(t, "10m", "real", [89, 90, 91]))
        check("更差：Δ = −19 ≤ −15", rc == 0 and "判定：真实时长更差" in out, out)
        rc, out = run(ctrl + arm(t, "10m", "real", [68, 68, 68]))
        check("预算内没有差别：|Δ| = 3 < 7.5，健康、走平", rc == 0 and "判定：预算内没有差别" in out, out)
        rc, out = run(ctrl + arm(t, "10m", "real", [61, 61, 61]))
        check("无结论：|Δ| = 10 在 E/2 与 E 之间", rc == 0 and "判定：无结论（|Δ| 在 E/2 与 E 之间" in out, out)
        rc, out = run(ctrl + arm(t, "10m", "real", [68, 68, 68], palm=9.0))
        check("无结论：真实时长操作段 palm 误差 9 cm > 7.5", rc == 0 and "判定：无结论（真实时长一臂不健康" in out, out)
        rc, out = run(ctrl + arm(t, "10m", "real", [85, 70, 55]))
        check("无结论：真实时长末段还在降（85 → 55，降 30 > max(10, S_对照 2)），|Δ| = 1 < E/2 = 30",
              rc == 0 and "判定：无结论（真实时长一臂可能没训够" in out and "E = max(15°, 2S) = 60.0°" in out, out)
        rc, out = run(arm(t, "10m", "frame", [60, 72, 66]) + arm(t, "10m", "real", [40, 41, 42]))
        check("E 随波动变大：对照三个 checkpoint 差 12 → E = 24，Δ = 25 仍判更好", rc == 0 and "E = max(15°, 2S) = 24.0°" in out
              and "判定：真实时长更好" in out, out)
        rc, out = run(ctrl + arm(t, "10m", "real", [40, 41, 42]) + arm(t, "1m", "frame", [60, 61, 62]) + arm(t, "1m", "real", [80, 81, 82]))
        check("100 万方向相反且 |Δ| = 20 ≥ E：注明", rc == 0 and "与满配方向相反" in out, out)
        rc, out = run(ctrl + arm(t, "10m", "real", [40, 41, 42], valid=False))
        check("有报告 VALIDITY FAIL → 退出码 3、不判定", rc == 3 and "VALIDITY 不是 PASS" in out
              and not any(ln.startswith("判定：") for ln in out.splitlines()), out)
        rc, out = run(ctrl + arm(t, "10m", "real", [40, 41, 42], frames=553))
        check("真实时长一臂的报告帧数是 553 → 退出码 3", rc == 3 and "921" in out, out)
        rc, out = run(ctrl + arm(t, "10m", "real", [40, 41, 42], ref=305.0))
        check("参考净转相差 2.1° > 1° → 退出码 3", rc == 3 and "参考净转相差" in out, out)
        rc, out = run(one_m)
        check("只有 100 万一对：不判定", rc == 0 and "判定：不给" in out, out)
        rc, out = run(["--arm", "10m", "frame", "a", "b"])
        check("每臂不是三份报告 → 报错", rc != 0 and "三份报告" in out, out)
    print(f"结果：{'全部通过' if fails == 0 else f'{fails} 项失败'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
