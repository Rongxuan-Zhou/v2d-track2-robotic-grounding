"""gb3_switch.py 的测试（暂无契约 ID；计划 agent-doc/plan-t3-gb3-switch-2026-10-17.md 判据 G2）。只用标准库，Mac 上跑。
在临时目录里造 runs/flash_chord/<run>/eval_<步数>_w4096.json 与规格，核：换（MPPE 在 margin 内、SP-SR 够）、不换（MPPE 超 margin；SP-SR 不够）、
不判（参考目录不对；世界数不对；缺文件）、只报的集不计入“换”。最后一行 ALL_OK。
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def ev(root: Path, run: str, mppe: float, sp: float, ref: str = "tier3_gb3", worlds: int = 4096) -> None:
    d = root / "runs/flash_chord" / run
    d.mkdir(parents=True)
    (d / "eval_349999104_w4096.json").write_text(json.dumps({
        "parquet": f"/repo/runs/motion/{ref}/whole_body/soma/sequence_id=x/robot_name=g1/data.parquet",
        "metrics": {"world_count": worlds, "mppe_cm": mppe, "spider_sr_uncentered": sp, "maniptrans_sr": 0.5, "add_auc": 0.6}}))


def pair(ep: int, run: str, cur_mppe: float, report_only: bool = False) -> dict:
    return {"ep": ep, "new": run, "ref_dir": "tier3_gb3", "cur": f"old{ep}", "cur_mppe": cur_mppe, "cur_sp": 0.99, "cur_mp": 0.9,
            "report_only": report_only}


def main() -> int:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        ev(root, "a", 6.5, 0.9)            # 2.6 + 4.0 = 6.6 → 换
        ev(root, "b", 6.7, 0.9)            # 超 margin → 不换
        ev(root, "c", 3.0, 0.45)           # SP-SR 不够 → 不换
        ev(root, "d", 3.0, 0.9, ref="tier3_hm")   # 参考目录不对 → 不判
        ev(root, "e", 3.0, 0.9, worlds=1)  # 世界数不对 → 不判
        ev(root, "g", 3.0, 0.9)            # 只报 → 不计入
        spec = {"step": 349999104, "margin_cm": 4.0, "sp_floor": 0.5, "source": "测试",
                "pairs": [pair(0, "a", 2.6), pair(1, "b", 2.6), pair(2, "c", 2.6), pair(3, "d", 2.6), pair(4, "e", 2.6), pair(5, "f", 2.6),
                          pair(6, "g", 2.6, report_only=True)]}
        (root / "spec.json").write_text(json.dumps(spec))
        r = subprocess.run([sys.executable, str(HERE / "gb3_switch.py"), "spec.json"], cwd=root, capture_output=True, text=True)
        out = r.stdout.splitlines()
        assert r.returncode == 0 and out[-1] == "GB3_SWITCH_DONE 1 3", r.stdout + r.stderr
        by = {ln.split()[0]: ln for ln in out[1:-1]}
        assert by["ep0"].endswith("：换"), by["ep0"]
        assert by["ep1"].endswith("：不换") and by["ep2"].endswith("：不换"), (by["ep1"], by["ep2"])
        assert "参考不在 tier3_gb3 下" in by["ep3"] and by["ep3"].endswith("不判"), by["ep3"]
        assert "世界数不是 4096" in by["ep4"] and "没有 runs/flash_chord/f/" in by["ep5"], (by["ep4"], by["ep5"])
        assert "［只报］" in by["ep6"] and by["ep6"].endswith("：换"), by["ep6"]
    with tempfile.TemporaryDirectory() as t:   # sp_drop_max（计划 agent-doc/plan-t3g4-2026-10-19.md 第 2 版 G2）：现选 SP-SR 0.99，下限 0.94
        root = Path(t)
        ev(root, "h", 3.0, 0.95)           # 0.95 ≥ 0.94 → 换
        ev(root, "i", 3.0, 0.93)           # 0.93 < 0.94（虽然 ≥ 0.5）→ 不换
        spec = {"step": 349999104, "margin_cm": 1.0, "sp_floor": 0.5, "sp_drop_max": 0.05, "source": "测试",
                "pairs": [pair(0, "h", 2.6), pair(1, "i", 2.6)]}
        (root / "spec.json").write_text(json.dumps(spec))
        r = subprocess.run([sys.executable, str(HERE / "gb3_switch.py"), "spec.json"], cwd=root, capture_output=True, text=True)
        out = r.stdout.splitlines()
        assert r.returncode == 0 and out[-1] == "GB3_SWITCH_DONE 1 0" and "不低于现选减 0.05" in out[0], r.stdout + r.stderr
        by = {ln.split()[0]: ln for ln in out[1:-1]}
        assert by["ep0"].endswith("：换") and by["ep1"].endswith("：不换") and "SP-SR ≥ 0.9400" in by["ep1"], (by["ep0"], by["ep1"])
    print("  换、不换（MPPE 超 margin、SP-SR 不够）、不判（参考目录、世界数、缺文件）、只报不计入、sp_drop_max：都对")
    print("ALL_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
