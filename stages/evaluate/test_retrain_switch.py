"""retrain_switch.py 的测试（暂无契约 ID；计划 agent-doc/plan-t3-kfix-retrain-2026-10-16.md 判据）。只用标准库。
临时目录里造新 run 的 eval_<步数>_w4096.json：
  1. ep0 新 0.92 / 0.95 过线（≥ 0.9151 / 0.9422）→ 换；ep18 新 0.80 → 不换；ep29 旧 MP-SR 为 0、新 SP-SR 0.70 ≥ 0.6631 → 换；
  2. 参考不在 tier3_kfix 下、世界数不是 4096、缺 JSON → 都不判；
  3. 真实规格文件 retrain_switch_t3pk.json、retrain_switch_t3ps.json 能读，门槛与计划登记的一致。
用法：python3 stages/evaluate/test_retrain_switch.py；最后一行 ALL_OK。
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = HERE / "retrain_switch.py"
SPEC = json.loads((HERE / "retrain_switch_t3pk.json").read_text())


def wj(root: Path, run: str, mp: float, sp: float, parquet: str, wc: int = 4096) -> None:
    d = root / "runs/flash_chord" / run
    d.mkdir(parents=True, exist_ok=True)
    (d / "eval_349999104_w4096.json").write_text(json.dumps({"metrics": {"maniptrans_sr": mp, "spider_sr_uncentered": sp, "add_auc": 0.5,
                                                                         "mppe_cm": 3.0, "world_count": wc}, "parquet": parquet}))


def run(root: Path) -> list:
    (root / "spec.json").write_text(json.dumps(SPEC))
    r = subprocess.run([sys.executable, str(TOOL), "spec.json"], cwd=root, capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout.strip().splitlines()


if __name__ == "__main__":
    pq = "/repo/runs/motion/tier3_kfix/seq/robot_name=g1/data.parquet"
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        new = {p["ep"]: p["new"] for p in SPEC["pairs"]}
        wj(root, new[0], 0.92, 0.95, pq)
        wj(root, new[18], 0.80, 0.93, pq)
        wj(root, new[29], 0.0, 0.70, pq)
        out = run(root)
        assert out[-1] == "RETRAIN_SWITCH_DONE 2 0", out
        assert out[0].endswith("：换") and out[1].endswith("：不换") and out[2].endswith("：换"), out
        assert "MP-SR ≥ 0.9151 且 SP-SR ≥ 0.9422" in out[0] and "只看 SP-SR ≥ 0.6631" in out[2], out
    print("  1. ep0 换、ep18 不换、ep29（旧 MP-SR 为 0）换")
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        wj(root, new[0], 0.99, 0.99, "/repo/runs/motion/tier3_v2/seq/robot_name=g1/data.parquet")
        wj(root, new[18], 0.99, 0.99, pq, wc=1)
        out = run(root)
        assert out[-1] == "RETRAIN_SWITCH_DONE 0 3", out
        assert "参考不在 tier3_kfix 下" in out[0] and "世界数不是 4096" in out[1] and "不判" in out[2], out
    print("  2. 参考不对、世界数不对、缺 JSON：都不判")
    ps = json.loads((HERE / "retrain_switch_t3ps.json").read_text())
    thr = {p["ep"]: (round(p["old_mp"] - 0.05, 4), round(p["old_sp"] - 0.05, 4)) for p in SPEC["pairs"] + ps["pairs"]}
    assert thr == {0: (0.9151, 0.9422), 18: (0.8121, 0.8875), 29: (-0.05, 0.6631), 20: (-0.05, 0.8448)}, thr
    print("  3. 规格文件的门槛：ep0 0.9151/0.9422、ep18 0.8121/0.8875、ep29 SP 0.6631、ep20 SP 0.8448")
    print("ALL_OK")
