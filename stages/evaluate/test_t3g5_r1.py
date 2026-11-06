"""stages/evaluate/t3g5_r1.py 的测试：守计划 agent-doc/plan-t3g5-2026-10-21.md 判据 R1 的门槛（SP-SR ≥ 0.5 且 MPPE ≤ 10 cm）。"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t3g5_r1 import verdict


def test_verdict_edges():
    """R1：两个门槛都含等号；任一不满足就是仍跟丢。"""
    assert verdict(0.5, 10.0) == "跟上"
    assert verdict(0.49, 5.0) == "仍跟丢"
    assert verdict(0.9, 10.01) == "仍跟丢"


def test_cli_missing_and_present():
    """R1：缺文件记“没有评测结果”，有 summary_json 的按门槛判，末行计数。"""
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "eval_x.txt"
        f.write_text("x\n  summary_json " + json.dumps({"checkpoint": "/repo/runs/flash_chord/t3-ep5-w2048-cap4m-r3a-gb5-mf4090/policy_1.safetensors",
                                                        "spider_sr_uncentered": 0.7, "mppe_cm": 4.2, "maniptrans_sr": 0.6, "add_auc": 0.5}) + "\n")
        out = subprocess.run([sys.executable, str(HERE / "t3g5_r1.py"), str(f), str(Path(d) / "none.txt")], capture_output=True, text=True, check=False).stdout
        assert "t3-ep5-w2048-cap4m-r3a-gb5-mf4090：SP-SR 0.700、MPPE 4.20 cm" in out and "→ 跟上" in out, out
        assert "没有评测结果" in out and out.strip().endswith("T3G5_R1_DONE 跟上 1 / 共 2"), out
