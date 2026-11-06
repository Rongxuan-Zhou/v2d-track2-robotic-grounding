"""Tier 2 交接的默认值不许落到 Tier 1 派生的版本上（守档位隔离，计划 agent-doc/plan-tier2-denoise-2026-10-04.md 第 2 版第 8 节第 2 步；契约表未落，暂无契约 ID）。
只用标准库，Mac 与训练机都能跑：
  1. export_tier2.py 不给版本、或给计划外的版本：在 import 任何数据相关模块之前就退出（退出码非零，打用法），不写任何文件；
  2. stages/retarget/one.sh 的 T2_POSES 默认是 tier2_v54，脚本里不再有 tier2_v53 的默认。
用法：python3 stages/recovery/test_tier2_defaults.py
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_export_requires_version():
    for args in ([], ["v5.5"], ["v5.4", "extra"]):
        r = subprocess.run([sys.executable, str(ROOT / "stages/recovery/export_tier2.py"), *args], capture_output=True, text=True)
        assert r.returncode != 0 and "必须写明版本" in r.stderr, (args, r.returncode, r.stderr[-300:])
        assert "Traceback" not in r.stderr, r.stderr[-300:]


def test_retarget_default_is_v54():
    s = (ROOT / "stages/retarget/one.sh").read_text()
    assert "${T2_POSES:-tier2_v54}" in s and "${T2_POSES:-tier2_v53}" not in s


if __name__ == "__main__":
    test_export_requires_version(); print("ok test_export_requires_version")
    test_retarget_default_is_v54(); print("ok test_retarget_default_is_v54")
    print("ALL_OK")
