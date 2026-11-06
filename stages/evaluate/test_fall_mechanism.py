"""fall_mechanism.first_over 的测试（暂无契约 ID：诊断工具，计划 agent-doc/plan-t12fall-2026-10-20.md）。要 numpy。
单帧门槛取第一次越过的帧；连续 run 帧取这一段的起始帧（复核 MINOR 5）；没有越过记 -1；不够长的段不算。
用法：python stages/evaluate/test_fall_mechanism.py   最后一行 ALL_OK
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.modules.setdefault("finger_close", type(sys)("finger_close"))   # 只测纯函数，不加载 URDF 工具与 pyarrow
for m in ("pyarrow", "pyarrow.parquet"):
    sys.modules.setdefault(m, type(sys)(m))
for k in ("URDF", "Robot", "quat_mat"):
    setattr(sys.modules["finger_close"], k, None)
import fall_mechanism as fm


def test_first_over() -> None:
    x = np.zeros((10, 4))
    x[3, 0] = 1                       # 世界 0：单帧越过于第 3 帧
    x[5:8, 1] = 1                     # 世界 1：第 5–7 帧连续 3 帧
    x[2:4, 2] = 1; x[6:9, 2] = 1      # 世界 2：第 2–3 帧只有 2 帧，第 6–8 帧 3 帧
    assert fm.first_over(x, 0.5).tolist() == [3, 5, 2, -1]
    assert fm.first_over(x, 0.5, 3).tolist() == [-1, 5, 6, -1]
    y = np.zeros((5, 1)); y[2:5, 0] = 1          # 段落在末尾
    assert fm.first_over(y, 0.5, 3).tolist() == [2]


if __name__ == "__main__":
    test_first_over()
    print("ok  test_first_over")
    print("ALL_OK")
