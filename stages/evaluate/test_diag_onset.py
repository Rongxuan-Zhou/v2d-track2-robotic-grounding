"""diag_onset.py 的合成数据测试（暂无契约 ID：诊断工具，计划 agent-doc/plan-t2dg-2026-10-20.md 第 2 节 D1b）。要 numpy；在容器里跑。
造 5 个世界：0 成功（物体跟着参考走）；1 贴住后掉（掉之前手掌贴着物体，之后物体落地）；2 没贴住（手掌一直离物体 0.5 m，物体抬起后就掉）；
3 参考无接触时离开（参考接触之前物体就被推开）；4 没离开（误差一直在 4–6 cm：MP 不成功，但从没超过 0.1 m）。核各世界的类别、离开帧与 D1b 的读法。
用法（容器内）：python /repo/stages/evaluate/test_diag_onset.py   最后一行 ALL_OK
"""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import diag_onset

T, W = 120, 5


def make() -> dict:
    t = np.arange(T, dtype=np.float64)
    ref = np.zeros((T, 1, 7)); ref[:, 0, 6] = 1.0
    ref[:, 0, 2] = np.where(t >= 40, np.minimum((t - 40) * 0.01, 0.5), 0.0)   # 第 41 帧起抬高，最多 0.5 m
    ach = np.repeat(ref[:, None], W, axis=1).astype(np.float32)               # (T,W,1,7)
    ach[60:, 1, 0, 2] = -0.8                                                  # 1：第 60 帧掉到地上
    ach[45:, 2, 0, 2] = -0.8                                                  # 2：第 45 帧掉
    ach[20:, 3, 0, 0] = 0.5                                                   # 3：第 20 帧被推开 0.5 m（参考第 30 帧才接触）
    ach[:, 4, 0, 0] = 0.05                                                    # 4：一直偏 5 cm
    contact = np.zeros((T, 2), bool); contact[30:, 1] = True                  # 只有右手，第 30 帧起接触
    verts = np.array([[[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]]])
    ref_palm = np.zeros((T, 2, 3)); ref_palm[:, 1] = ref[:, 0, :3] + [0.0, 0.0, 0.02]   # 参考右手掌贴着物体（2 cm）
    palm = np.zeros((T, W, 2, 3))
    palm[:, :, 1] = ach[:, :, 0, :3] + [0.0, 0.0, 0.02]                       # 默认机器人右手贴着实际物体
    palm[60:, 1, 1] = ref[60:, 0, :3] + [0.0, 0.0, 0.02]                      # 1：掉了以后手还在参考位置（掉之前贴着）
    palm[:, 2, 1] = ach[:, 2, 0, :3] + [0.5, 0.0, 0.0]                        # 2：手一直离物体 0.5 m
    return {"achieved_pose_w": ach, "reference_pose_w": ref, "robot_palm_pos": palm, "robot_palm_sides": np.array(["left", "right"]),
            "ref_palm_pos": ref_palm, "ref_palm_names": np.array(["left_hand_palm_link", "right_hand_palm_link"]),
            "contact_sides": np.array(["left", "right"]), "contact_active": contact, "object_vertices_o": verts,
            "policy": np.array("fake"), "upstream_md5": np.array("x")}


def test_kinds() -> None:
    r = diag_onset.analyze(make())
    assert r["mp_ok"].tolist() == [True, False, False, False, False], r["mp_ok"]
    assert r["kind"].tolist() == ["", "贴住后掉", "没贴住", "参考无接触时离开", "没离开"], r["kind"].tolist()
    assert r["t_o"].tolist()[:4] == [-1, 60, 45, 20], r["t_o"]
    assert r["t_c"] == 30 and r["t_m"] == 43 and r["active"] == ["right"], (r["t_c"], r["t_m"], r["active"])


def test_main_output() -> None:
    with tempfile.TemporaryDirectory() as t:
        f = Path(t) / "x.npz"
        np.savez(f, **make())
        sys.argv = ["diag_onset.py", str(f)]
        assert diag_onset.main() == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print(f"ok  {name}")
    print("ALL_OK")
