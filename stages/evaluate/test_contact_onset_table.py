"""contact_onset_table.py 的测试（暂无契约 ID；计划 agent-doc/plan-c10-extend-2026-10-16.md 第 3.1 节）。要 numpy 与 pyarrow（在 flash_chord 容器里跑）。
  1. lead_rule：delay ≥ 0 的最小值向上取到 5 的倍数、限在 [5, 20]；没有 delay ≥ 0 的给 None；
  2. 临时目录里造 4 集 Tier 2 参考（2 集跟丢），核对分段、空档、delay、三条特征、lead 与选集的输出；
  3. Tier 1 的路径、带后缀的 Tier 2 目录、元数据档位不符都拒收；帧 × 手 的表（方向反了）拒收。
用法：python stages/evaluate/test_contact_onset_table.py；最后一行 ALL_OK。
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from contact_onset_table import lead_rule  # noqa: E402

T = 80


def write_ref(root: Path, tdir: str, seq: str, ep: int, contact: np.ndarray, spin_deg_per_frame: float, meta_tier: int) -> str:
    p = root / "runs/motion" / tdir / "whole_body/soma" / f"sequence_id={seq}" / "robot_name=g1" / "data.parquet"
    p.parent.mkdir(parents=True)
    ang = np.radians(spin_deg_per_frame) * np.maximum(0, np.arange(T) - 20)   # 第 20 帧起绕 z 轴转
    quats = [[[float(np.cos(a / 2)), 0.0, 0.0, float(np.sin(a / 2))]] for a in ang]
    tbl = pa.table({"hand_sides": pa.array([["left", "right"]], type=pa.list_(pa.string())),
                    "hand_contact_active": pa.array([contact.tolist()], type=pa.list_(pa.list_(pa.float32()))),
                    "object_root_position": pa.array([[[0.0, 0.0, 0.5]] * T], type=pa.list_(pa.list_(pa.float32(), 3))),
                    "object_body_wxyz": pa.array([quats], type=pa.list_(pa.list_(pa.list_(pa.float32(), 4))))})
    pq.write_table(tbl, p)
    m = root / "runs/motion_src" / tdir / seq / "v2d_meta.json"
    m.parent.mkdir(parents=True)
    m.write_text(json.dumps({"tier": meta_tier, "episode": ep}))
    return str(p)


def run(args, cwd):
    return subprocess.run([sys.executable, str(HERE / "contact_onset_table.py"), *args], cwd=cwd, capture_output=True, text=True, check=False)


if __name__ == "__main__":
    assert lead_rule([9, 18, 25]) == 10 and lead_rule([3]) == 5 and lead_rule([0]) == 5 and lead_rule([33]) == 20
    assert lead_rule([-5, 12]) == 15 and lead_rule([-1]) is None and lead_rule([]) is None
    print("  1. lead 规则")
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        c_frag = np.zeros((2, T)); c_frag[1, 20:25] = 1; c_frag[1, 30:35] = 1; c_frag[1, 50:60] = 1   # 右手 3 段，空档 5、15
        c_one = np.zeros((2, T)); c_one[0, 20:70] = 1                                                # 左手 1 段
        ps = [write_ref(root, "tier2", "s_a", 1, c_frag, 2.0, 2), write_ref(root, "tier2", "s_b", 2, c_frag, 1.0, 2),
              write_ref(root, "tier2", "s_c", 3, c_one, 0.0, 2), write_ref(root, "tier2", "s_d", 4, c_one, 0.2, 2)]
        r = run(["--tier", "2", "--lost", "1,2", "--onset", "1:29,2:61", "--mppe", "1:10,2:50", *ps], root)
        out = r.stdout
        assert r.returncode == 0 and out.strip().splitlines()[-1] == "CONTACT_ONSET_DONE", out + r.stderr
        assert "已装 Tier 1 原始数据禁读" in out, out
        assert "ep1（跟丢）：left 没有接触；right 首次接触第 20 帧、接触 20 帧、3 段、空档 5、15" in out, out
        assert "delay +9" in out and "delay +41" in out, out
        assert "0 ≤ delay ≤ 40 的 1 集（ep1）" in out and "→ 不过" in out.split("特征 1")[1].splitlines()[0], out
        f2 = out.split("特征 2：")[1].splitlines()[0]
        assert "跟丢 43.5°" in f2 and "其余 2.9°" in f2 and f2.endswith("→ 过"), f2   # 30 帧窗口：2°×29、1°×29 → 中位 43.5；0、0.2×29 → 中位 2.9
        assert "接触段数中位 3.0" in out and out.split("特征 3")[1].splitlines()[0].endswith("→ 过"), out
        assert "CONTACT_ONSET_FEATURES 过 2/3" in out and "CONTACT_ONSET_LEAD 10" in out, out
        assert "CONTACT_ONSET_PICK 1（按 MPPE：ep1 10.0）" in out, out
        print("  2. 分段、空档、delay、三条特征、lead 与选集")
        p1 = write_ref(root, "tier1", "s_e", 5, c_one, 0.0, 1)
        r1 = run(["--tier", "2", "--lost", "1", *ps, p1], root)
        assert r1.returncode != 0 and "只收 runs/motion/tier2/" in (r1.stdout + r1.stderr), r1.stdout + r1.stderr
        p2 = write_ref(root, "tier2_c10", "s_f", 6, c_one, 0.0, 2)
        r2 = run(["--tier", "2", "--lost", "1", *ps, p2], root)
        assert r2.returncode != 0 and "只收 runs/motion/tier2/" in (r2.stdout + r2.stderr), r2.stdout + r2.stderr
        p3 = write_ref(root, "tier2", "s_g", 7, c_one, 0.0, 3)
        r3 = run(["--tier", "2", "--lost", "1", *ps, p3], root)
        assert r3.returncode != 0 and "的档位是 3" in (r3.stdout + r3.stderr), r3.stdout + r3.stderr
        p4 = write_ref(root, "tier2", "s_h", 8, c_one.T.copy(), 0.0, 2)
        r4 = run(["--tier", "2", "--lost", "1", *ps, p4], root)
        assert r4.returncode != 0 and "应为 手 × 帧" in (r4.stdout + r4.stderr), r4.stdout + r4.stderr
    print("  3. 别的档、带后缀的目录、元数据档位不符、方向反了的表都拒收")
    print("ALL_OK")
