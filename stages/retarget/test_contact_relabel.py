"""contact_relabel.py 的测试（暂无契约 ID；计划 agent-doc/plan-t1-contact-anticipation-2026-10-16.md）。要 numpy 与 pyarrow（在 flash_chord 容器里跑）。
  1. relabel：起点提前 lead 帧且不早于 0；不超过 gap 帧的空档补上，更长的不补；只增不减；
  2. rewrite：临时目录里造一份单行的参考，改完读回，其余列逐值相同，支撑面与元数据照抄并打印，目标已存在就拒绝；帧 × 手 的表（方向反了）拒绝。
用法：python stages/retarget/test_contact_relabel.py；最后一行 ALL_OK。
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
from contact_relabel import relabel

if __name__ == "__main__":
    c = np.zeros((2, 60))
    c[0, 20:30] = 1; c[0, 40:50] = 1          # 空档 10 帧
    c[1, 5:10] = 1; c[1, 40:45] = 1           # 空档 30 帧
    out = relabel(c, lead=10, gap=15)
    assert out[0].tolist() == [0] * 10 + [1] * 40 + [0] * 10, out[0]      # 补上空档，起点 20 → 10
    assert out[1, :10].tolist() == [1] * 10 and out[1, 10:30].sum() == 0 and out[1, 30:45].tolist() == [1] * 15, out[1]
    assert ((c > 0.5) <= (out > 0.5)).all()
    assert relabel(c, 0, 0).tolist() == c.tolist()
    print("  1. 起点提前、短空档补上、长空档不补、只增不减")
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        seq = "2026-01-01_00-00-00_box_01"
        p = root / "runs/motion/tier1/whole_body/soma" / f"sequence_id={seq}" / "robot_name=g1" / "data.parquet"
        p.parent.mkdir(parents=True)
        tbl = pa.table({"fps": pa.array([30.0], type=pa.float32()),
                        "hand_sides": pa.array([["left", "right"]], type=pa.list_(pa.string())),
                        "robot_root_position": pa.array([[[0.0, 0.0, 0.8]] * 60], type=pa.list_(pa.list_(pa.float32(), 3))),
                        "hand_contact_active": pa.array([c.tolist()], type=pa.list_(pa.list_(pa.float32()))),
                        "object_root_position": pa.array([[[0.1, 0.2, 0.3]] * 60], type=pa.list_(pa.list_(pa.float32(), 3)))})
        pq.write_table(tbl, p)
        sup = root / "runs/motion/tier1/whole_body/reconstructed_stage" / f"{seq}_support.usda"
        sup.parent.mkdir(parents=True); sup.write_text("usda")
        meta = root / "runs/motion_src/tier1" / seq / "v2d_meta.json"
        meta.parent.mkdir(parents=True); meta.write_text(json.dumps({"tier": 1, "episode": 7}))
        args = [sys.executable, str(HERE / "contact_relabel.py"), "--src", "runs/motion/tier1", "--dst", "runs/motion/tier1_c10",
                "--meta-src", "runs/motion_src/tier1", "--meta-dst", "runs/motion_src/tier1_c10", "--lead", "10", "--gap", "15", seq]
        r = subprocess.run(args, cwd=root, capture_output=True, text=True, check=False)
        assert r.returncode == 0 and r.stdout.strip().splitlines()[-1] == "CONTACT_RELABEL_DONE 1", r.stdout + r.stderr
        q = root / "runs/motion/tier1_c10/whole_body/soma" / f"sequence_id={seq}" / "robot_name=g1" / "data.parquet"
        back = pq.read_table(q)
        assert back.column("fps").equals(tbl.column("fps")) and back.column("object_root_position").equals(tbl.column("object_root_position"))
        assert np.asarray(back.column("hand_contact_active").to_pylist()[0]).tolist() == out.tolist()
        assert (root / "runs/motion/tier1_c10/whole_body/reconstructed_stage" / f"{seq}_support.usda").read_text() == "usda"
        assert json.loads((root / "runs/motion_src/tier1_c10" / seq / "v2d_meta.json").read_text())["episode"] == 7
        assert "形状 (2, 60)" in r.stdout and f"支撑面：{seq}_support.usda" in r.stdout, r.stdout
        r2 = subprocess.run(args, cwd=root, capture_output=True, text=True, check=False)
        assert r2.returncode != 0 and "不覆盖" in (r2.stdout + r2.stderr), r2.stdout + r2.stderr
        # 帧 × 手 的表（方向反了）要拒绝
        bad = root / "runs/motion/tier1/whole_body/soma" / "sequence_id=x" / "robot_name=g1" / "data.parquet"
        bad.parent.mkdir(parents=True)
        pq.write_table(tbl.set_column(tbl.schema.get_field_index("hand_contact_active"), "hand_contact_active",
                                      pa.array([c.T.tolist()], type=pa.list_(pa.list_(pa.float32())))), bad)
        (root / "runs/motion_src/tier1/x").mkdir(parents=True); (root / "runs/motion_src/tier1/x/v2d_meta.json").write_text("{}")
        r3 = subprocess.run(args[:-1] + ["x"], cwd=root, capture_output=True, text=True, check=False)
        assert r3.returncode != 0 and "应为 (2, 60)" in (r3.stdout + r3.stderr), r3.stdout + r3.stderr
    print("  2. 只换一列、其余列逐值相同、支撑面与元数据照抄、目标已存在就拒绝")
    print("ALL_OK")
