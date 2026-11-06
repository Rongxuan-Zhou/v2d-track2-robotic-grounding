"""t3_float_facts.py 的测试（守 agent-doc/experiments/2026-10-09-t3-float-structure.md 的做法与登记判读；运行规范未成文，暂无契约 ID）。
容器内跑（要 numpy、pyarrow、trimesh、scipy）。合成场景：OpenCV 相机离地 1.39 m（y 朝下），0.4 × 0.3 × 0.5 m 的盒子
（0.3 那一边竖直，所以静置时原点离面 0.15 m），8 集放在地面上、4 集放在 0.7 m 高的桌面上，前向 1.5–5 m、横向 ±1.2 m。核：
  1. 不加误差：间隙都在 ±1 mm 内，对照判据通过（CONTROL_OK）
  2. 物体原点沿视线缩到 k = 0.92（网格不变）：地面上的间隙都是 0.08 ×（1.39 − 0.15）m，判读“常数型”，尺度模型给 1/(1 + s) ≈ 0.92、判“尺度”
  3. 整个场景绕相机 x 轴转 2°：判读“倾角型”，俯仰约 2°
  4. 端到端：parquet、元数据（tier3_pack_remote.sh 的布局）、网格、地面与支撑面写进临时目录，跑 main()，输出里有每集一行、判读与结束标记；
     再加 --all-rest --offset-gate 30 跑一次：地面上 8 集的间隙约 +99 mm，打 OFFSET_GATE_FAIL，不再标“事后（未登记）”
  6. --support-root（计划 agent-doc/plan-t3-hm-p2-2026-10-12.md）：支撑面只放在转来的目录里、官方 Tier 2 目录里没有：加 --support-root 时
     4 集判在 0.7 m 的桌面上、第一行写明支撑面的目录；不加时读官方目录、找不到文件就报错
  5. --offset-gate 的判据（agent-doc/plan-t3-pilot-v2-2026-10-10.md 第 4 节）：地面上 14 集、偏移 +20 mm 时门槛 30 过、15 不过；
     偏移 −40 mm 时 30 不过；地面上只有 7 集时不判读
用法：scripts/dev/incontainer stages/recovery/test_t3_float_facts.py
"""
import contextlib
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t3_float_facts as F  # noqa: E402

H = 1.39
PLANE = [0.0, 1.0, 0.0, -H]            # y 朝下：地面上的点 y = H
TABLE = "#usda 1.0\ndef Xform \"s\"\n{\n  def Cube \"table\"\n  {\n    double size = 0.7\n    double3 xformOp:translate = (0.5, 0.2, 0.35)\n  }\n}\n"
BOX = trimesh.creation.box(extents=(0.4, 0.3, 0.5))
V = np.asarray(BOX.convex_hull.vertices, dtype=np.float64)
SPOTS = [(x, z, 0.0) for x, z in [(-1.2, 1.5), (0.0, 2.0), (1.0, 2.5), (-0.5, 3.0), (0.6, 3.5), (-1.0, 4.0), (0.3, 4.5), (1.2, 5.0)]] \
    + [(x, z, 0.7) for x, z in [(0.4, 2.2), (-0.8, 2.8), (0.9, 3.3), (-0.2, 4.2)]]


def poses(x, z, top, T=40):
    p = np.zeros((T, 7))
    p[:, 0], p[:, 1], p[:, 2], p[:, 3] = x, H - top - 0.15, z, 1.0
    return p


def scale(p, k):
    q = p.copy()
    q[:, :3] *= k
    return q


def pitch(p, deg):
    a = np.radians(deg)
    R = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    q = p.copy()
    q[:, :3] = p[:, :3] @ R.T
    w, xq = np.cos(a / 2), np.sin(a / 2)          # 绕 x 轴的四元数（wxyz）左乘单位四元数
    q[:, 3:7] = [w, xq, 0.0, 0.0]
    return q


def rows_for(err):
    out = []
    for x, z, top in SPOTS:
        r = F.episode_facts(err(poses(x, z, top)), np.ones(40, bool), V, PLANE, F.surface_tops(TABLE))
        out.append(r)
    return out


def test_clean():
    rows = rows_for(lambda p: p)
    assert all(abs(r["gap"]) < 1e-3 for r in rows), [r["gap"] for r in rows]
    assert [r["on_floor"] for r in rows] == [True] * 8 + [False] * 4
    assert abs(rows[0]["H"] - H) < 1e-9 and abs(rows[0]["Z"] - 1.5) < 1e-6 and abs(rows[0]["X"] + 1.2) < 1e-6
    assert F.control_ok(F.fit(rows))


def test_scale():
    rows = rows_for(lambda p: scale(p, 0.92))
    fl = [r["gap"] for r in rows if r["on_floor"]]
    assert np.allclose(fl, 0.08 * (H - 0.15), atol=1e-3), fl
    out = F.fit(rows)
    main, sub = F.reading(out)
    (s, o), _, _ = out["scale"]
    assert main == "常数型" and sub == "尺度", (main, sub)
    assert abs(1 / (1 + s) - 0.92) < 2e-3 and abs(o) < 2e-3, (s, o)
    assert not F.control_ok(out)


def test_pitch():
    rows = rows_for(lambda p: pitch(p, 2.0))
    out = F.fit(rows)
    main, _ = F.reading(out)
    p_deg, r_deg = F.tilt_deg(out)
    assert main == "倾角型", main
    assert abs(abs(p_deg) - 2.0) < 0.2 and abs(r_deg) < 0.2, (p_deg, r_deg)


def test_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        t2, root = d / "tier2", d / "pack"
        (t2 / "mesh" / "box").mkdir(parents=True)
        BOX.export(t2 / "mesh" / "box" / "box.obj")
        (t2 / "ground_plane").mkdir()
        (t2 / "support_surface").mkdir()
        (root / "data" / "chunk-000").mkdir(parents=True)
        (root / "meta").mkdir()
        for ep, (x, z, top) in enumerate(SPOTS):
            seq = f"seq{ep}"
            (t2 / "ground_plane" / f"{seq}.json").write_text(json.dumps({"plane": PLANE}))
            (t2 / "support_surface" / f"{seq}_support.usda").write_text(TABLE)
            meta = {"episode_index": ep, "object": "box", "mesh": "mesh/box/box.obj", "ground_plane": f"ground_plane/{seq}.json",
                    "support_surface": f"support_surface/{seq}_support.usda"}
            (root / "meta" / f"episode_{ep:06d}.metadata.json").write_text(json.dumps(meta))
            p = scale(poses(x, z, top), 0.92)
            pq.write_table(pa.table({"observation.object.pose": p.tolist(), "observation.object.visible": [True] * len(p)}),
                           root / "data" / "chunk-000" / f"episode_{ep:06d}.parquet")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert F.main([str(root), "--tier2", str(t2)]) == 0
        text = buf.getvalue()
        assert text.count("\tbox\t") == len(SPOTS), text
        assert "判读：常数型；尺度模型：尺度" in text and text.rstrip().endswith("T3_FLOAT_FACTS_DONE"), text
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert F.main([str(root), "--tier2", str(t2), "--all-rest", "--offset-gate", "30"]) == 0
        text = buf.getvalue()
        assert "OFFSET_GATE_FAIL（地面上 8 集" in text and "每集标准差" in text and "事后（未登记）" not in text and "10/10 登记" in text, text
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert F.main([str(root), "--tier2", str(t2), "--all-rest"]) == 0
        text = buf.getvalue()
        assert "事后（未登记）" in text and "OFFSET_GATE" not in text, text


def test_support_root():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        t2, root = d / "tier2", d / "pack"
        (t2 / "mesh" / "box").mkdir(parents=True)
        BOX.export(t2 / "mesh" / "box" / "box.obj")
        (t2 / "ground_plane").mkdir()
        (root / "data" / "chunk-000").mkdir(parents=True)
        (root / "meta").mkdir()
        (root / "support_surface").mkdir()
        for ep, (x, z, top) in enumerate(SPOTS):
            seq = f"seq{ep}"
            (t2 / "ground_plane" / f"{seq}.json").write_text(json.dumps({"plane": PLANE}))
            (root / "support_surface" / f"{seq}_support.usda").write_text(TABLE)
            meta = {"episode_index": ep, "object": "box", "mesh": "mesh/box/box.obj", "ground_plane": f"ground_plane/{seq}.json",
                    "support_surface": f"support_surface/{seq}_support.usda"}
            (root / "meta" / f"episode_{ep:06d}.metadata.json").write_text(json.dumps(meta))
            p = poses(x, z, top)
            pq.write_table(pa.table({"observation.object.pose": p.tolist(), "observation.object.visible": [True] * len(p)}),
                           root / "data" / "chunk-000" / f"episode_{ep:06d}.parquet")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            assert F.main([str(root), "--tier2", str(t2), "--support-root", str(root), "--all-rest", "--offset-gate", "30"]) == 0
        text = buf.getvalue()
        assert text.splitlines()[0].startswith(f"目录 {root}；官方 Tier 2 {t2}；支撑面 {root}；"), text
        assert "在地面上 8，在支撑面上 4" in text and "OFFSET_GATE_PASS" in text, text
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                F.main([str(root), "--tier2", str(t2), "--all-rest"])
            raise AssertionError("官方目录里没有支撑面，应当报错")
        except FileNotFoundError:
            pass


def test_offset_gate():
    def rows(n, gap):
        return [{"gap": gap + 0.001 * (i % 3 - 1), "disp_mm": 50.0, "on_floor": True, "Z": 1.5 + 0.3 * i, "X": 0.1 * i - 0.6,
                 "L": 1.0 + 0.05 * i} for i in range(n)]
    out = F.fit(rows(14, 0.020), float("inf"))
    assert out["n_floor"] == 14 and F.offset_gate(out, 30) is True and F.offset_gate(out, 15) is False
    assert F.offset_gate(F.fit(rows(14, -0.040), float("inf")), 30) is False
    assert F.offset_gate(F.fit(rows(7, 0.0), float("inf")), 30) is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"{name} ok")
    print("ALL_OK")
