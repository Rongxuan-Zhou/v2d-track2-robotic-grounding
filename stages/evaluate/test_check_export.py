"""check_export.py 的行为测试（暂无契约 ID：运行规范待成文）。要 numpy、pyarrow：
  Mac 上 uv run --no-project --with numpy --with pyarrow python stages/evaluate/test_check_export.py；训练机宿主机的 python3 也能直接跑。

临时目录里搭最小仓库：真的 check_export.py 与 third_party/patches/flash_chord-8c778f9.eval.json；轨迹 parquet 按上游 8c778f9
rollout.py 的 write_object_trajectories_parquet 复刻（同样的列、zstd、字典列、按 32 步分块、flash_chord.* 元数据，见
third_party/patches/flash_chord-v0.4.0-to-8c778f9.patch），motion_v1 参考是一行多个 list 列。
  1. 1 个世界、2 个世界、--no-traj：EXPORT_OK；
  2. 2 个刚体：行数与每步行数按刚体数算，EXPORT_OK；
  3. 有世界被标非有限、位姿里有 NaN、JSON 的 step_count 不对、给的世界数不对、eval_code_sha256 不对、checkpoint 被换：都 EXPORT_FAIL。
用法见上。审查（Fable 5.1，10/1）留下的原型改写而来。
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
RUN, STEP, SEQ, EP, T = "t1-ep6-w2048-cap4m-mf4090", 349999104, "2026-06-01_11-26-44_monitor_rotate_05", 6, 546
POSE = ("position_x", "position_y", "position_z", "quaternion_x", "quaternion_y", "quaternion_z", "quaternion_w")


def schema(steps, worlds, bodies, provenance):
    md = {b"flash_chord.schema": b"flash_chord_object_trajectories_v1", b"flash_chord.position_unit": b"meter",
          b"flash_chord.quaternion_order": b"xyzw", b"flash_chord.row_order": b"step,environment_id,object_body_id",
          b"flash_chord.control_fps": b"50.0", b"flash_chord.start_frame": b"0", b"flash_chord.step_count": str(steps).encode(),
          b"flash_chord.world_count": str(worlds).encode(), b"flash_chord.object_body_count": str(bodies).encode(),
          b"flash_chord.pose_width": b"7"}
    for k, v in provenance.items():
        md[f"flash_chord.{k}".encode()] = str(v).encode()
    fields = [pa.field("environment_id", pa.int32(), nullable=False), pa.field("step", pa.int32(), nullable=False),
              pa.field("reference_frame", pa.int32(), nullable=False), pa.field("time_s", pa.float64(), nullable=False),
              pa.field("object_id", pa.int32(), nullable=False), pa.field("object_body_id", pa.int32(), nullable=False),
              pa.field("object_body_name", pa.dictionary(pa.int32(), pa.string()), nullable=False),
              pa.field("environment_valid", pa.bool_(), nullable=False)]
    fields += [pa.field(f"{p}_{n}", pa.float32(), nullable=False) for p in ("achieved", "reference") for n in POSE]
    return pa.schema(fields, metadata=md)


def write_traj(path, achieved, reference, names, non_finite, provenance, step_chunk=32):
    steps, worlds, bodies, _ = achieved.shape
    sch = schema(steps, worlds, bodies, provenance)
    valid = np.ones(worlds, dtype=np.bool_)
    if non_finite:
        valid[np.asarray(non_finite)] = False
    with pq.ParquetWriter(path, sch, compression="zstd", use_dictionary=True) as w:
        for b in range(0, steps, step_chunk):
            e = min(b + step_chunk, steps); cs = e - b
            rs = np.repeat(np.arange(b, e, dtype=np.int32), worlds * bodies)
            rw = np.tile(np.repeat(np.arange(worlds, dtype=np.int32), bodies), cs)
            rb = np.tile(np.arange(bodies, dtype=np.int32), cs * worlds)
            nm = pa.DictionaryArray.from_arrays(pa.array(rb, type=pa.int32()), pa.array(list(names)))
            ach = np.asarray(achieved[b:e], dtype=np.float32).reshape(-1, 7)
            ref = np.broadcast_to(np.asarray(reference[b:e], dtype=np.float32)[:, None], (cs, worlds, bodies, 7)).reshape(-1, 7)
            cols = {"environment_id": pa.array(rw), "step": pa.array(rs), "reference_frame": pa.array(rs),
                    "time_s": pa.array(rs.astype(np.float64) / 50.0), "object_id": pa.array(np.zeros_like(rb)),
                    "object_body_id": pa.array(rb), "object_body_name": nm, "environment_valid": pa.array(valid[rw])}
            for i, n in enumerate(POSE):
                cols[f"achieved_{n}"] = pa.array(ach[:, i]); cols[f"reference_{n}"] = pa.array(ref[:, i])
            w.write_table(pa.Table.from_pydict(cols, schema=sch))


def setup(root: Path, worlds=1, bodies=1, non_finite=(), tamper=None) -> None:
    (root / "stages/evaluate").mkdir(parents=True)
    shutil.copy(REPO / "stages/evaluate/check_export.py", root / "stages/evaluate")
    (root / "third_party/patches").mkdir(parents=True)
    shutil.copy(REPO / "third_party/patches/flash_chord-8c778f9.eval.json", root / "third_party/patches")
    dig = json.loads((REPO / "third_party/patches/flash_chord-8c778f9.eval.json").read_text())
    rd = root / "runs/flash_chord" / RUN
    rd.mkdir(parents=True)
    pol = rd / f"policy_{STEP}.safetensors"
    pol.write_bytes(os.urandom(1000))
    ck = hashlib.sha256(pol.read_bytes()).hexdigest()
    rng = np.random.default_rng(0)
    pos = rng.normal(size=(T, bodies, 3))
    q = rng.normal(size=(T, bodies, 4)); q /= np.linalg.norm(q, axis=-1, keepdims=True)
    mp = root / f"runs/motion/tier1/whole_body/soma/sequence_id={SEQ}/robot_name=g1"
    mp.mkdir(parents=True)
    pq.write_table(pa.table({"fps": [30.0], "object_body_position": [pos.tolist()], "object_body_wxyz": [q.tolist()]}), mp / "data.parquet")
    parquet = f"/repo/runs/motion/tier1/whole_body/soma/sequence_id={SEQ}/robot_name=g1/data.parquet"
    prov = {"checkpoint_sha256": ck, "eval_code_sha256": dig["eval_code_sha256"], "reference_parquet": parquet,
            "eval_script_sha256": dig["eval_script_sha256"]}
    if tamper == "code":
        prov["eval_code_sha256"] = "0" * 64
    ref = np.concatenate([pos, q[..., 1:], q[..., :1]], axis=-1)
    ach = np.repeat(ref[:, None], worlds, axis=1) + 0.001
    if tamper == "nan":
        ach[3, 0, 0, 0] = np.nan
    od = rd / "export_8c778f9" / f"s{STEP}_w{worlds}"
    od.mkdir(parents=True)
    write_traj(od / f"episode_{EP:06d}.parquet", ach, ref, [f"b{i}" for i in range(bodies)], non_finite, prov)
    rep = {"world_count": worlds, "step_count": T - (tamper == "steps"), "checkpoint_sha256": ck,
           "non_finite_worlds": list(non_finite), "parquet": parquet, "metrics": {"add_auc": 0.7, "chord_sr": 0.5}}
    (od / f"episode_{EP:06d}.json").write_text(json.dumps(rep))
    if tamper == "policy":
        pol.write_bytes(os.urandom(1000))


def check(root: Path, *args) -> tuple:
    r = subprocess.run([sys.executable, "stages/evaluate/check_export.py", RUN, str(STEP), *args], cwd=root,
                       capture_output=True, text=True, timeout=120)
    return r.returncode, r.stdout + r.stderr


def case(kw: dict, args: list, want: int) -> str:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        setup(root, **kw)
        rc, out = check(root, *args)
        last = out.strip().splitlines()[-1]
        assert rc == want and last.startswith("EXPORT_OK" if want == 0 else "EXPORT_FAIL"), (kw, args, rc, out[-1500:])
        return out


def test_happy() -> None:
    out = case({}, ["1"], 0)
    m = re.search(r"位置最大差 ([0-9.]+) mm，朝向最大差 ([0-9.]+)°", out)   # 评测器记下的参考就是 motion 参考，只差 float32 舍入
    assert m and float(m.group(1)) < 0.01 and float(m.group(2)) < 0.1, out[-600:]
    case({"worlds": 2}, ["2"], 0)
    case({}, ["1", "--no-traj"], 0)


def test_two_bodies() -> None:
    case({"bodies": 2, "worlds": 2}, ["2"], 0)


def test_failures() -> None:
    for kw, args in (({"non_finite": (0,)}, ["1"]), ({"tamper": "nan"}, ["1"]), ({"tamper": "steps"}, ["1"]), ({}, ["2"]),
                     ({"tamper": "code"}, ["1"]), ({"tamper": "policy"}, ["1"])):
        case(kw, args, 1)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("ALL_OK")
