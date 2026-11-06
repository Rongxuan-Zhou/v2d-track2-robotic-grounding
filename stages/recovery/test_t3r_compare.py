"""stages/recovery/t3r_compare.py 的测试：守计划 agent-doc/plan-t3takeover-v3-2026-10-21.md 判据 K1 的两个比较器与门槛自检。"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from t3r_compare import IGNORE_KEY, compare, self_test


def _args(**kw):
    a = argparse.Namespace(tol=1e-6, tol_rad=1e-6, subset=False, all=False, ignore_key=IGNORE_KEY, quat_col=["observation.object.pose"])
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def _pose_table(pose: np.ndarray) -> pa.Table:
    return pa.table({"frame_index": list(range(len(pose))), "observation.object.pose": pose.tolist()})


def test_bytes_numbers_different_npz():
    """K1：npz 逐字节相同、数值在门槛内、超门槛三种结论。"""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        x = np.linspace(0, 1, 10)
        np.savez(p / "a.npz", x=x)
        np.savez(p / "b.npz", x=x)
        assert compare(p / "a.npz", p / "b.npz", _args())[0] == "BYTES_EQUAL"
        np.savez(p / "c.npz", x=x + 5e-7)
        assert compare(p / "a.npz", p / "c.npz", _args())[0] == "NUMBERS_EQUAL"
        np.savez(p / "e.npz", x=x + 3e-6)
        v, msg = compare(p / "a.npz", p / "e.npz", _args())
        assert v == "DIFFERENT" and "x" in msg, msg


def test_parquet_quaternion_sign_and_position():
    """K1：parquet 的物体位姿——四元数取反算相同，位置挪 2e-6 m 算不同；门槛自检三项都过。"""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        rng = np.random.default_rng(0)
        q = rng.normal(size=(20, 4))
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        pose = np.concatenate([rng.normal(size=(20, 3)), q], axis=1)
        pq.write_table(_pose_table(pose), p / "a.parquet")
        flip = pose.copy()
        flip[:, 3:] *= -1
        pq.write_table(_pose_table(flip), p / "b.parquet")
        assert compare(p / "b.parquet", p / "a.parquet", _args())[0] == "NUMBERS_EQUAL"
        shift = pose.copy()
        shift[5, 1] += 2e-6
        pq.write_table(_pose_table(shift), p / "c.parquet")
        assert compare(p / "c.parquet", p / "a.parquet", _args())[0] == "DIFFERENT"
        assert self_test(p / "a.parquet", _args()) == []


def test_json_subset_and_ignored_keys():
    """K1：JSON 去掉路径、时间字段后按字段比；--subset 时参照多出来的集不算不同，我们这份里的值不同就算不同。"""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        ours = {"time": "now", "episodes": {"ep14": {"k": 1.0, "path": "/a"}}}
        ref = {"time": "then", "episodes": {"ep14": {"k": 1.0 + 1e-9, "path": "/b"}, "ep15": {"k": 2.0}}}
        (p / "o.json").write_text(json.dumps(ours))
        (p / "r.json").write_text(json.dumps(ref))
        assert compare(p / "o.json", p / "r.json", _args(subset=True))[0] == "NUMBERS_EQUAL"
        assert compare(p / "o.json", p / "r.json", _args(subset=False))[0] == "DIFFERENT"
        ours["episodes"]["ep14"]["k"] = 1.1
        (p / "o.json").write_text(json.dumps(ours))
        assert compare(p / "o.json", p / "r.json", _args(subset=True))[0] == "DIFFERENT"


def test_other_suffix_bytes_only():
    """K1：其余后缀（如 usda）只比字节。"""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        (p / "a.usda").write_text("x 1.0\n")
        (p / "b.usda").write_text("x 1.0\n")
        (p / "c.usda").write_text("x 1.00\n")
        assert compare(p / "a.usda", p / "b.usda", _args())[0] == "BYTES_EQUAL"
        assert compare(p / "a.usda", p / "c.usda", _args())[0] == "DIFFERENT"


def test_json_subset_matches_episode_lists():
    """K1“按集逐行比”：每集一条的列表按 episode 配对，参照多出来、顺序不同都不算；值不同、参照缺集算不同。"""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        ours = [{"episode": "ep14", "k": 1.0, "tier": "t"}, {"episode": "ep27", "k": 2.0, "tier": "t"}]
        ref = [{"episode": "ep27", "k": 2.0, "tier": "t"}, {"episode": "ep05", "k": 9.0, "tier": "t"}, {"episode": "ep14", "k": 1.0, "tier": "t"}]
        (p / "o.json").write_text(json.dumps(ours, indent=1))
        (p / "r.json").write_text(json.dumps(ref))
        assert compare(p / "o.json", p / "r.json", _args(subset=True))[0] == "NUMBERS_EQUAL"
        assert compare(p / "o.json", p / "r.json", _args(subset=False))[0] == "DIFFERENT"
        ours[1]["k"] = 2.1
        (p / "o.json").write_text(json.dumps(ours))
        v, msg = compare(p / "o.json", p / "r.json", _args(subset=True))
        assert v == "DIFFERENT" and "ep27" in msg, msg
        ours[1] = {"episode": "ep99", "k": 2.0, "tier": "t"}
        (p / "o.json").write_text(json.dumps(ours))
        v, msg = compare(p / "o.json", p / "r.json", _args(subset=True))
        assert v == "DIFFERENT" and "ep99" in msg, msg


def test_all_lists_every_difference():
    """K2 排查：--all 列出所有不同的键与 JSON 路径，不带时只报第一处；结论两种情况相同。"""
    with tempfile.TemporaryDirectory() as d:
        p = Path(d)
        np.savez(p / "a.npz", x=np.zeros(3), y=np.zeros(3), z=np.zeros(3))
        np.savez(p / "b.npz", x=np.ones(3), y=np.zeros(3), z=np.ones(3))
        v1, m1 = compare(p / "a.npz", p / "b.npz", _args())
        v2, m2 = compare(p / "a.npz", p / "b.npz", _args(all=True))
        assert v1 == v2 == "DIFFERENT"
        assert "键 x" in m1 and "键 z" not in m1, m1
        assert "键 x" in m2 and "键 z" in m2 and "键 y" not in m2, m2
        (p / "o.json").write_text(json.dumps({"a": 1.0, "b": {"c": 2.0}, "d": 3.0}))
        (p / "r.json").write_text(json.dumps({"a": 1.5, "b": {"c": 2.5}, "d": 3.0}))
        v, m = compare(p / "o.json", p / "r.json", _args(all=True))
        assert v == "DIFFERENT" and "$.a" in m and "$.b.c" in m and "$.d" not in m, m


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name} 过")
    print("T3R_COMPARE_TESTS_OK")
