"""t3takeover 第 1 级的比较器（只读；要 numpy、pyarrow）。计划 agent-doc/plan-t3takeover-v3-2026-10-21.md 判据 K1 与复核第 2 条。
对一对文件（我们的、参照）给三种结论之一：BYTES_EQUAL（逐字节相同）、NUMBERS_EQUAL（字节不同、数值在门槛内相同）、DIFFERENT（数值也不同；报第一处的键、列与差值）。
- .npz：逐键比，键集合要相同；浮点数组 |差| ≤ --tol，其余 array_equal。
- .parquet：逐列比，列集合要相同；数值列展开后 |差| ≤ --tol；四元数列（--quat-col，默认 observation.object.pose 的后 4 维）按 q ≡ −q 比，转角 ≤ --tol-rad；其余列要相等。
- .json：按字段比，去掉路径、时间类字段（--ignore-key 的正则）；--subset 时只比我们这份里有的键（参照有 30 集、我们只有 3 集），
  两边都是“每集一条”的列表（每条是带 episode 键的字典）时按 episode 配对比（判据 K1 的“按集逐行比”）。浮点 |差| ≤ --tol。
- 其余后缀：只比字节。
--all：不在第一处不同停下，列出所有不同的键、列与 JSON 路径（K2 排查用；结论不变）。
--self-test：对给定的第一对文件做门槛自检（字节改一位要判“不同”；parquet 的一个位置挪 2·tol 要判 DIFFERENT、一个四元数取反要判 NUMBERS_EQUAL），不过就退出码 3。
用法：python stages/recovery/t3r_compare.py [--tol 1e-6] [--tol-rad 1e-6] [--subset] [--all] [--self-test] <我们的> <参照> [<我们的> <参照> ...]
每对一行结论，最后一行 T3R_COMPARE_DONE 相同字节 a、数值相同 b、不同 c（有不同时退出码 1）。测试：stages/recovery/test_t3r_compare.py。
"""
import argparse
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

IGNORE_KEY = r"(path|dir|file|root|time|date|stamp|argv|host|cwd|user|from|created|deployed|code_files)"


def _num_close(a: np.ndarray, b: np.ndarray, tol: float) -> tuple[bool, str]:
    if a.shape != b.shape:
        return False, f"形状 {a.shape} 对 {b.shape}"
    if a.dtype.kind in "fc" or b.dtype.kind in "fc":
        a64, b64 = a.astype(np.float64), b.astype(np.float64)
        both_nan = np.isnan(a64) & np.isnan(b64)
        d = np.where(both_nan, 0.0, np.abs(a64 - b64))
        m = float(np.nanmax(d)) if d.size else 0.0
        return m <= tol, f"最大差 {m:.3e}"
    eq = np.array_equal(a, b)
    return eq, "相等" if eq else "不相等"


def _quat_close(a: np.ndarray, b: np.ndarray, tol_rad: float) -> tuple[bool, str]:
    a, b = a.astype(np.float64), b.astype(np.float64)
    if a.shape != b.shape:
        return False, f"形状 {a.shape} 对 {b.shape}"
    an = a / np.linalg.norm(a, axis=-1, keepdims=True)
    bn = b / np.linalg.norm(b, axis=-1, keepdims=True)
    dot = np.clip(np.abs(np.sum(an * bn, axis=-1)), 0.0, 1.0)
    ang = 2.0 * np.arccos(dot)
    m = float(ang.max()) if ang.size else 0.0
    return m <= tol_rad, f"四元数最大转角 {m:.3e} rad"


def _verdict(bad: list[str], ok_msg: str) -> tuple[bool, str]:
    return (False, "；".join(bad)) if bad else (True, ok_msg)


def cmp_npz(ours: Path, ref: Path, a) -> tuple[bool, str]:
    x, y = np.load(ours, allow_pickle=False), np.load(ref, allow_pickle=False)
    if sorted(x.files) != sorted(y.files):
        return False, f"键不同：{sorted(set(x.files) ^ set(y.files))}"
    bad = []
    for k in sorted(x.files):
        ok, msg = _num_close(x[k], y[k], a.tol)
        if not ok:
            bad.append(f"键 {k}：{msg}")
            if not a.all:
                break
    return _verdict(bad, "各键都在门槛内")


def _col(table, name: str) -> np.ndarray:
    v = table.column(name).to_pylist()
    try:
        return np.asarray(v, dtype=np.float64)
    except (TypeError, ValueError):
        return np.asarray(v, dtype=object)


def cmp_parquet(ours: Path, ref: Path, a) -> tuple[bool, str]:
    x, y = pq.read_table(ours), pq.read_table(ref)
    if x.column_names != y.column_names:
        return False, f"列不同：{sorted(set(x.column_names) ^ set(y.column_names))}"
    bad = []
    for c in x.column_names:
        if bad and not a.all:
            break
        u, v = _col(x, c), _col(y, c)
        if u.dtype == object or v.dtype == object:
            if x.column(c).to_pylist() != y.column(c).to_pylist():
                bad.append(f"列 {c}：不相等")
            continue
        if c in a.quat_col and u.ndim >= 2 and u.shape[-1] >= 4:
            ok, msg = _num_close(u[..., :-4], v[..., :-4], a.tol) if u.shape[-1] > 4 else (True, "")
            if not ok:
                bad.append(f"列 {c} 的位置：{msg}")
            ok, msg = _quat_close(u[..., -4:], v[..., -4:], a.tol_rad)
            if not ok:
                bad.append(f"列 {c}：{msg}")
            continue
        ok, msg = _num_close(u, v, a.tol)
        if not ok:
            bad.append(f"列 {c}：{msg}")
    return _verdict(bad, "各列都在门槛内")


def _by_episode(u: list, v: list) -> bool:
    return all(isinstance(e, dict) and "episode" in e for e in u + v)


def _json_diff(u, v, a, where: str, bad: list[str]) -> None:
    """把 u（我们的）对 v（参照）的不同处追加到 bad；不带 --all 时有一处就停。"""
    if bad and not a.all:
        return
    if isinstance(u, dict) and isinstance(v, dict):
        keys = [k for k in u if not re.search(a.ignore_key, str(k), re.IGNORECASE)]
        if not a.subset:
            other = [k for k in v if not re.search(a.ignore_key, str(k), re.IGNORECASE)]
            if sorted(keys) != sorted(other):
                bad.append(f"{where}：键不同 {sorted(set(keys) ^ set(other))}")
                return
        for k in keys:
            if k not in v:
                bad.append(f"{where}：参照里没有键 {k}")
            else:
                _json_diff(u[k], v[k], a, f"{where}.{k}", bad)
            if bad and not a.all:
                return
        return
    if isinstance(u, list) and isinstance(v, list):
        if a.subset and _by_episode(u, v):  # 每集一条：按 episode 配对，参照多出来的集不算
            ref = {e["episode"]: e for e in v}
            if len(ref) != len(v) or len({e["episode"] for e in u}) != len(u):
                bad.append(f"{where}：episode 有重复")
                return
            for e in u:
                if e["episode"] not in ref:
                    bad.append(f"{where}：参照里没有 {e['episode']}")
                else:
                    _json_diff(e, ref[e["episode"]], a, f"{where}[{e['episode']}]", bad)
                if bad and not a.all:
                    return
            return
        if len(u) != len(v):
            bad.append(f"{where}：长度 {len(u)} 对 {len(v)}")
            return
        for i, (p, q) in enumerate(zip(u, v)):
            _json_diff(p, q, a, f"{where}[{i}]", bad)
            if bad and not a.all:
                return
        return
    if isinstance(u, (int, float)) and isinstance(v, (int, float)) and not isinstance(u, bool) and not isinstance(v, bool):
        d = abs(float(u) - float(v))
        if d > a.tol:
            bad.append(f"{where}：差 {d:.3e}（{u!r} 对 {v!r}）")
        return
    if u != v:
        bad.append(f"{where}：{u!r} 对 {v!r}")


def cmp_json(ours: Path, ref: Path, a) -> tuple[bool, str]:
    bad: list[str] = []
    _json_diff(json.loads(ours.read_text()), json.loads(ref.read_text()), a, "$", bad)
    return _verdict(bad, "")


def compare(ours: Path, ref: Path, a) -> tuple[str, str]:
    if ours.read_bytes() == ref.read_bytes():
        return "BYTES_EQUAL", ""
    fn = {".npz": cmp_npz, ".parquet": cmp_parquet, ".json": cmp_json}.get(ours.suffix)
    if fn is None:
        return "DIFFERENT", "字节不同（这种文件只比字节）"
    ok, msg = fn(ours, ref, a)
    return ("NUMBERS_EQUAL", msg) if ok else ("DIFFERENT", msg)


def self_test(ours: Path, a) -> list[str]:
    """门槛自检：返回不过的项（空表示都过）。"""
    bad = []
    data = bytearray(ours.read_bytes())
    data[len(data) // 2] ^= 0x01
    tmp = ours.with_name(ours.stem + ".selftest" + ours.suffix)
    try:
        tmp.write_bytes(bytes(data))
        if ours.read_bytes() == tmp.read_bytes():
            bad.append("改一个字节后字节比较仍判相同")
        if ours.suffix == ".parquet":
            t = pq.read_table(ours)
            c = next((c for c in a.quat_col if c in t.column_names), None)
            if c is not None:
                arr = _col(t, c)
                for kind, change, want in (("挪 2·tol", "shift", "DIFFERENT"), ("取反四元数", "negate", "NUMBERS_EQUAL")):
                    arr2 = arr.copy()
                    if change == "shift":
                        arr2[0, 0] += 2 * a.tol
                    else:
                        arr2[0, -4:] *= -1
                    f = t.schema.field(c)
                    vals = [arr2.tolist()] if t.num_rows == 1 and arr2.ndim >= 2 and len(arr2) != 1 else arr2.tolist()
                    t2 = t.set_column(t.schema.get_field_index(c), f, pa.array(vals, type=f.type))
                    buf = io.BytesIO()
                    pq.write_table(t2, buf)
                    tmp.write_bytes(buf.getvalue())
                    got, _ = compare(tmp, ours, a)
                    if got != want:
                        bad.append(f"{kind}：判成 {got}，应为 {want}")
    finally:
        tmp.unlink(missing_ok=True)
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--tol-rad", type=float, default=1e-6)
    ap.add_argument("--subset", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--ignore-key", default=IGNORE_KEY)
    ap.add_argument("--quat-col", action="append", default=None)
    a = ap.parse_args()
    a.quat_col = a.quat_col or ["observation.object.pose"]
    if len(a.files) % 2:
        print("文件要成对给（我们的、参照）")
        return 2
    pairs = [(Path(a.files[i]), Path(a.files[i + 1])) for i in range(0, len(a.files), 2)]
    if a.self_test:
        bad = self_test(pairs[0][0], a)
        print("门槛自检：" + ("都过" if not bad else "；".join(bad)))
        if bad:
            return 3
    counts = {"BYTES_EQUAL": 0, "NUMBERS_EQUAL": 0, "DIFFERENT": 0}
    for ours, ref in pairs:
        verdict, msg = compare(ours, ref, a)
        counts[verdict] += 1
        print(f"{verdict}\t{ours}\t{ref}" + (f"\t{msg}" if msg else ""))
    print(f"T3R_COMPARE_DONE 相同字节 {counts['BYTES_EQUAL']}、数值相同 {counts['NUMBERS_EQUAL']}、不同 {counts['DIFFERENT']}")
    return 1 if counts["DIFFERENT"] else 0


if __name__ == "__main__":
    sys.exit(main())
