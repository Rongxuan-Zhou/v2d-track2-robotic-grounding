"""比较两版 Tier 3 档目录里同一集 parquet 的各列（训练机上、仓库根执行，只读，只报）。
计划 agent-doc/plan-t3-simreg-p2-2026-10-11.md：simreg 与 f830 重定向后的间隙几乎相同（最大差 0.9 mm），查两版到底改了哪些列。
经 scripts/dev/pipe-py 传过去，用宿主机 python3 的 pyarrow 与 numpy（mf4090：pyarrow 25.0.1、numpy 2.2.6，10/11 查过；
pipe-py 的说明写的是只用标准库，这个脚本是例外，只在 mf4090 上跑）。
开头调 tier_guard.forbid_tier1()：本进程读不到 Tier 1 的任何文件。
输入：两个档目录（如 runs/tier3/root_v2、runs/tier3/root_simreg）与集号，读 <目录>/data/chunk-000/episode_<NNNNNN>.parquet。
输出：每集一段：两版的行数；两版都有的每一列：数值列（含定长列表）给最大绝对差、有差的帧数、一边是 NaN 一边不是的元素数，
列表宽度不超过 8 的另给每个分量的最大绝对差；非数值列或形状不同的列只报相同或不同；只在一版里有的列。
最后每列一行汇总：有差的集数、各集最大绝对差的最大值。结束行 T3_VERSION_DIFF_DONE。
用法：scripts/dev/pipe-py stages/recovery/t3_version_diff.py <A 目录> <B 目录> <集>...
      scripts/dev/pipe-py stages/recovery/t3_version_diff.py --selftest   （在临时目录里造两份小 parquet，核对输出）
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "libs" / "v2d_infra"))
from tier_guard import forbid_tier1  # noqa: E402

forbid_tier1()
import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402


def as_array(t, name):
    try:
        return np.asarray(t.column(name).to_pylist(), dtype=float)
    except (TypeError, ValueError):
        return None


def compare(a_dir: Path, b_dir: Path, eps: list) -> list:
    out, summary = [], {}
    for ep in eps:
        ta = pq.read_table(a_dir / f"data/chunk-000/episode_{ep:06d}.parquet")
        tb = pq.read_table(b_dir / f"data/chunk-000/episode_{ep:06d}.parquet")
        out.append(f"## ep{ep}：{a_dir.name} {ta.num_rows} 行，{b_dir.name} {tb.num_rows} 行")
        ca, cb = set(ta.column_names), set(tb.column_names)
        for name in sorted(ca & cb):
            xa, xb = as_array(ta, name), as_array(tb, name)
            if xa is not None and xb is not None and xa.shape == xb.shape and xa.size:
                d = np.abs(xa - xb)
                dmax = float(np.nanmax(d)) if not np.all(np.isnan(d)) else 0.0
                rows = d.reshape(len(d), -1)
                frames = int(np.count_nonzero(np.nan_to_num(rows, nan=0.0).max(axis=1) > 0))
                nan_mis = int(np.count_nonzero(np.isnan(xa) != np.isnan(xb)))
                line = f"  {name}：最大绝对差 {dmax:.6g}，有差的帧 {frames}/{len(d)}，NaN 不一致 {nan_mis}"
                if xa.ndim == 2 and 1 < xa.shape[1] <= 8:
                    comp = np.nan_to_num(d, nan=0.0).max(axis=0)
                    line += "，各分量 [" + ", ".join(f"{v:.4g}" for v in comp) + "]"
                out.append(line)
                summary.setdefault(name, []).append(dmax if nan_mis == 0 else float("inf"))
            else:
                same = ta.column(name).equals(tb.column(name))
                out.append(f"  {name}：{'相同' if same else '不同'}（非数值或形状不同）")
                summary.setdefault(name, []).append(0.0 if same else float("inf"))
        out += [f"  只在 {a_dir.name}：{n}" for n in sorted(ca - cb)] + [f"  只在 {b_dir.name}：{n}" for n in sorted(cb - ca)]
    out.append("## 汇总（列、有差的集数、各集最大绝对差的最大值）")
    for name, v in sorted(summary.items()):
        out.append(f"{name}\t{sum(1 for x in v if x > 0)}/{len(v)}\t{max(v):.6g}")
    return out + ["T3_VERSION_DIFF_DONE"]


def selftest() -> int:
    with tempfile.TemporaryDirectory() as t:
        for sub, shift in (("a", 0.0), ("b", 0.5)):
            d = Path(t) / sub / "data/chunk-000"
            d.mkdir(parents=True)
            pq.write_table(pa.table({"observation.object.pose": [[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]] * 3,
                                     "observation.human.translation": [[0.0, 0.0, shift], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                                     "task": ["x", "x", "y" if shift else "x"]}), d / "episode_000007.parquet")
        out = compare(Path(t) / "a", Path(t) / "b", [7])
    text = "\n".join(out)
    ok = ("observation.object.pose：最大绝对差 0，有差的帧 0/3" in text
          and "observation.human.translation：最大绝对差 0.5，有差的帧 1/3，NaN 不一致 0，各分量 [0, 0, 0.5]" in text
          and "task：不同（非数值或形状不同）" in text and "observation.human.translation\t1/1\t0.5" in text
          and text.endswith("T3_VERSION_DIFF_DONE"))
    print(text + ("\nSELFTEST_OK" if ok else "\nSELFTEST_FAIL"))
    return 0 if ok else 1


def main(argv: list) -> int:
    if argv == ["--selftest"]:
        return selftest()
    if len(argv) < 3 or not all(x.isdigit() for x in argv[2:]):
        raise SystemExit("用法：t3_version_diff.py <A 目录> <B 目录> <集>...（或 --selftest）")
    for line in compare(Path(argv[0]), Path(argv[1]), [int(x) for x in argv[2:]]):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
