"""参考 motion parquet 里物体相邻帧转角的统计（训练机上经 scripts/dev/in-flash-chord 在容器里跑，只用 CPU；只读，只报不判）。
起因：agent-doc/plan-t3hand-2026-10-27.md 第 3 版 H0(b)：去跳变要在训练真正读的 motion parquet 上再量一次。
每份参考一行：帧数、相邻帧转角 > --jump（默认 24°）的帧数与前 12 个帧号、转角的 99% 分位与最大值；--window a:b 另列这段里每帧的转角。
读 object_body_wxyz 的第一个刚体（四元数 wxyz）。
用法（容器内）：python /repo/stages/retarget/ref_object_turns.py [--jump 24] [--window a:b] <motion parquet>...   最后一行 REF_OBJECT_TURNS_DONE
"""
import argparse
import sys

import numpy as np
import pyarrow.parquet as pq


def quat_mat(q):
    q = np.asarray(q, dtype=float)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z); R[..., 0, 1] = 2 * (x * y - z * w); R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w); R[..., 1, 1] = 1 - 2 * (x * x + z * z); R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w); R[..., 2, 1] = 2 * (y * z + x * w); R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jump", type=float, default=24.0)
    ap.add_argument("--window", default=None)
    ap.add_argument("parquet", nargs="+")
    a = ap.parse_args()
    for path in a.parquet:
        row = pq.read_table(path).to_pylist()[0]
        R = quat_mat(np.asarray(row["object_body_wxyz"], dtype=float)[:, 0])
        rel = np.einsum("nji,njk->nik", R[:-1], R[1:])
        td = np.r_[0.0, np.degrees(np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1)))]
        big = np.flatnonzero(td > a.jump)
        print(f"{path}：{len(td)} 帧；转角 > {a.jump:g}° 的帧 {len(big)} 个 {big[:12].tolist()}；转角 99% {np.percentile(td, 99):.1f}°、最大 {td.max():.1f}°（第 {int(td.argmax())} 帧）")
        if a.window:
            s0, s1 = (int(x) for x in a.window.split(":"))
            print("  " + " ".join(f"{t}:{td[t]:.0f}" for t in range(s0, min(s1, len(td)))))
    print("REF_OBJECT_TURNS_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
