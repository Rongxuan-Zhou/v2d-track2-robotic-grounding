"""列出一份 motion parquet 的每一列：类型与（能转成数组时的）形状，并标出随帧变的列（训练机上经 scripts/dev/in-flash-chord 在容器里跑，只读）。
计划 agent-doc/plan-t3static-2026-10-21.md 第 1 节第 1 步（复核 MINOR 6）：--stand 要先列出 schema 里所有随帧变的机器人列。
用法：python stages/retarget/parquet_columns.py <motion parquet>   最后一行 PARQUET_COLUMNS_DONE
"""
import sys

import numpy as np
import pyarrow.parquet as pq


def main() -> int:
    t = pq.read_table(sys.argv[1])
    row = t.to_pylist()[0]
    T = len(row["robot_joint_positions"])
    print(f"{sys.argv[1]}：{t.num_rows} 行，{T} 帧")
    for n in t.schema.names:
        v = row[n]
        try:
            a = np.asarray(v, dtype=float)
            varies = a.ndim >= 1 and a.shape[0] == T and T > 1 and bool(np.any(a != a[:1]))
            print(f"{n}\t{t.schema.field(n).type}\tshape {a.shape}\t{'随帧变' if varies else ''}")
        except (TypeError, ValueError):
            desc = len(v) if hasattr(v, "__len__") and not isinstance(v, str) else v
            print(f"{n}\t{t.schema.field(n).type}\t{desc}")
    print("PARQUET_COLUMNS_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
