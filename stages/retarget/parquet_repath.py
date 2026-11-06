"""把一份 motion parquet 里物体网格与 URDF 的绝对路径前缀换成训练容器里的 /repo（训练机上经 scripts/dev/in-flash-chord 在容器里跑，只用 CPU）。
起因：agent-doc/plan-t3flip13-2026-10-27.md 执行记录：ml40 上重定向在 t3recon/gate_e/repo 里跑，写进 parquet 的 object_mesh_paths、object_urdf_paths 指向那个目录，
训练容器只挂训练仓库（/repo），check_inputs.py 报缺失。只改这两列的前缀，其余列逐值不动；写回前核对新路径在容器里都存在。
用法（容器内）：python /repo/stages/retarget/parquet_repath.py <parquet> <旧前缀> <新前缀>   原地改写，先写 .partial 再改名；最后一行 PARQUET_REPATH_DONE <改了几条>
"""
import os
import sys

import pyarrow as pa
import pyarrow.parquet as pq


def main() -> int:
    path, old, new = sys.argv[1], sys.argv[2], sys.argv[3]
    t = pq.read_table(path)
    changed = 0
    t2 = t
    for col in ("object_mesh_paths", "object_urdf_paths"):
        if col not in t2.column_names:
            continue
        vals = t2.column(col).to_pylist()
        out = [[p.replace(old, new, 1) if p.startswith(old) else p for p in row] for row in vals]
        for row in out:
            for p in row:
                if not os.path.exists(p):
                    print(f"新路径不存在：{p}")
                    return 2
        changed += sum(a != b for ra, rb in zip(vals, out) for a, b in zip(ra, rb))
        t2 = t2.set_column(t2.schema.get_field_index(col), t2.schema.field(col), pa.array(out, type=t2.schema.field(col).type))
    # 其余列逐值相同
    for name in t.column_names:
        if name not in ("object_mesh_paths", "object_urdf_paths") and not t.column(name).equals(t2.column(name)):
            print(f"列 {name} 变了，拒绝")
            return 3
    pq.write_table(t2, path + ".partial")
    os.replace(path + ".partial", path)
    print(f"PARQUET_REPATH_DONE {changed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
