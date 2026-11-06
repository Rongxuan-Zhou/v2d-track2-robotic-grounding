"""以 SOMAWithFlex 运行官方 scripts/retarget/soma_to_g1.py（容器内），并修正输出的 fps。

参数原样转给官方脚本，本包装额外做两件事：
1. 人体读取走 SOMAWithFlex，bone_length_flexibles 生效（见 soma_flex.py）。
2. 修正 fps。官方把 IK 求解器频率 kin.frequency（默认 200）写进了输出的 fps
   （soma_to_g1.py:914/1075/1119），可数据本身是 30 fps、逐帧不插值。训练端
   mdp/utils.py:50 用 arange(n)/fps 生成时间戳，不修的话动作快 6.7 倍。
   求解器频率不能改——它是微分 IK 的积分步长，改了 IK 行为就和官方不一致——
   所以只改写存盘 parquet 的 fps 字段，值取自转换器写的 v2d_meta.json。
   另两处（FirstPassResult 与 first_pass_cache）只供仓库里不存在的
   rerun_post_process.py 使用，且缓存落在一次性容器的 /tmp，不处理。

强制要求 --save 与 --motion-root：官方默认 motion-root 是 third_party 里的资产目录，
不传就会往官方目录树里写产物。

用法：scripts/dev/incontainer stages/retarget/run_soma_to_g1.py <data_folder> --save --motion-root <dir>
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "libs" / "v2d_infra"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pyarrow as pa
import pyarrow.parquet as pq
from paths import UPSTREAM
from soma_flex import load_official_soma_to_g1


def fix_fps(parquet: Path, source_fps: float) -> float:
    t = pq.read_table(parquet)
    old = t.column("fps").to_pylist()
    assert len(set(old)) == 1, f"fps 列不唯一: {set(old)}"
    i = t.schema.get_field_index("fps")
    t = t.set_column(i, t.schema.field(i), pa.array([source_fps] * t.num_rows, type=t.schema.field(i).type))
    pq.write_table(t, parquet)
    return old[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("data_folder")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--motion-root")
    ap.add_argument("--soma-subdir", default="soma")
    ap.add_argument("--robot-name", default="g1")
    a, _ = ap.parse_known_args()
    if not (a.save and a.motion_root):
        sys.exit("必须同时传 --save 和 --motion-root（否则官方默认写进 third_party）")
    meta = json.loads((Path(a.data_folder) / "v2d_meta.json").read_text())

    mod = load_official_soma_to_g1(UPSTREAM)
    sys.argv = ["soma_to_g1.py"] + sys.argv[1:]
    mod.main()

    out = (Path(a.motion_root) / "whole_body" / a.soma_subdir
           / f"sequence_id={Path(a.data_folder).name}" / f"robot_name={a.robot_name}" / "data.parquet")
    old = fix_fps(out, meta["source_fps"])
    print(f"[v2d] fps 修正: {old} -> {meta['source_fps']}  ({out})")
