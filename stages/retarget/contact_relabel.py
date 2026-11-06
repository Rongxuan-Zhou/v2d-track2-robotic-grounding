"""接触标记提前、补空档（训练机上经 scripts/dev/in-flash-chord 在容器里跑；只改训练用的 hand_contact_active 一列，参考轨迹一字不动）。
计划 agent-doc/plan-t1-contact-anticipation-2026-10-16.md：FlashCHORD 的 force_closure 只在参考标记为接触的帧计分，跟丢集接触后 0.2–0.6 秒就丢，
把每只手每段接触的起点往前提 --lead 帧（不早于第 0 帧），并把两段之间不超过 --gap 帧的空档补成接触。评测只看物体，不读这个标记。
对每个序列：读 <src>/whole_body/soma/sequence_id=<序列>/robot_name=g1/data.parquet，只换 hand_contact_active，写到 <dst> 的同一相对路径；
写完读回核对：其余各列逐值相同，这一列只有 0/1、形状不变、新标记包含原标记（只增不减）。支撑面 <src>/whole_body/reconstructed_stage/<序列>_support.usda
在就照抄；元数据 <meta-src>/<序列>/v2d_meta.json 照抄到 <meta-dst>/<序列>/（fc_export.sh 与 arm_var_chain.sh 从这里取档位与 episode）。
网格路径列不改，仍指向 <src> 下的网格。目标已存在就拒绝（不覆盖）。
用法（容器内，仓库根 /repo）：python stages/retarget/contact_relabel.py --src runs/motion/tier1 --dst runs/motion/tier1_c10 \
      --meta-src runs/motion_src/tier1 --meta-dst runs/motion_src/tier1_c10 --lead 10 --gap 15 <序列>...；最后一行 CONTACT_RELABEL_DONE <序列数>。
测试：stages/retarget/test_contact_relabel.py。
"""
import argparse
import shutil
import sys
from pathlib import Path

import numpy as np

COL = "hand_contact_active"


def segments(x: np.ndarray) -> list:
    """布尔序列里连续为真的段，[起, 止) 列表。"""
    d = np.diff(np.r_[0, x.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def relabel(c: np.ndarray, lead: int, gap: int) -> np.ndarray:
    """c：手 × 帧的 0/1。先补不超过 gap 帧的空档，再把每段起点提前 lead 帧。"""
    if lead < 0 or gap < 0:
        raise ValueError("lead、gap 要 ≥ 0")
    out = np.zeros_like(c)
    for h in range(c.shape[0]):
        segs = segments(c[h] > 0.5)
        merged = []
        for s, e in segs:
            if merged and s - merged[-1][1] <= gap:
                merged[-1] = (merged[-1][0], e)
            else:
                merged.append((s, e))
        for s, e in merged:
            out[h, max(0, s - lead):e] = 1
    return out


def rewrite(src: Path, dst: Path, lead: int, gap: int) -> str:
    import pyarrow as pa
    import pyarrow.parquet as pq

    t = pq.read_table(src)
    if t.num_rows != 1:
        raise SystemExit(f"{src}：应是单行（按帧存的列表），实际 {t.num_rows} 行")
    i = t.schema.get_field_index(COL)
    old = np.asarray(t.column(COL).to_pylist()[0], dtype=float)
    # 形状必须是 手数 × 帧数（10/16 有一次按帧读错了方向；独立复核 MINOR 2）：手数取 hand_sides，帧数取 robot_root_position
    hands = len(t.column("hand_sides").to_pylist()[0]) if "hand_sides" in t.column_names else None
    frames = len(t.column("robot_root_position").to_pylist()[0]) if "robot_root_position" in t.column_names else None
    want = (hands if hands is not None else old.shape[0], frames if frames is not None else old.shape[1])
    if old.ndim != 2 or old.shape != want or not np.isin(old, (0.0, 1.0)).all():
        raise SystemExit(f"{src}：{COL} 不是 手 × 帧 的 0/1（形状 {old.shape}，应为 {want}）")
    new = relabel(old, lead, gap).astype(float)
    field = t.schema.field(i)
    t2 = t.set_column(i, field, pa.array([new.tolist()], type=field.type))
    dst.parent.mkdir(parents=True, exist_ok=False)
    pq.write_table(t2, dst)
    back = pq.read_table(dst)
    for name in t.column_names:
        if name != COL and not back.column(name).equals(t.column(name)):
            raise SystemExit(f"{dst}：列 {name} 读回与原文件不同")
    nb = np.asarray(back.column(COL).to_pylist()[0], dtype=float)
    if nb.shape != old.shape or not np.array_equal(nb, new) or ((old > 0.5) & (nb < 0.5)).any():
        raise SystemExit(f"{dst}：{COL} 读回不对（形状、取值，或丢了原来的接触帧）")
    parts = [f"形状 {old.shape}（手 × 帧）"]
    for h in range(old.shape[0]):
        so, sn = segments(old[h] > 0.5), segments(new[h] > 0.5)
        parts.append(f"手 {h}：接触帧 {int((old[h] > 0.5).sum())} → {int((new[h] > 0.5).sum())}，段数 {len(so)} → {len(sn)}，"
                     f"首段起点 {so[0][0] if so else None} → {sn[0][0] if sn else None}")
    return "；".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    for k in ("--src", "--dst", "--meta-src", "--meta-dst"):
        ap.add_argument(k, required=True, type=Path)
    ap.add_argument("--lead", type=int, required=True)
    ap.add_argument("--gap", type=int, required=True)
    ap.add_argument("seqs", nargs="+")
    a = ap.parse_args()
    for seq in a.seqs:
        rel = Path("whole_body/soma") / f"sequence_id={seq}" / "robot_name=g1" / "data.parquet"
        src, dst = a.src / rel, a.dst / rel
        if not src.exists():
            raise SystemExit(f"没有 {src}")
        if dst.exists():
            raise SystemExit(f"{dst} 已存在，不覆盖")
        print(f"{seq}：{rewrite(src, dst, a.lead, a.gap)}")
        stage = a.src / "whole_body/reconstructed_stage"
        sups = sorted(stage.glob(f"{seq}_*support.usda")) if stage.is_dir() else []   # 含 <序列>_g1_support.usda 的写法
        for sup in sups:
            out = a.dst / "whole_body/reconstructed_stage" / sup.name
            out.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sup, out)
        print(f"  支撑面：{'、'.join(x.name for x in sups) if sups else '没有（只有地面）'}")
        meta = a.meta_src / seq / "v2d_meta.json"
        if not meta.exists():
            raise SystemExit(f"没有 {meta}")
        (a.meta_dst / seq).mkdir(parents=True, exist_ok=False)
        shutil.copy2(meta, a.meta_dst / seq / "v2d_meta.json")
    print(f"CONTACT_RELABEL_DONE {len(a.seqs)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
