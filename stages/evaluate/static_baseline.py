"""“物体全程不动”基线与参考旋转剖面（FlashCHORD 容器内执行，只读）。

给一个 policy checkpoint：从它内嵌的 resolved config 取参考 parquet 与回放设置，按评测器同样的方式载入参考
（flash_chord.data.loader.load_reference），把第 0 帧的物体位姿保持到底当作“策略输出”，用评测器自己的
object_add / add_auc 算 ADD-AUC。一个 policy 的 ADD-AUC 不高于这个数，说明它对物体的跟踪不比“不碰物体”好。

顶点按评测器 sample_object_vertices 的规则取：物体网格顶点里用 ADD_VERTEX_SEED 随机抽 ADD_VERTEX_COUNT 个。
评测器读的是仿真模型里已缩放、已变换的网格；这里用 trimesh 读同一个网格文件，若两者顶点顺序或数目不同，
抽中的子集就不同，所以这个数与评测器“同定义、近似同顶点”。

--tier1-episode N 另按 2026-09-27 最初的算法复算一遍，对账实验记录里的 0.4708：Tier 1 数据集的物体位姿，
libs/v2d_infra/add_metrics 的网格表面 500 点（seed=episode），阈值与积分同评测器。两种算法都报。

用法：scripts/dev/in-flash-chord 'python /repo/stages/evaluate/static_baseline.py \
        /repo/runs/flash_chord/<run>/policy_<N>.safetensors [--tier1-episode 20]'
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]


def embedded_config(policy: str) -> dict:
    with open(policy, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        md = json.loads(f.read(n))["__metadata__"]
    return json.loads(md["user.resolved_config_json"])


def rotation_profile(quat: np.ndarray, gate_rad: float) -> None:
    """quat (T,4)，分量顺序不限（只用点积）。打印相对第 0 帧的测地角与逐帧增量。"""
    q = quat / np.linalg.norm(quat, axis=1, keepdims=True)
    ang0 = np.degrees(2 * np.arccos(np.clip(np.abs(q @ q[0]), 0, 1)))
    inc = np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(q[1:] * q[:-1], axis=1)), 0, 1)))
    moving = np.where(inc > 0.5)[0] + 1
    cross = np.where(ang0 > np.degrees(gate_rad))[0]
    print(f"  angle vs frame 0: mean {ang0.mean():.1f} deg, max {ang0.max():.1f} deg (frame {ang0.argmax()}), final {ang0[-1]:.1f} deg")
    print(f"  per-frame increment: median {np.median(inc):.3f} deg, p99 {np.percentile(inc, 99):.2f}, max {inc.max():.2f} (frame {inc.argmax() + 1}); "
          f"> 5 deg: {int(np.sum(inc > 5))} frames")
    print(f"  cumulative rotation {inc.sum():.1f} deg; frames with increment > 0.5 deg: {moving.size}"
          + (f" (first {moving.min()}, last {moving.max()})" if moving.size else ""))
    print(f"  a static object first exceeds the {gate_rad} rad ({np.degrees(gate_rad):.1f} deg) gate at frame "
          + (str(cross[0]) if cross.size else "never") + f"; frames above the gate: {cross.size}")
    print("  angle vs frame 0 every 20 frames: " + " ".join(f"{i}:{ang0[i]:.0f}" for i in range(0, len(ang0), 20)))


def evaluator_baseline(cfg: dict, gate_rad: float) -> None:
    import trimesh
    from flash_chord.assets.registry import resolve_asset_path
    from flash_chord.data.loader import load_reference
    from flash_chord.evaluation.metrics import (
        ADD_VERTEX_COUNT,
        ADD_VERTEX_SEED,
        add_auc,
        maniptrans_success,
        object_add,
        object_mppe_cm,
        spider_success,
    )
    from flash_chord.evaluation.rollout import reference_object_pose

    parquet = cfg["task"]["parquet"].strip("'")
    ref = load_reference(parquet, control_fps=cfg["reference"]["playback_fps"],
                         motion_speed=cfg["task"]["motion_speed"],
                         source_frame_playback=cfg["task"]["source_frame_playback"])
    pose = reference_object_pose(ref, 0, len(ref.object_body_pos_w()))      # (T,B,7) 位置 + xyzw
    static = np.broadcast_to(pose[:1], pose.shape)[:, None]                  # (T,1,B,7)：一个“世界”
    mesh_paths = ref.object_mesh_paths
    mesh_paths = mesh_paths() if callable(mesh_paths) else mesh_paths
    rng = np.random.default_rng(ADD_VERTEX_SEED)
    verts = []
    for mp in mesh_paths:
        v = np.asarray(trimesh.load(resolve_asset_path(str(mp), reference_path=parquet), force="mesh").vertices, dtype=np.float64)
        verts.append(v[rng.choice(v.shape[0], size=ADD_VERTEX_COUNT, replace=v.shape[0] < ADD_VERTEX_COUNT)])
    add = object_add(static, pose, np.stack(verts))
    auc, per_body = add_auc(add)
    drift = np.linalg.norm(pose[:, :, :3] - pose[:1, :, :3], axis=-1).max() * 100
    print(f"[evaluator definition] reference {parquet}")
    print(f"  frames {pose.shape[0]}, bodies {pose.shape[1]}, source_frame_playback {cfg['task']['source_frame_playback']}, "
          f"playback_fps {cfg['reference']['playback_fps']}, vertices {ADD_VERTEX_COUNT} per body (seed {ADD_VERTEX_SEED}, trimesh)")
    print(f"  static-object ADD-AUC {auc:.4f} (per body {', '.join(f'{x:.4f}' for x in per_body)}); "
          f"mean ADD {add.mean() * 100:.2f} cm; max position drift {drift:.1f} cm")
    # 其余物体指标也用评测器自己的函数算，与评测 JSON 的同名字段逐项可比（chord_sr 需要仿真里的 palm/pelvis 误差，算不了）
    sp_ok, sp_pos, sp_ori = spider_success(static, pose)
    mt_ok, _, _ = maniptrans_success(static, pose, np.zeros(pose.shape[1], dtype=np.int64))
    mppe = object_mppe_cm(static, pose)
    print(f"  static-object object_position_error_m {sp_pos.mean():.4f}; object_orientation_error_deg {np.degrees(sp_ori.mean()):.2f}; "
          f"mppe_cm {mppe:.3f}; spider_sr_uncentered {sp_ok.mean():.3f}; maniptrans_sr {mt_ok.mean():.3f}")
    # 机器可读的一行（字段名同评测 JSON），供 stages/evaluate/verdict.py 当“不动基线”读
    print("  summary_json " + json.dumps({
        "add_auc": auc, "mean_add_m": float(add.mean()), "object_position_error_m": float(sp_pos.mean()),
        "object_orientation_error_deg": float(np.degrees(sp_ori.mean())), "mppe_cm": mppe,
        "spider_sr_uncentered": float(sp_ok.mean()), "maniptrans_sr": float(mt_ok.mean()), "parquet": parquet,
    }, sort_keys=True))
    xyzw = pose[:, 0, 3:7]
    rotation_profile(xyzw, gate_rad)


def tier1_baseline(ep: int, gate_rad: float) -> None:
    import pyarrow.parquet as pq

    sys.path.insert(0, str(REPO / "libs" / "v2d_infra"))
    import add_metrics as A
    from paths import TIER1

    def col(t, c):
        v = t.column(c).to_numpy(zero_copy_only=False)
        return np.stack(v) if v.dtype == object else np.asarray(v)

    meta = {}
    with open(TIER1 / "meta" / "episodes_metadata.jsonl") as f:
        for line in f:
            m = json.loads(line)
            meta[m["episode_index"]] = m
    files = [f for f in sorted((TIER1 / "data" / "chunk-000").glob("*.parquet"))
             if int(pq.read_table(f, columns=["episode_index"]).column(0)[0].as_py()) == ep]
    if len(files) != 1:
        raise SystemExit(f"episode {ep}: expected one parquet, found {len(files)}")
    p = col(pq.read_table(files[0]), "observation.object.pose").astype(np.float64)
    v = A.sample_mesh_vertices(str(TIER1 / meta[ep]["mesh"]), n=500, seed=ep)
    ref = A.transform_vertices(v, p)
    static = A.transform_vertices(v, np.repeat(p[:1], len(p), axis=0))
    add = np.linalg.norm(ref - static, axis=-1).mean(axis=-1)
    th = np.arange(0.01, 0.10, 0.01)
    auc = float(np.trapezoid([np.mean(add < x) for x in th], x=np.linspace(0, 1, th.size)))
    drift = np.linalg.norm(p[:, :3] - p[0, :3], axis=1).max() * 100
    print(f"[original 2026-09-27 method] Tier 1 episode {ep} {meta[ep].get('sequence_id', '')} ({files[0].name})")
    print(f"  frames {len(p)}; static-object ADD-AUC {auc:.4f}; mean ADD {add.mean() * 100:.1f} cm; max position drift {drift:.1f} cm")
    rotation_profile(p[:, 3:7], gate_rad)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("policy", help="policy_<N>.safetensors（容器内路径）")
    ap.add_argument("--tier1-episode", type=int, help="另按最初的算法在 Tier 1 数据集上复算")
    ap.add_argument("--gate-rad", type=float, default=1.5,
                    help="评测器的物体朝向门槛（metrics JSON 的 object_orientation_threshold_rad）")
    a = ap.parse_args()
    evaluator_baseline(embedded_config(a.policy), a.gate_rad)
    if a.tier1_episode is not None:
        tier1_baseline(a.tier1_episode, a.gate_rad)


if __name__ == "__main__":
    main()
