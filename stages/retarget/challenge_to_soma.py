"""挑战数据（LeRobot v2.1，一个档位）→ 官方 soma_to_g1 的输入目录。

输出两处，都按档位分开，不混档：
  runs/motion_src/tier{N}/<sequence_id>/            soma_to_g1 的 data_folder
      soma_params.npz                               人体参数，含逐帧 bone_length_flexibles
      poses.npy                                     物体 (T,4,4) float64，OpenCV 世界系，不做变换
      reconstructed_mesh/output_aligned.glb         物体网格（官方读取器写死的路径）
      ground_plane.json                             原样复制
      v2d_meta.json                                 档位/episode/源帧率；源帧率由 timestamp 算出，
                                                    run_soma_to_g1.py 用它修正输出里错写的 fps
  runs/motion/tier{N}/whole_body/reconstructed_stage/<sequence_id>_support.usda
      scene_config._discover_support_surface 按这个路径找；找不到时静默返回 None，
      场景里就少了桌子而不报错，所以这里负责放好，下游必须断言它被找到。

坐标：poses.npy 保持 OpenCV 世界系——官方 load_data 自己左乘 diag(1,-1,-1,1)，
这里再转一次就错了。
物体不可见帧：沿用上一有效位姿；开头若不可见，用第一个有效位姿回填。
（全数据集只有 1 帧不可见，官方 soma_to_g1 完全不处理 visible。）

档位隔离：只经 paths.tier(N) 读一个档位的根目录，不读其他档位的任何文件。命令行入口（main）对 --tier 2、3
先调 tier_guard.forbid_tier1()，进程此后读不到 Tier 1 的任何文件（计划 agent-doc/plan-t3-delivery-p2-2026-10-12.md 审查 12）；
convert() 本身不调，因为守卫装上就撤不掉，测试要在同一进程里转 Tier 1。
Tier 3 设了 V2D_T3_SET=<名> 时，读 runs/tier3/root_<名>，上面两处的 tier3 换成 tier3_<名>（paths.tier_dirname；
计划 agent-doc/plan-t3-pilot-v2-2026-10-10.md），不覆盖旧版的文件。

物体位姿来源：默认取该档位 parquet 的 observation.object.pose；传 --object-poses 时改用
recovery 阶段落盘的 npz（object_pose (T,7)，OpenCV 系，已补齐不可见帧）。环节间只经文件交接，
不 import recovery 的代码。来源记进 v2d_meta.json。

support：v0.4.0 的 soma_to_g1 先做地面平面找平（地面 → z=0），挑战原版 support 与之对齐
（ep20 实测间隙 −1.3 mm），直接用挑战给的文件，不再重建。

用法：python stages/retarget/challenge_to_soma.py --tier 1 --episode 20
      python stages/retarget/challenge_to_soma.py --tier 2 --episode 20 --object-poses runs/recovery/tier2_v52/episode_000020.npz
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "libs" / "v2d_infra"))
import numpy as np
import pyarrow.parquet as pq
from paths import RUNS, tier, tier_dirname
from soma_fk import col, joint_names
from tier_guard import forbid_tier1

FLEX = "observation.human.bone_length_flexibles"


def pose7_to_mat(p7: np.ndarray) -> np.ndarray:
    """(T,7) [x,y,z,qw,qx,qy,qz] → (T,4,4) float64 齐次矩阵。"""
    t = p7[:, :3].astype(np.float64)
    q = p7[:, 3:7].astype(np.float64)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    R = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)
    M = np.tile(np.eye(4), (len(p7), 1, 1))
    M[:, :3, :3] = R
    M[:, :3, 3] = t
    return M


def fill_invisible(p7: np.ndarray, visible: np.ndarray) -> tuple[np.ndarray, int]:
    if not visible.any():
        raise ValueError("整条序列物体都不可见")
    out = p7.copy()
    first = int(np.argmax(visible))
    out[:first] = p7[first]
    last = first
    for i in range(first, len(p7)):
        if visible[i]:
            last = i
        else:
            out[i] = p7[last]
    return out, int((~visible).sum())


def tier3_provenance(root: Path, episode: int) -> dict:
    """Tier 3 的档目录是我们拼的（stages/recovery/tier3_root.py）：这一集的来源（重建的部署版本、各文件哈希）在 meta/provenance.json，
    转换时原样写进 v2d_meta.json，让每条 run 都能指回重建版本（计划 agent-doc/plan-t3-pilot-2026-10-09.md，审查 m5）。"""
    p = root / "meta" / "provenance.json"
    if not p.is_file():
        raise SystemExit(f"challenge_to_soma: Tier 3 的档目录缺 {p}（由 stages/recovery/tier3_root.py 写）")
    eps = json.loads(p.read_text())["episodes"]
    if str(episode) not in eps:
        raise SystemExit(f"challenge_to_soma: {p} 里没有 episode {episode}")
    return eps[str(episode)]


def convert(tier_n: int, episode: int, runs_root: Path = RUNS, object_poses: Path | None = None) -> dict:
    root = tier(tier_n)
    prov = tier3_provenance(root, episode) if tier_n == 3 else None
    meta = {json.loads(l)["episode_index"]: json.loads(l)
            for l in open(root / "meta" / "episodes_metadata.jsonl")}[episode]
    t = pq.read_table(root / "data" / "chunk-000" / f"episode_{episode:06d}.parquet")
    T = t.num_rows
    assert T == meta["frames"], f"parquet {T} 帧，metadata 记 {meta['frames']} 帧"
    seq = meta["sequence_id"]
    ts = np.asarray(col(t, "timestamp"), dtype=np.float64).ravel()
    source_fps = float(round(1.0 / np.median(np.diff(ts)), 3))

    src = runs_root / "motion_src" / tier_dirname(tier_n) / seq
    (src / "reconstructed_mesh").mkdir(parents=True, exist_ok=True)

    np.savez(
        src / "soma_params.npz",
        poses=col(t, "observation.human.pose").reshape(T, 77, 3).astype(np.float32),
        transl=col(t, "observation.human.translation").astype(np.float32),
        identity_coeffs=col(t, "observation.human.identity_coeffs").astype(np.float32),
        scale_params=col(t, "observation.human.scale_params").astype(np.float32),
        bone_length_flexibles=col(t, FLEX).astype(np.float32),   # 官方读取器不认，靠 SOMAWithFlex 注入
        joint_names=np.array(joint_names()),
        identity_model_type=np.array("mhr"),
        unit=np.array("meters"),
        keep_root=np.array(False),
        joint_orient=np.zeros((78, 3, 3), np.float32),   # 必填，但 _reconstruct 直接 del 掉
    )

    vis = np.asarray(col(t, "observation.object.visible")).astype(bool).ravel()
    if object_poses is None:
        p7, n_filled = fill_invisible(col(t, "observation.object.pose"), vis)
        pose_src = {"kind": "parquet", "column": "observation.object.pose"}
        if prov is not None:
            pose_src = {"kind": "tier3_reconstruction", "column": "observation.object.pose", "provenance": prov}
    else:
        z = np.load(object_poses)
        p7 = np.asarray(z["object_pose"], dtype=np.float64); n_filled = int((~z["visible"]).sum())
        assert p7.shape == (T, 7), f"{object_poses} 形状 {p7.shape}，期望 ({T}, 7)"
        pose_src = {"kind": "recovery", "path": str(object_poses), "meta": json.loads(str(z["meta"]))}
    np.save(src / "poses.npy", pose7_to_mat(p7))

    shutil.copyfile(root / meta["mesh"], src / "reconstructed_mesh" / "output_aligned.glb")
    shutil.copyfile(root / meta["ground_plane"], src / "ground_plane.json")

    (src / "v2d_meta.json").write_text(json.dumps(
        {"tier": tier_n, "episode": episode, "sequence_id": seq, "source_fps": source_fps,
         "object_pose_source": pose_src},
        ensure_ascii=False, indent=2))

    motion_root = runs_root / "motion" / tier_dirname(tier_n)
    stage = motion_root / "whole_body" / "reconstructed_stage"
    stage.mkdir(parents=True, exist_ok=True)
    support = stage / f"{seq}_support.usda"
    shutil.copyfile(root / meta["support_surface"], support)

    return {"tier": tier_n, "episode": episode, "sequence_id": seq, "frames": T, "source_fps": source_fps,
            "invisible_filled": n_filled, "data_folder": str(src),
            "motion_root": str(motion_root), "support": str(support),
            "support_surface_frame": meta["support_surface_frame"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, required=True)
    ap.add_argument("--episode", type=int, required=True)
    ap.add_argument("--object-poses", type=Path, default=None)
    a = ap.parse_args(argv)
    if a.tier in (2, 3):
        forbid_tier1()   # 档位隔离的机制守卫：本进程此后任何读取 Tier 1 的动作都会抛错
    print(json.dumps(convert(a.tier, a.episode, object_poses=a.object_poses), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
