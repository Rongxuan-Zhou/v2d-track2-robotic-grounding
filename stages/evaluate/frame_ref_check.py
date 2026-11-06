"""打包器的 Track 2 坐标换算能不能过，先在 CPU 上核（训练机宿主机上跑，要 numpy、pyarrow）。计划 agent-doc/plan-box6-and-eval-runtime-2026-10-01.md 第 3.4 节。

kit 的 tools/pack_submission.py 对每条 Track 2 episode 做 convert_track2_frame（78–130 行）：在计分帧上取评测器记下的参考位姿与“这一档的
输入参考”，用第一个计分帧求一个刚体变换，所有计分帧的残差都要在 1 mm、1e-3 rad 以内，否则整份打包报错。评测器记下的参考，就是这条 run
训练用的 motion parquet 里第一个物体刚体的位姿（导出后由 check_export.py 的“评测器参考 vs motion parquet”一行复核）。所以不用 GPU，
就能对全部 episode 先算残差：
  Tier 1：输入参考 = 发布的 Tier 1 数据（data/challenge/track_2/tier_1_multiview_caption，observation.object.pose，wxyz）；
  Tier 2：输入参考 = 我们要给 --frame-references 的 v5.4（runs/motion_src/tier2/<序列>/v2d_meta.json 的 object_pose_source，
          object_pose 是 OpenCV 世界系、wxyz，与发布数据同一约定）。进程开头 forbid_tier1()，只读 Tier 2 的文件。
计分帧 = 发布数据里 observation.object.visible 为真的帧（kit 的 v2dlb/trajectory.py 只保留这些帧）；Tier 2 只读 Tier 2 的可见性。
算法照 kit 原样用 numpy 重写（_quaternion_multiply、_rotation_matrix、_normalise_quaternions、两项容差都是 1e-3）：服务器上没有 pandas，
导入不了打包器。用 kit 原函数的核对在 Mac 上做（计划 E6、E7）。
用法（仓库根）：python3 stages/evaluate/frame_ref_check.py --tier 1|2 [--episodes 0 1 …]   默认 runs/motion_src/tier<档位> 下全部 episode
  逐条打残差（位置 mm、角度 mrad）与 ok / FAIL，最后一行 FRAME_REF_OK 或 FRAME_REF_FAIL n。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/challenge/track_2"
TIER_DIR = {1: "tier_1_multiview_caption", 2: "tier_2_synthetic_noise"}
POS_TOL = ANG_TOL = 1e-3   # 同 kit 的 _TRACK2_POSITION_TOLERANCE_M、_TRACK2_ANGLE_TOLERANCE_RAD


def qmul(a, b):
    ax, ay, az, aw = (a[..., i] for i in range(4))
    bx, by, bz, bw = (b[..., i] for i in range(4))
    return np.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], axis=-1)


def rotmat(q):
    x, y, z, w = (q[..., i] for i in range(4))
    return np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                     2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                     2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], axis=-1).reshape(q.shape[:-1] + (3, 3))


def unit(pose):
    pose = pose.copy()
    n = np.linalg.norm(pose[..., 3:], axis=-1)
    if np.any(n < 1e-8):
        raise ValueError("零四元数")
    pose[..., 3:] /= n[..., None]
    return pose


def wxyz_to_xyzw(pose):
    return np.concatenate([pose[..., :3], pose[..., 4:7], pose[..., 3:4]], axis=-1)


def residual(evaluator, released):
    """kit convert_track2_frame 的核对部分：返回（位置残差最大值 m，角度残差最大值 rad）。"""
    evaluator, released = unit(evaluator), unit(released)
    e0, r0 = evaluator[0], released[0]
    rq = qmul(r0[None, 3:], (e0[3:] * np.array([-1.0, -1.0, -1.0, 1.0]))[None])[0]
    rq /= np.linalg.norm(rq)
    R = rotmat(rq[None])[0]
    off = r0[:3] - R @ e0[:3]
    pos = evaluator[:, :3] @ R.T + off
    quat = qmul(np.broadcast_to(rq, evaluator[:, 3:].shape), evaluator[:, 3:])
    dp = float(np.linalg.norm(pos - released[:, :3], axis=1).max())
    dot = np.abs(np.sum(quat * released[:, 3:], axis=1))
    return dp, float((2.0 * np.arccos(np.clip(dot, 0.0, 1.0))).max())


def motion_pose(tier: int, seq: str):
    t = pq.read_table(ROOT / f"runs/motion/tier{tier}/whole_body/soma/sequence_id={seq}/robot_name=g1/data.parquet",
                      columns=["object_body_position", "object_body_wxyz"])
    pos = np.asarray(t.column("object_body_position")[0].as_py(), dtype=np.float64)
    wxyz = np.asarray(t.column("object_body_wxyz")[0].as_py(), dtype=np.float64)
    pos, wxyz = pos.reshape(len(pos), -1, 3)[:, 0], wxyz.reshape(len(wxyz), -1, 4)[:, 0]
    return np.concatenate([pos, wxyz[:, 1:], wxyz[:, :1]], axis=1)


def released(tier: int, ep: int):
    f = DATA / TIER_DIR[tier] / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
    t = pq.read_table(f, columns=["observation.object.pose", "observation.object.visible"])
    pose = np.asarray(t.column("observation.object.pose").combine_chunks().flatten().to_numpy(), dtype=np.float64).reshape(-1, 7)
    vis = np.asarray(t.column("observation.object.visible").to_numpy(zero_copy_only=False), dtype=bool).reshape(-1)
    return pose, vis


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, choices=(1, 2), required=True)
    ap.add_argument("--episodes", type=int, nargs="*")
    a = ap.parse_args()
    if a.tier == 2:
        sys.path.insert(0, str(ROOT / "libs/v2d_infra"))
        from tier_guard import forbid_tier1
        forbid_tier1()   # 档位隔离：此后读 Tier 1 的任何文件都会抛错
    metas = {}
    for f in sorted((ROOT / f"runs/motion_src/tier{a.tier}").glob("*/v2d_meta.json")):
        m = json.loads(f.read_text())
        if str(m.get("tier")) == str(a.tier):
            metas[int(m["episode"])] = m
    eps = a.episodes if a.episodes else sorted(metas)
    print(f"== Tier {a.tier}：{len(eps)} 条 episode；容差位置 {POS_TOL} m、角度 {ANG_TOL} rad（同 kit）")
    bad = 0
    for ep in eps:
        m = metas.get(ep)
        if m is None:
            print(f"  FAIL ep{ep}：没有 runs/motion_src/tier{a.tier}/*/v2d_meta.json"); bad += 1; continue
        seq = m["sequence_id"]
        try:
            ev = motion_pose(a.tier, seq)
            rel, vis = released(a.tier, ep)
            if a.tier == 2:
                src = m["object_pose_source"]["path"]
                with np.load(ROOT / src, allow_pickle=False) as z:
                    rel = np.asarray(z["object_pose"], dtype=np.float64)
            if not (len(ev) == len(rel) == len(vis)):
                raise ValueError(f"帧数不一致：motion {len(ev)}，输入 {len(rel)}，可见性 {len(vis)}")
            dp, da = residual(ev[vis], wxyz_to_xyzw(rel)[vis])
        except Exception as e:   # noqa: BLE001  逐条报，不让一条的问题挡住其他条
            print(f"  FAIL ep{ep} {seq}：{type(e).__name__}: {e}"); bad += 1; continue
        ok = dp <= POS_TOL and da <= ANG_TOL
        bad += not ok
        print(f"  {'ok  ' if ok else 'FAIL'} ep{ep:<2} {seq}：计分帧 {int(vis.sum())} / {len(vis)}，位置残差 {dp * 1000:.4f} mm，角度残差 {da * 1000:.4f} mrad")
    print("FRAME_REF_OK" if bad == 0 else f"FRAME_REF_FAIL {bad}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
