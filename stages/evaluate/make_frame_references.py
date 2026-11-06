"""生成 Tier 2 打包用的 --frame-references（训练机宿主机上跑，要 numpy、pyarrow；只读 Tier 2 的文件）。计划 agent-doc/plan-submission-exports-2026-10-02.md 第 3.3 节。

kit 的 tracks/track_2/README.md：输入不是发布的那份时，打包要加 --frame-references，给每条 episode 一个 episode_XXXXXX.npz，含 episode_index、
coordinate_frame=released_track2_world、frame_index [T]、pose_xyzw [T,1,7]（米，xyzw）。我们 Tier 2 的 policy 用 v5.4 降噪后的参考训练，
所以帧参考就是 v5.4：
  - 来源：runs/motion_src/tier2/<序列>/v2d_meta.json 的 object_pose_source.path（训练用的 motion parquet 就是从它重定向出来的，P5′ 已核残差）；
  - v5.4 的 object_pose 是 [x,y,z,qw,qx,qy,qz]，OpenCV 世界系，与发布数据同一约定；这里换成 xyzw，形状 [T,1,7]；
  - frame_index 取发布的 Tier 2 parquet 的 frame_index 列，要求与 v5.4 帧数相同、严格递增，episode_index 列全等于这条 episode。
同一世界系与四元数布局的粗核（计划第 3.3 节，按审查意见 m6）：v5.4 与发布的 Tier 2 带噪输入第 0 帧位置差 ≤ 20 mm（Tier 2 噪声从第 0 帧的约 0 开始；
容差同 stages/recovery/check_tier2_export.py 对 v5.4 的 F0_TOL，ep17 被不穿地约束抬了 13.5 mm），逐帧角度差的中位数 < 45°（噪声上限 30°；
四元数布局弄错时角度差会大得多）。逐帧位置差的中位数只报不卡。换错世界系（例如 SOMA 与 OpenCV 差一个 diag(1,-1,-1)）第 0 帧就会差出米级。
进程开头 forbid_tier1()：此后读 Tier 1 的任何文件都会抛错。
输出 runs/submission/frame_refs_t2/episode_XXXXXX.npz 与 manifest.tsv（episode、序列、帧数、来源 npz 的 sha256、输出的 sha256）；
输出目录已存在就拒绝。最后一行 FRAME_REFS_OK 30 或 FRAME_REFS_FAIL n。
用法（仓库根）：python3 stages/evaluate/make_frame_references.py [--out runs/submission/frame_refs_t2] [--episodes …]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
RELEASED = ROOT / "data/challenge/track_2/tier_2_synthetic_noise/data/chunk-000"
FRAME = "released_track2_world"
F0_TOL, ANG_MED_TOL = 0.020, 45.0   # 米、度；见文件头


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def angle_deg(a_xyzw, b_xyzw):
    dot = np.abs(np.sum(a_xyzw * b_xyzw, axis=-1)) / (np.linalg.norm(a_xyzw, axis=-1) * np.linalg.norm(b_xyzw, axis=-1))
    return np.degrees(2 * np.arccos(np.clip(dot, 0.0, 1.0)))


def one(ep: int, meta: dict, out: Path) -> tuple:
    src = ROOT / meta["object_pose_source"]["path"]
    with np.load(src, allow_pickle=False) as z:
        pose = np.asarray(z["object_pose"], dtype=np.float64)
    t = pq.read_table(RELEASED / f"episode_{ep:06d}.parquet",
                      columns=["frame_index", "episode_index", "observation.object.pose", "observation.object.visible"])
    fi = np.asarray(t.column("frame_index").to_numpy(), dtype=np.int64)
    epi = np.asarray(t.column("episode_index").to_numpy(), dtype=np.int64)
    noisy = np.asarray(t.column("observation.object.pose").combine_chunks().flatten().to_numpy(), dtype=np.float64).reshape(-1, 7)
    vis = np.asarray(t.column("observation.object.visible").to_numpy(zero_copy_only=False), dtype=bool).reshape(-1)
    if pose.ndim != 2 or pose.shape[1] != 7 or len(pose) != len(fi):
        raise ValueError(f"v5.4 形状 {pose.shape}，发布数据 {len(fi)} 帧")
    if not (np.all(epi == ep) and np.all(np.diff(fi) > 0)):
        raise ValueError("发布数据的 episode_index 不全等于这条 episode，或 frame_index 不严格递增")
    if not np.isfinite(pose).all() or np.any(np.linalg.norm(pose[:, 3:], axis=1) < 1e-8):
        raise ValueError("v5.4 有非有限值或零四元数")
    xyzw = np.concatenate([pose[:, :3], pose[:, 4:7], pose[:, 3:4]], axis=1)
    noisy_xyzw = np.concatenate([noisy[:, :3], noisy[:, 4:7], noisy[:, 3:4]], axis=1)
    # 粗核只比发布输入可见、四元数非零的帧（ep11 有一帧不可见，发布输入那一帧的四元数是零）；v5.4 本身每帧都已核过有限、非零
    use = vis & (np.linalg.norm(noisy[:, 3:], axis=1) > 1e-8)
    if not use[0]:
        raise ValueError("发布的 Tier 2 输入第 0 帧不可见或四元数为零，核不了第 0 帧")
    dpos = np.linalg.norm(xyzw[:, :3] - noisy_xyzw[:, :3], axis=1)
    dang = np.full(len(fi), np.nan); dang[use] = angle_deg(xyzw[use, 3:], noisy_xyzw[use, 3:])
    if not (dpos[0] <= F0_TOL and np.median(dang[use]) < ANG_MED_TOL):
        raise ValueError(f"v5.4 与发布的 Tier 2 输入第 0 帧位置差 {dpos[0] * 1000:.1f} mm（要 ≤ {F0_TOL * 1000:.0f} mm）、"
                         f"角度差中位数 {np.median(dang[use]):.1f}°（要 < {ANG_MED_TOL:.0f}°），像是世界系或四元数布局不对")
    path = out / f"episode_{ep:06d}.npz"
    np.savez(path, episode_index=np.int64(ep), coordinate_frame=np.array(FRAME), frame_index=fi, pose_xyzw=xyzw[:, None, :])
    with np.load(path, allow_pickle=False) as z:   # 照 kit 的 input_frame_reference 读回来核一遍
        assert str(z["coordinate_frame"].item()) == FRAME and int(z["episode_index"].item()) == ep
        assert z["pose_xyzw"].shape == (len(fi), 1, 7) and np.array_equal(z["frame_index"], fi)
    return (len(fi), sha256(src), sha256(path), float(np.median(dpos[use]) * 1000), float(np.median(dang[use])), float(dpos[0] * 1000), float(dang[0]),
            int((~use).sum()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ROOT / "runs/submission/frame_refs_t2")
    ap.add_argument("--episodes", type=int, nargs="*", help="只做这些 episode（测试用；默认 0..29 全部）")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "libs/v2d_infra"))
    from tier_guard import forbid_tier1
    forbid_tier1()   # 档位隔离：此后读 Tier 1 的任何文件都会抛错
    if a.out.exists():
        sys.exit(f"{a.out} 已存在，拒绝覆盖")
    metas = {}
    for f in sorted((ROOT / "runs/motion_src/tier2").glob("*/v2d_meta.json")):
        m = json.loads(f.read_text())
        if str(m.get("tier")) == "2":
            metas.setdefault(int(m["episode"]), []).append(m)
    a.out.mkdir(parents=True)
    rows, bad = [], 0
    print(f"== Tier 2 帧参考：{len(metas)} 条 episode → {a.out}（{FRAME}，xyzw，[T,1,7]）")
    for ep in (a.episodes if a.episodes else range(30)):
        ms = metas.get(ep, [])
        if len(ms) != 1 or ms[0].get("object_pose_source", {}).get("meta", {}).get("denoiser") != "v5.4":
            print(f"  FAIL ep{ep}：v2d_meta 有 {len(ms)} 份，或来源不是 v5.4"); bad += 1; continue
        try:
            n, src_sha, out_sha, mp, ma, p0, a0, skip = one(ep, ms[0], a.out)
        except Exception as e:   # noqa: BLE001  逐条报
            print(f"  FAIL ep{ep} {ms[0]['sequence_id']}：{type(e).__name__}: {e}"); bad += 1; continue
        rows.append(f"{ep}\t{ms[0]['sequence_id']}\t{n}\t{ms[0]['object_pose_source']['path']}\t{src_sha}\tepisode_{ep:06d}.npz\t{out_sha}\n")
        print(f"  ok   ep{ep:<2} {ms[0]['sequence_id']}：{n} 帧；与发布的带噪输入相比位置差中位 {mp:.1f} mm、角度差中位 {ma:.1f}°，第 0 帧 {p0:.1f} mm、{a0:.1f}°"
              + (f"（粗核排除 {skip} 帧：发布输入不可见或四元数为零）" if skip else ""))
    (a.out / "manifest.tsv").write_text("episode\tsequence\tframes\tsource\tsource_sha256\tfile\tsha256\n" + "".join(rows), encoding="utf-8")
    print(f"FRAME_REFS_OK {len(rows)}" if bad == 0 else f"FRAME_REFS_FAIL {bad}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
