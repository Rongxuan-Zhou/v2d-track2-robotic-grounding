import pathlib
"""ep29 行李箱：双手接触为 0 是数据如此，还是阈值太小？

序列名是 rolling_luggage_left_foot_push_06（用左脚推）。先量清楚手和脚各自离物体多近、
近的时候物体是不是在动，再决定要不要调 --contact-threshold。
只做分析，读 Tier 1（Tier 1 管线自身的数据，不进 Tier 2/3）。
"""
import os, sys, json, glob, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "libs" / "v2d_infra"))
from soma_fk import fk, joint_names, R_CV
import numpy as np, pyarrow.parquet as pq, add_metrics as A

ROOT = pathlib.Path(__file__).resolve().parents[2] / "data/challenge/track_2"
TIERS = {1: ROOT / "tier_1_multiview_caption", 2: ROOT / "tier_2_synthetic_noise"}
EP = 29


def col(t, c):
    v = t.column(c).to_numpy(zero_copy_only=False)
    return np.stack(v) if v.dtype == object else np.asarray(v)


def to_soma(p):
    q = p.copy(); q[:, :3] = p[:, :3] @ R_CV.T
    M = np.einsum("ij,tjk->tik", R_CV, A.quat_wxyz_to_matrix(p[:, 3:7]))
    w = np.sqrt(np.clip(1 + M[:, 0, 0] + M[:, 1, 1] + M[:, 2, 2], 1e-12, None)) / 2
    q[:, 3] = w; q[:, 4] = (M[:, 2, 1] - M[:, 1, 2]) / (4 * w)
    q[:, 5] = (M[:, 0, 2] - M[:, 2, 0]) / (4 * w); q[:, 6] = (M[:, 1, 0] - M[:, 0, 1]) / (4 * w)
    return q


names = joint_names()
GROUPS = {
    "左手": [i for i, n in enumerate(names) if n.startswith("LeftHand") or (n.startswith("Left") and "Hand" in n)],
    "右手": [i for i, n in enumerate(names) if n.startswith("RightHand") or (n.startswith("Right") and "Hand" in n)],
    "左脚": [i for i, n in enumerate(names) if n.startswith("Left") and ("Foot" in n or "Toe" in n)],
    "右脚": [i for i, n in enumerate(names) if n.startswith("Right") and ("Foot" in n or "Toe" in n)],
}
print("关节分组：", {k: [names[i] for i in v][:6] + (["…"] if len(v) > 6 else []) for k, v in GROUPS.items()})

for tier, root in TIERS.items():
    meta = {json.loads(l)["episode_index"]: json.loads(l) for l in open(root / "meta/episodes_metadata.jsonl")}
    f = sorted(glob.glob(str(root / "data/chunk-000/*.parquet")))[EP]
    t = pq.read_table(f)
    assert int(np.asarray(col(t, "episode_index")).ravel()[0]) == EP
    J = fk(t, flex=True)                                   # (T,77,3) SOMA 系
    p = to_soma(col(t, "observation.object.pose"))
    v = A.sample_mesh_vertices(os.path.join(root, meta[EP]["mesh"]), n=500, seed=EP)
    ov = A.transform_vertices(v, p)                       # (T,500,3)
    step = np.r_[0, np.linalg.norm(np.diff(p[:, :3], axis=0), axis=1)]
    moving = step > 0.0015                                # 1.5 mm/帧，与 contact_veto 一致
    print(f"\n== Tier {tier} · ep{EP} {meta[EP].get('sequence_id', '')} · {len(t)} 帧 · 物体总位移 "
          f"{np.linalg.norm(p[-1, :3] - p[0, :3]):.2f} m · 运动帧 {moving.mean() * 100:.0f}%")
    print(f"{'部位':<4} {'最近p5':>7} {'中位':>7} {'<2cm帧':>7} {'<5cm帧':>7} {'<10cm帧':>8} {'运动帧中<5cm':>12}")
    for g, idx in GROUPS.items():
        d = np.linalg.norm(J[:, idx, None, :] - ov[:, None, :, :], axis=-1).min(axis=(1, 2))
        near5 = d < 0.05
        mv = (near5 & moving).sum() / max(moving.sum(), 1)
        print(f"{g:<4} {np.percentile(d, 5):>6.3f}m {np.median(d):>6.3f}m {(d < 0.02).sum():>7d} "
              f"{near5.sum():>7d} {(d < 0.10).sum():>8d} {mv * 100:>11.0f}%")
