import pathlib
"""Tier2 vs Tier1 object-trajectory error, per episode and aggregate.
This is the 'before denoising' anchor: how much of the MP-SR budget the
corruption alone consumes, and what ADD-AUC the raw Tier-2 reference scores.
"""
import sys, os, pathlib, json, glob
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "libs" / "v2d_infra"))
import numpy as np, pyarrow.parquet as pq
import add_metrics as A

ROOT = str(pathlib.Path(__file__).resolve().parents[2] / "data/challenge/track_2")
T1, T2 = ROOT + "/tier_1_multiview_caption", ROOT + "/tier_2_synthetic_noise"

def col(t, c):
    v = t.column(c).to_numpy(zero_copy_only=False)
    return np.stack(v) if v.dtype == object else np.asarray(v)

meta = {json.loads(l)["episode_index"]: json.loads(l) for l in open(T1 + "/meta/episodes_metadata.jsonl")}
rows = []
for f1 in sorted(glob.glob(T1 + "/data/chunk-000/*.parquet")):
    f2 = f1.replace("tier_1_multiview_caption", "tier_2_synthetic_noise")
    t1, t2 = pq.read_table(f1), pq.read_table(f2)
    ei = int(np.asarray(col(t1, "episode_index")).ravel()[0])
    vis = np.asarray(col(t1, "observation.object.visible")).astype(bool).ravel()
    p1, p2 = col(t1, "observation.object.pose")[vis], col(t2, "observation.object.pose")[vis]
    verts = A.sample_mesh_vertices(os.path.join(T1, meta[ei]["mesh"]), n=500, seed=ei)
    add = A.add_per_frame(verts, p1, p2)
    auc, _ = A.auc_from_add(add)
    dt, dq = A.pose_errors(p1, p2)
    rows.append(dict(ep=ei, obj=meta[ei]["object"], nv=len(verts), T=len(p1),
                     auc=auc, add_mm=1000*add.mean(), add_p95=1000*np.percentile(add, 95),
                     dt_mm=1000*dt.mean(), dq_deg=np.degrees(dq).mean()))

rows.sort(key=lambda r: r["auc"])
print(f"{'ep':>3} {'物体':<21} {'AUC':>6} {'ADD均值':>8} {'ADD p95':>8} {'平移mm':>7} {'旋转°':>6}")
for r in rows:
    print(f"{r['ep']:>3} {r['obj']:<21} {r['auc']:>6.3f} {r['add_mm']:>7.1f}mm {r['add_p95']:>7.1f}mm {r['dt_mm']:>7.1f} {r['dq_deg']:>6.1f}")
a = np.array([r["auc"] for r in rows]); d = np.array([r["add_mm"] for r in rows])
print(f"\n=== 汇总 (30 episodes) ===")
print(f"ADD-AUC      : mean {a.mean():.3f}  median {np.median(a):.3f}  min {a.min():.3f}  max {a.max():.3f}")
print(f"ADD 均值     : mean {d.mean():.1f} mm  median {np.median(d):.1f} mm  max {d.max():.1f} mm")
print(f"物体平移误差 : {np.mean([r['dt_mm'] for r in rows]):.1f} mm   (MP-SR 阈值 30mm)")
print(f"物体旋转误差 : {np.mean([r['dq_deg'] for r in rows]):.1f}°    (MP-SR 阈值 30°)")
json.dump(rows, open(str(pathlib.Path(__file__).resolve().parents[2] / "results/baseline_t2.json"), "w"), indent=1, default=float)
