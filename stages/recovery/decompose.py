import pathlib
"""Which half of the corruption costs the AUC: translation or rotation?
Builds two hybrid trajectories from Tier1/Tier2 and scores each.
  trans_only  = Tier2 translation + Tier1 rotation   (rotation perfectly fixed)
  rot_only    = Tier1 translation + Tier2 rotation   (translation perfectly fixed)
"""
import sys, os, pathlib, json, glob
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "libs" / "v2d_infra"))
import numpy as np, pyarrow.parquet as pq
import add_metrics as A

ROOT = str(pathlib.Path(__file__).resolve().parents[2] / "data/challenge/track_2")
T1 = ROOT + "/tier_1_multiview_caption"
def col(t,c):
    v=t.column(c).to_numpy(zero_copy_only=False)
    return np.stack(v) if v.dtype==object else np.asarray(v)
meta={json.loads(l)["episode_index"]:json.loads(l) for l in open(T1+"/meta/episodes_metadata.jsonl")}

res={"raw":[], "trans_only":[], "rot_only":[]}
sizes=[]
for f1 in sorted(glob.glob(T1+"/data/chunk-000/*.parquet")):
    f2=f1.replace("tier_1_multiview_caption","tier_2_synthetic_noise")
    t1,t2=pq.read_table(f1),pq.read_table(f2)
    ei=int(np.asarray(col(t1,"episode_index")).ravel()[0])
    vis=np.asarray(col(t1,"observation.object.visible")).astype(bool).ravel()
    p1,p2=col(t1,"observation.object.pose")[vis],col(t2,"observation.object.pose")[vis]
    verts=A.sample_mesh_vertices(os.path.join(T1,meta[ei]["mesh"]),n=500,seed=ei)
    radius=float(np.linalg.norm(verts-verts.mean(0),axis=1).max())
    sizes.append((ei,meta[ei]["object"],radius))
    hyb_t=p2.copy(); hyb_t[:,3:7]=p1[:,3:7]      # 只剩平移噪声
    hyb_r=p2.copy(); hyb_r[:,:3]=p1[:,:3]        # 只剩旋转噪声
    for k,p in (("raw",p2),("trans_only",hyb_t),("rot_only",hyb_r)):
        res[k].append(A.auc_from_add(A.add_per_frame(verts,p1,p))[0])

print("=== AUC 分解 (30 episodes 平均) ===")
for k in ("raw","trans_only","rot_only"):
    a=np.array(res[k]); print(f"  {k:12s} AUC mean {a.mean():.3f}  median {np.median(a):.3f}")
print("\n解读: trans_only = 旋转完美修复后能达到的 AUC；rot_only = 平移完美修复后的 AUC")
print("\n=== 物体尺寸 vs AUC 相关性 ===")
r=np.array([s[2] for s in sizes]); a=np.array(res["raw"])
print(f"  物体半径 mean {r.mean():.3f} m  range {r.min():.3f}-{r.max():.3f} m")
print(f"  Pearson(半径, AUC) = {np.corrcoef(r,a)[0,1]:.3f}")
print("\n  最大的 5 个物体:")
for ei,nm,rad in sorted(sizes,key=lambda s:-s[2])[:5]:
    print(f"    {nm:<20} 半径 {rad:.2f} m  AUC {res['raw'][[s[0] for s in sizes].index(ei)]:.3f}")
