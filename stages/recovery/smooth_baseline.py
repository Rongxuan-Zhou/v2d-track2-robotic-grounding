import pathlib
"""How much AUC does pure temporal smoothing buy, with no anchor?
Random-walk noise has white increments, so smoothing removes the high-frequency
half but leaves the accumulated drift. This measures where that ceiling sits.
"""
import sys,os,glob,json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "libs" / "v2d_infra"))
import numpy as np, pyarrow.parquet as pq, add_metrics as A
from scipy.signal import savgol_filter
ROOT=str(pathlib.Path(__file__).resolve().parents[2] / "data/challenge/track_2"); T1=ROOT+"/tier_1_multiview_caption"
def col(t,c):
    v=t.column(c).to_numpy(zero_copy_only=False)
    return np.stack(v) if v.dtype==object else np.asarray(v)
meta={json.loads(l)["episode_index"]:json.loads(l) for l in open(T1+"/meta/episodes_metadata.jsonl")}

def smooth_pose(p, win, poly=2):
    """Savitzky-Golay on translation; quaternion smoothed after sign alignment."""
    q=p[:,3:7].copy()
    for i in range(1,len(q)):                      # 解决双重覆盖的符号翻转
        if np.dot(q[i],q[i-1])<0: q[i]=-q[i]
    w=min(win, len(p)-(1-len(p)%2))
    if w<poly+2: return p.copy()
    if w%2==0: w-=1
    out=p.copy()
    out[:,:3]=savgol_filter(p[:,:3],w,poly,axis=0)
    qs=savgol_filter(q,w,poly,axis=0)
    out[:,3:7]=qs/np.linalg.norm(qs,axis=1,keepdims=True)
    return out

wins=[5,11,21,41,81,161]
acc={w:[] for w in wins}; raw=[]
for f1 in sorted(glob.glob(T1+"/data/chunk-000/*.parquet")):
    f2=f1.replace("tier_1_multiview_caption","tier_2_synthetic_noise")
    t1,t2=pq.read_table(f1),pq.read_table(f2)
    ei=int(np.asarray(col(t1,"episode_index")).ravel()[0])
    vis=np.asarray(col(t1,"observation.object.visible")).astype(bool).ravel()
    p1,p2=col(t1,"observation.object.pose")[vis],col(t2,"observation.object.pose")[vis]
    v=A.sample_mesh_vertices(os.path.join(T1,meta[ei]["mesh"]),n=500,seed=ei)
    raw.append(A.auc_from_add(A.add_per_frame(v,p1,p2))[0])
    for w in wins:
        acc[w].append(A.auc_from_add(A.add_per_frame(v,p1,smooth_pose(p2,w)))[0])
print(f"{'窗口(帧)':>9} {'AUC mean':>9} {'中位数':>8} {'相对原始':>9}")
r=np.mean(raw); print(f"{'原始':>9} {r:>9.3f} {np.median(raw):>8.3f} {'—':>9}")
for w in wins:
    a=np.array(acc[w]); print(f"{w:>9} {a.mean():>9.3f} {np.median(a):>8.3f} {a.mean()-r:>+9.3f}")
print(f"\n参考: 只修旋转的上界 0.881 | CHORD 论文 0.918")
