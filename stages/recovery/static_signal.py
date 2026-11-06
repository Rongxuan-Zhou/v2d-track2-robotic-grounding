import pathlib
"""Is 'object is static' an observable handle on the Tier-2 drift?

If the object is truly motionless over a window, any apparent motion in the
corrupted trace within that window is pure noise. Measure how much of the
corruption is exposed this way, using Tier 1 to define ground-truth stillness.
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

def still_mask(p, tr_thr=0.0015, rot_thr=np.radians(0.25)):
    """Per-frame 'object not moving' from consecutive-frame deltas."""
    dt=np.r_[0, np.linalg.norm(np.diff(p[:,:3],axis=0),axis=1)]
    q=p[:,3:7]/ (np.linalg.norm(p[:,3:7],axis=1,keepdims=True)+1e-12)
    dq=np.r_[0, 2*np.arccos(np.clip(np.abs((q[:-1]*q[1:]).sum(1)),0,1))]
    return (dt<tr_thr)&(dq<rot_thr)

def segments(mask, min_len=30):
    out=[]; s=None
    for i,m in enumerate(mask):
        if m and s is None: s=i
        elif not m and s is not None:
            if i-s>=min_len: out.append((s,i))
            s=None
    if s is not None and len(mask)-s>=min_len: out.append((s,len(mask)))
    return out

tot_frames=0; cov_frames=0; rows=[]
for f1 in sorted(glob.glob(T1+"/data/chunk-000/*.parquet")):
    f2=f1.replace("tier_1_multiview_caption","tier_2_synthetic_noise")
    t1,t2=pq.read_table(f1),pq.read_table(f2)
    ei=int(np.asarray(col(t1,"episode_index")).ravel()[0])
    vis=np.asarray(col(t1,"observation.object.visible")).astype(bool).ravel()
    p1,p2=col(t1,"observation.object.pose")[vis],col(t2,"observation.object.pose")[vis]
    segs=segments(still_mask(p1))
    cov=sum(e-s for s,e in segs); tot_frames+=len(p1); cov_frames+=cov
    # 在静止段内，tier2 的表观运动幅度 = 可直接观测到的噪声
    app_rot=[]; true_rot=[]
    for s,e in segs:
        q2=p2[s:e,3:7]; q2=q2/(np.linalg.norm(q2,axis=1,keepdims=True)+1e-12)
        c=np.abs((q2*q2[0]).sum(1)); app_rot.append(np.degrees(2*np.arccos(np.clip(c,0,1))).max())
        q1=p1[s:e,3:7]; q1=q1/(np.linalg.norm(q1,axis=1,keepdims=True)+1e-12)
        c1=np.abs((q1*q1[0]).sum(1)); true_rot.append(np.degrees(2*np.arccos(np.clip(c1,0,1))).max())
    rows.append((ei, meta[ei]["object"], len(p1), len(segs), 100*cov/max(len(p1),1),
                 np.mean(app_rot) if app_rot else 0.0, np.mean(true_rot) if true_rot else 0.0))

print(f"{'ep':>3} {'物体':<21} {'帧':>5} {'静止段':>5} {'覆盖%':>6} {'段内表观转角':>10} {'真实残留':>8}")
for r in sorted(rows,key=lambda r:-r[4]):
    print(f"{r[0]:>3} {r[1]:<21} {r[2]:>5} {r[3]:>5} {r[4]:>5.0f}% {r[5]:>9.2f}° {r[6]:>7.2f}°")
print(f"\n静止帧总覆盖率: {100*cov_frames/tot_frames:.1f}%  ({cov_frames}/{tot_frames})")
a=np.array([r[5] for r in rows]); t=np.array([r[6] for r in rows])
print(f"静止段内表观转角 mean {a.mean():.2f}°  (真实残留 {t.mean():.2f}°)")
print(f"=> 信噪比约 {a.mean()/max(t.mean(),1e-6):.1f}x")
