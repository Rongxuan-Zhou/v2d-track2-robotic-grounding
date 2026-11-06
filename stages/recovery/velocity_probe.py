import pathlib
import sys,os,glob,json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "libs" / "v2d_infra"))
import numpy as np, pyarrow.parquet as pq
ROOT=str(pathlib.Path(__file__).resolve().parents[2] / "data/challenge/track_2"); T1=ROOT+"/tier_1_multiview_caption"
def col(t,c):
    v=t.column(c).to_numpy(zero_copy_only=False)
    return np.stack(v) if v.dtype==object else np.asarray(v)
def dq_per_frame(p):
    q=p[:,3:7]/(np.linalg.norm(p[:,3:7],axis=1,keepdims=True)+1e-12)
    return np.r_[0,np.degrees(2*np.arccos(np.clip(np.abs((q[:-1]*q[1:]).sum(1)),0,1)))]
def dt_per_frame(p): return np.r_[0,np.linalg.norm(np.diff(p[:,:3],axis=0),axis=1)]*1000

S1=[];S2=[];M1=[];M2=[]
for f1 in sorted(glob.glob(T1+"/data/chunk-000/*.parquet")):
    f2=f1.replace("tier_1_multiview_caption","tier_2_synthetic_noise")
    t1,t2=pq.read_table(f1),pq.read_table(f2)
    vis=np.asarray(col(t1,"observation.object.visible")).astype(bool).ravel()
    p1,p2=col(t1,"observation.object.pose")[vis],col(t2,"observation.object.pose")[vis]
    r1,r2=dq_per_frame(p1),dq_per_frame(p2)
    still=(r1<0.25)&(dt_per_frame(p1)<1.5)
    S1.append(r1[still]);S2.append(r2[still]);M1.append(r1[~still]);M2.append(r2[~still])
S1,S2,M1,M2=[np.concatenate(x) for x in (S1,S2,M1,M2)]
print("\n逐帧转角增量 (度/帧)")
print(f"{'':22s}{'p50':>8}{'p90':>8}{'p99':>8}{'max':>8}")
print(f"{'真静止 · Tier1':22s}{np.median(S1):8.3f}{np.percentile(S1,90):8.3f}{np.percentile(S1,99):8.3f}{S1.max():8.3f}")
print(f"{'真静止 · Tier2':22s}{np.median(S2):8.3f}{np.percentile(S2,90):8.3f}{np.percentile(S2,99):8.3f}{S2.max():8.3f}")
print(f"{'真运动 · Tier1':22s}{np.median(M1):8.3f}{np.percentile(M1,90):8.3f}{np.percentile(M1,99):8.3f}{M1.max():8.3f}")
print(f"{'真运动 · Tier2':22s}{np.median(M2):8.3f}{np.percentile(M2,90):8.3f}{np.percentile(M2,99):8.3f}{M2.max():8.3f}")
print(f"\n=> Tier2 静止期的噪声增量中位数 {np.median(S2):.3f}°/帧，是 Tier1 的 {np.median(S2)/max(np.median(S1),1e-9):.0f} 倍")
print(f"   但仍远小于 Tier2 真运动期的 {np.median(M2):.3f}°/帧")
# 找最优阈值
best=None
for thr in np.arange(0.2,4.0,0.05):
    tp=(S2<thr).sum(); fp=(M2<thr).sum(); fn=(S2>=thr).sum()
    prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1); f1=2*prec*rec/max(prec+rec,1e-9)
    if best is None or f1>best[3]: best=(thr,prec,rec,f1)
print(f"\n只调旋转阈值的最优点: thr={best[0]:.2f}°/帧  精确率 {100*best[1]:.0f}%  召回率 {100*best[2]:.0f}%  F1 {100*best[3]:.0f}")
