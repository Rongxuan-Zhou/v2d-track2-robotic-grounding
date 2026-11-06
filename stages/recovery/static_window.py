import pathlib
"""Window-displacement static detector.

Random-walk noise accumulates as sqrt(W) over a window; real motion accumulates
linearly. So a window-endpoint displacement separates them far better than a
per-frame velocity threshold does.
"""
import sys,os,glob,json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "libs" / "v2d_infra"))
import numpy as np, pyarrow.parquet as pq
ROOT=str(pathlib.Path(__file__).resolve().parents[2] / "data/challenge/track_2"); T1=ROOT+"/tier_1_multiview_caption"
def col(t,c):
    v=t.column(c).to_numpy(zero_copy_only=False)
    return np.stack(v) if v.dtype==object else np.asarray(v)

def norm_q(q): return q/(np.linalg.norm(q,axis=-1,keepdims=True)+1e-12)

def window_rot(p, W):
    """Max rotation from the window's first frame, per frame (deg)."""
    q=norm_q(p[:,3:7]); T=len(q); out=np.zeros(T)
    h=W//2
    for i in range(T):
        a,b=max(0,i-h),min(T,i+h+1)
        c=np.abs((q[a:b]*q[a:b][:,None,:]).sum(-1))
        out[i]=np.degrees(2*np.arccos(np.clip(c.min(),0,1)))
    return out

def gt_still(p, tr=0.0015, rot=np.radians(0.25)):
    dt=np.r_[0,np.linalg.norm(np.diff(p[:,:3],axis=0),axis=1)]
    q=norm_q(p[:,3:7]); dq=np.r_[0,2*np.arccos(np.clip(np.abs((q[:-1]*q[1:]).sum(1)),0,1))]
    return (dt<tr)&(dq<rot)

if __name__ == "__main__":
    eps=sorted(glob.glob(T1+"/data/chunk-000/*.parquet"))
    G={}; Wv={}
    WINS=[15,31,61,91]
    for f1 in eps:
        f2=f1.replace("tier_1_multiview_caption","tier_2_synthetic_noise")
        t1,t2=pq.read_table(f1),pq.read_table(f2)
        vis=np.asarray(col(t1,"observation.object.visible")).astype(bool).ravel()
        p1,p2=col(t1,"observation.object.pose")[vis],col(t2,"observation.object.pose")[vis]
        k=f1
        G[k]=gt_still(p1); Wv[k]={W:window_rot(p2,W) for W in WINS}

    print(f"{'窗口':>5} {'阈值°':>7} {'精确率':>7} {'召回率':>7} {'F1':>6}")
    best_overall=None
    for W in WINS:
        g=np.concatenate([G[k] for k in eps]); v=np.concatenate([Wv[k][W] for k in eps])
        best=None
        for thr in np.arange(0.5,25,0.25):
            d=v<thr
            tp=int((g&d).sum()); fp=int((~g&d).sum()); fn=int((g&~d).sum())
            prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1); f1=2*prec*rec/max(prec+rec,1e-9)
            if best is None or f1>best[3]: best=(thr,prec,rec,f1)
        print(f"{W:>5} {best[0]:>7.2f} {100*best[1]:>6.0f}% {100*best[2]:>6.0f}% {100*best[3]:>5.0f}")
        if best_overall is None or best[3]>best_overall[1][3]: best_overall=(W,best)
    W,b=best_overall
    print(f"\n最优: 窗口 {W} 帧, 阈值 {b[0]:.2f}°  ->  精确率 {100*b[1]:.0f}% 召回率 {100*b[2]:.0f}% F1 {100*b[3]:.0f}")
    print(f"对比逐帧阈值的 F1 64 (精确率 48% / 召回率 94%)")
