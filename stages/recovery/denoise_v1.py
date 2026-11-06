import pathlib
"""Drift estimation by static-segment integration.

observed(t) = true(t) . d(t),  d(0) = I  (frame 0 is clean: 0.6 mm measured)
Inside a static segment true(t) is constant, so the observed increment IS the
drift increment. During motion the drift increment is unobservable, but it is
zero-mean white noise, so the best predictor of a random walk is its last value.

    d(t) = d(t-1) . delta_obs(t)   if t is static
    d(t) = d(t-1)                  otherwise

The corrected trajectory is observed(t) . d(t)^-1.
"""
import sys,os,glob,json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "libs" / "v2d_infra"))
import numpy as np, pyarrow.parquet as pq, add_metrics as A

ROOT=str(pathlib.Path(__file__).resolve().parents[2] / "data/challenge/track_2"); T1=ROOT+"/tier_1_multiview_caption"
def col(t,c):
    v=t.column(c).to_numpy(zero_copy_only=False)
    return np.stack(v) if v.dtype==object else np.asarray(v)
meta={json.loads(l)["episode_index"]:json.loads(l) for l in open(T1+"/meta/episodes_metadata.jsonl")}
def nq(q): return q/(np.linalg.norm(q,axis=-1,keepdims=True)+1e-12)
def qmul(a,b):
    w1,x1,y1,z1=a[...,0],a[...,1],a[...,2],a[...,3]; w2,x2,y2,z2=b[...,0],b[...,1],b[...,2],b[...,3]
    return np.stack([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2],axis=-1)
def qinv(q): return np.stack([q[...,0],-q[...,1],-q[...,2],-q[...,3]],axis=-1)

def window_rot(p,W=91):
    q=nq(p[:,3:7]); T=len(q); out=np.zeros(T); h=W//2
    for i in range(T):
        a,b=max(0,i-h),min(T,i+h+1)
        c=np.abs((q[a:b]*q[a:b][:,None,:]).sum(-1)); out[i]=np.degrees(2*np.arccos(np.clip(c.min(),0,1)))
    return out
def window_tr(p,W=91):
    x=p[:,:3]; T=len(x); out=np.zeros(T); h=W//2
    for i in range(T):
        a,b=max(0,i-h),min(T,i+h+1); s=x[a:b]
        out[i]=np.linalg.norm(s[:,None,:]-s[None,:,:],axis=-1).max()
    return out

def denoise(p, W=91, rot_thr=23.5, tr_thr=0.06):
    """Integrate drift through static frames, hold it through motion."""
    q=nq(p[:,3:7]).copy(); x=p[:,:3].copy(); T=len(p)
    static=(window_rot(p,W)<rot_thr)&(window_tr(p,W)<tr_thr)
    dq=np.tile(np.array([1.,0,0,0]),(T,1)); dx=np.zeros((T,3))
    for t in range(1,T):
        dq[t]=dq[t-1]; dx[t]=dx[t-1]
        if static[t] and static[t-1]:
            inc=qmul(q[t],qinv(q[t-1]))                 # 观测增量 = 漂移增量
            if inc[0]<0: inc=-inc
            dq[t]=nq(qmul(inc,dq[t-1])); dx[t]=dx[t-1]+(x[t]-x[t-1])
    out=p.copy()
    out[:,3:7]=nq(qmul(qinv(dq),q))                     # 去掉漂移
    out[:,:3]=x-dx
    return out, static

if __name__ == "__main__":
    rows=[]
    for f1 in sorted(glob.glob(T1+"/data/chunk-000/*.parquet")):
        f2=f1.replace("tier_1_multiview_caption","tier_2_synthetic_noise")
        t1,t2=pq.read_table(f1),pq.read_table(f2)
        ei=int(np.asarray(col(t1,"episode_index")).ravel()[0])
        vis=np.asarray(col(t1,"observation.object.visible")).astype(bool).ravel()
        p1,p2=col(t1,"observation.object.pose")[vis],col(t2,"observation.object.pose")[vis]
        v=A.sample_mesh_vertices(os.path.join(T1,meta[ei]["mesh"]),n=500,seed=ei)
        pd,st=denoise(p2)
        rows.append((ei,meta[ei]["object"],
                     A.auc_from_add(A.add_per_frame(v,p1,p2))[0],
                     A.auc_from_add(A.add_per_frame(v,p1,pd))[0],
                     100*st.mean()))
    rows.sort(key=lambda r:-(r[3]-r[2]))
    print(f"{'ep':>3} {'物体':<21} {'原始':>6} {'降噪后':>7} {'增益':>7} {'静止%':>6}")
    for r in rows: print(f"{r[0]:>3} {r[1]:<21} {r[2]:>6.3f} {r[3]:>7.3f} {r[3]-r[2]:>+7.3f} {r[4]:>5.0f}%")
    a=np.array([r[2] for r in rows]); b=np.array([r[3] for r in rows])
    print(f"\n原始   AUC mean {a.mean():.3f}")
    print(f"降噪后 AUC mean {b.mean():.3f}   (增益 {b.mean()-a.mean():+.3f})")
    print(f"改善的序列 {int((b>a).sum())}/30   变差的 {int((b<a).sum())}/30")
    print(f"参考: 只修旋转上界 0.881 | CHORD 0.918")
