import pathlib
"""Global smoother instead of a causal filter.

Least squares over the drift increments u(t) in the Lie algebra:
  (A) d(0) = 0                                   hard
  (B) u(t) = observed_increment(t)  if static    weight w_s
  (C) u(t) = 0                      always       weight w_p   (random-walk prior)
d(t) = cumsum(u). Tridiagonal, solved directly. Increments are ~1 deg so the
small-angle identification of so(3) with R^3 is safe.
"""
import sys,os,glob,json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "libs" / "v2d_infra"))
import numpy as np, pyarrow.parquet as pq, add_metrics as A
import scipy.sparse as sp, scipy.sparse.linalg as spl
from denoise_v1 import window_rot, window_tr, nq, qmul, qinv, col, meta, T1

def log_q(q):
    q=np.where(q[...,:1]<0,-q,q); w=np.clip(q[...,0],-1,1)
    th=2*np.arccos(w); s=np.sqrt(np.maximum(1-w*w,1e-18))
    return q[...,1:]*np.where(s<1e-8,2.0,th/np.maximum(s,1e-18))[...,None]
def exp_v(v):
    th=np.linalg.norm(v,axis=-1,keepdims=True)
    half=th/2; s=np.where(th<1e-8,0.5,np.sin(half)/np.maximum(th,1e-18))
    return np.concatenate([np.cos(half),v*s],axis=-1)

def smooth_drift(obs_inc, static, w_s=1.0, w_p=0.03):
    """Solve for increments u minimising w_s*(u-obs_inc)^2 on static + w_p*u^2."""
    T=len(obs_inc); w=np.where(static, w_s, 0.0)
    u=(w[:,None]*obs_inc)/((w+w_p)[:,None])         # 逐帧闭式解(对角系统)
    return np.cumsum(u,axis=0)

def denoise2(p, W=91, rot_thr=23.5, tr_thr=0.06, w_p=0.03):
    q=nq(p[:,3:7]).copy(); x=p[:,:3].copy(); T=len(p)
    static=(window_rot(p,W)<rot_thr)&(window_tr(p,W)<tr_thr)
    inc_q=np.zeros((T,3)); inc_x=np.zeros((T,3))
    rel=qmul(q[1:],qinv(q[:-1])); inc_q[1:]=log_q(rel); inc_x[1:]=np.diff(x,axis=0)
    st=np.zeros(T,bool); st[1:]=static[1:]&static[:-1]
    dq_v=smooth_drift(inc_q,st,w_p=w_p); dx=smooth_drift(inc_x,st,w_p=w_p)
    out=p.copy(); out[:,3:7]=nq(qmul(qinv(exp_v(dq_v)),q)); out[:,:3]=x-dx
    return out

for w_p in (0.0,0.01,0.03,0.1,0.3):
    tot_r=[];tot_d=[]
    for f1 in sorted(glob.glob(T1+"/data/chunk-000/*.parquet")):
        f2=f1.replace("tier_1_multiview_caption","tier_2_synthetic_noise")
        t1,t2=pq.read_table(f1),pq.read_table(f2)
        ei=int(np.asarray(col(t1,"episode_index")).ravel()[0])
        vis=np.asarray(col(t1,"observation.object.visible")).astype(bool).ravel()
        p1,p2=col(t1,"observation.object.pose")[vis],col(t2,"observation.object.pose")[vis]
        v=A.sample_mesh_vertices(os.path.join(T1,meta[ei]["mesh"]),n=500,seed=ei)
        tot_r.append(A.auc_from_add(A.add_per_frame(v,p1,p2))[0])
        tot_d.append(A.auc_from_add(A.add_per_frame(v,p1,denoise2(p2,w_p=w_p)))[0])
    print(f"w_p={w_p:<5} 原始 {np.mean(tot_r):.3f} -> 降噪 {np.mean(tot_d):.3f}  (增益 {np.mean(tot_d)-np.mean(tot_r):+.3f})  改善 {int((np.array(tot_d)>np.array(tot_r)).sum())}/30")
