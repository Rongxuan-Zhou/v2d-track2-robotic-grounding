import pathlib
"""Denoiser v4: grasp-rigidity makes the motion segments observable.

observed(t) = D(t) . true(t),  D(0) = I.
Static segment : D(t) = dObs(t) . D(t-1)                  (true is constant)
Grasp segment  : true(t) = Hrel(t0->t) . true(t0)          (object rides the hand)
                 => D(t) = obs(t) . true(t0)^-1 . Hrel(t)^-1
Hrel comes from Procrustes on 12 hand keypoints, so only RELATIVE hand motion is
needed -- the SOMA layer gives joint positions but not joint orientations.
"""
import os,sys,json,glob,warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "libs" / "v2d_infra"))
from soma_fk import fk as _soma_fk
import numpy as np, torch, pyarrow.parquet as pq, add_metrics as A
from denoise_v1 import window_rot, window_tr, nq, qmul, qinv, col, meta, T1
ASSETS=str(pathlib.Path(__file__).resolve().parents[2] / "third_party/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/body_models/soma")
from soma import SOMALayer
layer=SOMALayer(data_root=ASSETS,identity_model_type="mhr",device="cpu",enable_procedural_transforms=False)
names=list(layer.rig_data["joint_names"])[1:]
LH=[i for i,n in enumerate(names) if n.startswith("LeftHand") or n=="LeftHand"]
RH=[i for i,n in enumerate(names) if n.startswith("RightHand") or n=="RightHand"]
R_CV=np.diag([1.0,-1.0,-1.0])

def mat_to_q(M):
    w=np.sqrt(np.clip(1+M[...,0,0]+M[...,1,1]+M[...,2,2],1e-12,None))/2
    return np.stack([w,(M[...,2,1]-M[...,1,2])/(4*w),(M[...,0,2]-M[...,2,0])/(4*w),(M[...,1,0]-M[...,0,1])/(4*w)],-1)
def to_soma(p):
    q=p.copy(); q[:,:3]=p[:,:3]@R_CV.T
    q[:,3:7]=mat_to_q(np.einsum("ij,tjk->tik",R_CV,A.quat_wxyz_to_matrix(p[:,3:7])))
    return q
def fk(t):   # 委托给 libs/v2d_infra/soma_fk（FK 唯一入口，带 bone_length_flexibles）
    return _soma_fk(t, flex=True)

def procrustes(P,Q):
    """Rigid R,t mapping P -> Q (both (N,3)). Kabsch."""
    pc,qc=P.mean(0),Q.mean(0)
    H=(P-pc).T@(Q-qc); U,_,Vt=np.linalg.svd(H); d=np.sign(np.linalg.det(Vt.T@U.T))
    R=Vt.T@np.diag([1,1,d])@U.T
    return R, qc-R@pc

def segments(mask,minlen):
    out=[];s=None
    for i,m in enumerate(mask):
        if m and s is None: s=i
        elif not m and s is not None:
            if i-s>=minlen: out.append((s,i))
            s=None
    if s is not None and len(mask)-s>=minlen: out.append((s,len(mask)))
    return out

def denoise(p_obs, J, static, grasp_d, grasp_thr=0.06, minlen=10):
    T=len(p_obs)
    Dq=np.tile(np.array([1.,0,0,0]),(T,1)); Dx=np.zeros((T,3))
    q=nq(p_obs[:,3:7]); x=p_obs[:,:3]
    grasp=(grasp_d<grasp_thr)&(~static)
    gsegs=segments(grasp,minlen)
    gstart={s:(s,e) for s,e in gsegs}
    inside={}
    for s,e in gsegs:
        for t in range(s,e): inside[t]=s
    for t in range(1,T):
        Dq[t]=Dq[t-1]; Dx[t]=Dx[t-1]
        if static[t] and static[t-1]:
            inc=qmul(q[t],qinv(q[t-1]))
            if inc[0]<0: inc=-inc
            Dq[t]=nq(qmul(inc,Dq[t-1])); Dx[t]=Dx[t-1]+(x[t]-x[t-1])
        elif t in inside:
            t0=inside[t]
            hs=LH if np.linalg.norm(J[t0][LH].mean(0)-x[t0])<np.linalg.norm(J[t0][RH].mean(0)-x[t0]) else RH
            try: R,tt=procrustes(J[t0][hs],J[t][hs])
            except Exception: continue
            # true(t0) = D(t0)^-1 . obs(t0)
            Dq0,Dx0=Dq[t0],Dx[t0]
            Rq0=A.quat_wxyz_to_matrix(Dq0[None])[0]
            true0_R=Rq0.T@A.quat_wxyz_to_matrix(q[t0][None])[0]; true0_x=x[t0]-Dx0
            # true(t) = Hrel . true(t0)
            trueR=R@true0_R; truex=R@true0_x+tt
            # D(t) = obs(t) . true(t)^-1
            Rt=A.quat_wxyz_to_matrix(q[t][None])[0]
            DR=Rt@trueR.T; Dq[t]=nq(mat_to_q(DR[None])[0]); Dx[t]=x[t]-truex
    out=p_obs.copy()
    out[:,3:7]=nq(qmul(qinv(Dq),q))
    out[:,:3]=x-Dx          # 平移漂移是加性的，与旋转独立(实测腐蚀就是这个形式)
    return out

if __name__=="__main__":
    res={"raw":[],"v1_geom":[],"v4_grasp":[]}
    for f1 in sorted(glob.glob(T1+"/data/chunk-000/*.parquet")):
        f2=f1.replace("tier_1_multiview_caption","tier_2_synthetic_noise")
        t1,t2=pq.read_table(f1),pq.read_table(f2)
        ei=int(np.asarray(col(t1,"episode_index")).ravel()[0])
        vis=np.asarray(col(t1,"observation.object.visible")).astype(bool).ravel()
        p1=to_soma(col(t1,"observation.object.pose")[vis]); p2=to_soma(col(t2,"observation.object.pose")[vis])
        v=A.sample_mesh_vertices(os.path.join(T1,meta[ei]["mesh"]),n=500,seed=ei)
        J=fk(t2)[vis]
        ov=A.transform_vertices(A.sample_mesh_vertices(os.path.join(T1,meta[ei]["mesh"]),n=200,seed=ei),p2)
        gd=np.linalg.norm(J[:,LH+RH,None,:]-ov[:,None,:,:],axis=-1).min(axis=(1,2))
        st=(window_rot(p2,91)<23.5)&(window_tr(p2,91)<0.06)
        res["raw"].append(A.auc_from_add(A.add_per_frame(v,p1,p2))[0])
        res["v1_geom"].append(A.auc_from_add(A.add_per_frame(v,p1,denoise(p2,J,st,gd,grasp_thr=0.0)))[0])
        res["v4_grasp"].append(A.auc_from_add(A.add_per_frame(v,p1,denoise(p2,J,st,gd)))[0])
    r=np.array(res["raw"])
    for k in res:
        a=np.array(res[k]); print(f"{k:<10} AUC {a.mean():.3f}  中位 {np.median(a):.3f}  改善 {int((a>r).sum())}/30")
