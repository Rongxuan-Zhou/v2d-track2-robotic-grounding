import pathlib
"""Denoiser v3: contact veto AND geometric window, combined.

Two semi-independent static signals. Geometry sees the corrupted trace only;
contact sees the (clean) body. Their intersection should be far more precise
than either, and precision is what matters -- a false static anchor injects error.
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
H=[i for i,n in enumerate(names) if (n.endswith("End") and "Hand" in n) or n in ("LeftHand","RightHand")]
R_CV=np.diag([1.0,-1.0,-1.0])
def to_soma(p):
    q=p.copy(); q[:,:3]=p[:,:3]@R_CV.T
    M=np.einsum("ij,tjk->tik",R_CV,A.quat_wxyz_to_matrix(p[:,3:7]))
    w=np.sqrt(np.clip(1+M[:,0,0]+M[:,1,1]+M[:,2,2],1e-12,None))/2
    q[:,3]=w;q[:,4]=(M[:,2,1]-M[:,1,2])/(4*w);q[:,5]=(M[:,0,2]-M[:,2,0])/(4*w);q[:,6]=(M[:,1,0]-M[:,0,1])/(4*w)
    return q
def fk(t):   # 委托给 libs/v2d_infra/soma_fk（FK 唯一入口，带 bone_length_flexibles）
    return _soma_fk(t, flex=True)

def integrate(p, static):
    q=nq(p[:,3:7]).copy(); x=p[:,:3].copy(); T=len(p)
    dq=np.tile(np.array([1.,0,0,0]),(T,1)); dx=np.zeros((T,3))
    for t in range(1,T):
        dq[t]=dq[t-1]; dx[t]=dx[t-1]
        if static[t] and static[t-1]:
            inc=qmul(q[t],qinv(q[t-1]))
            if inc[0]<0: inc=-inc
            dq[t]=nq(qmul(inc,dq[t-1])); dx[t]=dx[t-1]+(x[t]-x[t-1])
    out=p.copy(); out[:,3:7]=nq(qmul(qinv(dq),q)); out[:,:3]=x-dx
    return out

if __name__ == "__main__":
    res={"raw":[]}
    for f1 in sorted(glob.glob(T1+"/data/chunk-000/*.parquet")):
        f2=f1.replace("tier_1_multiview_caption","tier_2_synthetic_noise")
        t1,t2=pq.read_table(f1),pq.read_table(f2)
        ei=int(np.asarray(col(t1,"episode_index")).ravel()[0])
        vis=np.asarray(col(t1,"observation.object.visible")).astype(bool).ravel()
        p1=col(t1,"observation.object.pose")[vis]; p2=col(t2,"observation.object.pose")[vis]
        v=A.sample_mesh_vertices(os.path.join(T1,meta[ei]["mesh"]),n=500,seed=ei)
        # 接触用 tier2 的身体(推理时只有它)，但身体 76 关节干净，只有根带误差
        J=fk(t2)[vis]; ov=A.transform_vertices(A.sample_mesh_vertices(os.path.join(T1,meta[ei]["mesh"]),n=200,seed=ei), to_soma(p2))
        d=np.linalg.norm(J[:,H,None,:]-ov[:,None,:,:],axis=-1).min(axis=(1,2))
        geom=(window_rot(p2,91)<23.5)&(window_tr(p2,91)<0.06)
        cont=d>0.15
        res["raw"].append(A.auc_from_add(A.add_per_frame(v,p1,p2))[0])
        variants=[("geom",geom),("contact",cont),("union",geom|cont),("inter",geom&cont)]
        for thr in (0.10,0.20,0.30):
            variants.append((f"inter_d{thr:.2f}", geom&(d>thr)))
        for k,m in variants:
            res.setdefault(k,[]).append(A.auc_from_add(A.add_per_frame(v,p1,integrate(p2,m)))[0])
    print(f"{'方案':<10}{'AUC mean':>10}{'中位数':>9}{'改善数':>8}")
    r=np.array(res["raw"])
    for k in res:
        a=np.array(res[k]); print(f"{k:<10}{a.mean():>10.3f}{np.median(a):>9.3f}{int((a>r).sum()) if k!='raw' else 0:>8}/30")
    print(f"\n上界 0.881(只修旋转) | CHORD 0.918")
