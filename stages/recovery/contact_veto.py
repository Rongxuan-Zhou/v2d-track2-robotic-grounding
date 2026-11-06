import pathlib
"""'Hand far from object => object cannot be moving' as a one-sided veto.

Asymmetric by design: we want very high precision on the 'static' call, because
a false static anchor actively injects error into the drift integration.
Recall can be low; we only need enough anchor frames.
"""
import os,sys,json,glob,warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "libs" / "v2d_infra"))
from soma_fk import fk as _soma_fk
import numpy as np, torch, pyarrow.parquet as pq, add_metrics as A
ROOT=str(pathlib.Path(__file__).resolve().parents[2] / "data/challenge/track_2"); T1=ROOT+"/tier_1_multiview_caption"
ASSETS=str(pathlib.Path(__file__).resolve().parents[2] / "third_party/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/body_models/soma")
def col(t,c):
    v=t.column(c).to_numpy(zero_copy_only=False)
    return np.stack(v) if v.dtype==object else np.asarray(v)
meta={json.loads(l)["episode_index"]:json.loads(l) for l in open(T1+"/meta/episodes_metadata.jsonl")}
from soma import SOMALayer
layer=SOMALayer(data_root=ASSETS,identity_model_type="mhr",device="cpu",enable_procedural_transforms=False)
names=list(layer.rig_data["joint_names"])[1:]
H=[i for i,n in enumerate(names) if (n.endswith("End") and "Hand" in n) or n in ("LeftHand","RightHand")]
R_CV=np.diag([1.0,-1.0,-1.0])
def to_soma(p):
    q=p.copy(); q[:,:3]=p[:,:3]@R_CV.T
    M=np.einsum("ij,tjk->tik",R_CV,A.quat_wxyz_to_matrix(p[:,3:7]))
    w=np.sqrt(np.clip(1+M[:,0,0]+M[:,1,1]+M[:,2,2],1e-12,None))/2
    q[:,3]=w; q[:,4]=(M[:,2,1]-M[:,1,2])/(4*w); q[:,5]=(M[:,0,2]-M[:,2,0])/(4*w); q[:,6]=(M[:,1,0]-M[:,0,1])/(4*w)
    return q
def fk(t):   # 委托给 libs/v2d_infra/soma_fk（FK 唯一入口，带 bone_length_flexibles）
    return _soma_fk(t, flex=True)

D=[];MOV=[]
for f1 in sorted(glob.glob(T1+"/data/chunk-000/*.parquet")):
    t1=pq.read_table(f1); ei=int(np.asarray(col(t1,"episode_index")).ravel()[0])
    J=fk(t1); p1=to_soma(col(t1,"observation.object.pose"))
    v=A.sample_mesh_vertices(os.path.join(T1,meta[ei]["mesh"]),n=200,seed=ei)
    ov=A.transform_vertices(v,p1)
    d=np.linalg.norm(J[:,H,None,:]-ov[:,None,:,:],axis=-1).min(axis=(1,2))
    step=np.r_[0,np.linalg.norm(np.diff(p1[:,:3],axis=0),axis=1)]
    q=p1[:,3:7]/(np.linalg.norm(p1[:,3:7],axis=1,keepdims=True)+1e-12)
    dq=np.r_[0,2*np.arccos(np.clip(np.abs((q[:-1]*q[1:]).sum(1)),0,1))]
    D.append(d); MOV.append((step>0.0015)|(dq>np.radians(0.25)))
D=np.concatenate(D); MOV=np.concatenate(MOV); STILL=~MOV
print(f"总帧 {len(D)}, 真静止 {100*STILL.mean():.0f}%\n")
print(f"{'阈值m':>7} {'判静止%':>8} {'精确率':>7} {'召回率':>7}")
for thr in (0.10,0.15,0.20,0.25,0.30,0.40,0.50,0.70):
    veto=D>thr
    if veto.sum()==0: continue
    prec=100*(veto&STILL).sum()/veto.sum(); rec=100*(veto&STILL).sum()/STILL.sum()
    print(f"{thr:>7.2f} {100*veto.mean():>7.0f}% {prec:>6.1f}% {rec:>6.1f}%")
