import pathlib
"""Forward-kinematics the challenge human pose, then measure hand-object distance.

The 76 clean joints give trustworthy hand geometry; only the root carries error
(16 mm / 10 deg). If 'a hand is near the object' tracks 'the object is moving',
then contact is a physical handle on when the object may move at all.
"""
import os, sys, json, glob, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "libs" / "v2d_infra"))
from soma_fk import fk as _soma_fk
import numpy as np, torch, pyarrow.parquet as pq
import add_metrics as A

ROOT = str(pathlib.Path(__file__).resolve().parents[2] / "data/challenge/track_2")
T1 = ROOT + "/tier_1_multiview_caption"
ASSETS = str(pathlib.Path(__file__).resolve().parents[2] / "third_party/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/body_models/soma")
def col(t,c):
    v=t.column(c).to_numpy(zero_copy_only=False)
    return np.stack(v) if v.dtype==object else np.asarray(v)
meta={json.loads(l)["episode_index"]:json.loads(l) for l in open(T1+"/meta/episodes_metadata.jsonl")}

from soma import SOMALayer
layer = SOMALayer(data_root=ASSETS, identity_model_type="mhr", device="cpu",
                  enable_procedural_transforms=False)
names = list(layer.rig_data["joint_names"])[1:]
TIPS = [i for i,n in enumerate(names) if n.endswith("End") and "Hand" in n]
HANDS = [i for i,n in enumerate(names) if n in ("LeftHand","RightHand")]
print(f"指尖关节 {len(TIPS)} 个, 手掌 {HANDS}")

def fk(t, n=None):   # 委托给 libs/v2d_infra/soma_fk（FK 唯一入口）
    return _soma_fk(t.slice(0, n) if n else t, flex=True)

rows=[]
for f1 in sorted(glob.glob(T1+"/data/chunk-000/*.parquet"))[:8]:
    t1=pq.read_table(f1); ei=int(np.asarray(col(t1,"episode_index")).ravel()[0])
    J = fk(t1)
    p1 = col(t1,"observation.object.pose").copy()
    # OpenCV world -> SOMA world, per soma_to_g1.py:364-371 (_R_CV_TO_SOMA)
    R_CV = np.diag([1.0,-1.0,-1.0])
    p1[:,:3] = p1[:,:3] @ R_CV.T
    M = A.quat_wxyz_to_matrix(p1[:,3:7])
    M = np.einsum("ij,tjk->tik", R_CV, M)
    w = np.sqrt(np.clip(1+M[:,0,0]+M[:,1,1]+M[:,2,2],1e-12,None))/2
    p1[:,3]=w
    p1[:,4]=(M[:,2,1]-M[:,1,2])/(4*w)
    p1[:,5]=(M[:,0,2]-M[:,2,0])/(4*w)
    p1[:,6]=(M[:,1,0]-M[:,0,1])/(4*w)
    verts = A.sample_mesh_vertices(os.path.join(T1,meta[ei]["mesh"]), n=300, seed=ei)
    ov = A.transform_vertices(verts, p1)                    # (T,V,3)
    hands = J[:, TIPS+HANDS, :]                             # (T,H,3)
    d = np.linalg.norm(hands[:,:,None,:]-ov[:,None,:,:],axis=-1).min(axis=(1,2))  # (T,)
    step = np.r_[0, np.linalg.norm(np.diff(p1[:,:3],axis=0),axis=1)]
    moving = step > 0.0015
    rows.append((ei, meta[ei]["object"], d.min(), np.median(d),
                 np.median(d[moving]) if moving.any() else np.nan,
                 np.median(d[~moving]) if (~moving).any() else np.nan))

print(f"\n{'ep':>3} {'物体':<20} {'最近':>6} {'中位':>6} {'运动时':>7} {'静止时':>7}")
for r in rows:
    print(f"{r[0]:>3} {r[1]:<20} {r[2]:>5.2f}m {r[3]:>5.2f}m {r[4]:>6.2f}m {r[5]:>6.2f}m")
mv=np.array([r[4] for r in rows]); st=np.array([r[5] for r in rows])
print(f"\n运动时手-物距离中位数 {np.nanmedian(mv):.3f} m")
print(f"静止时手-物距离中位数 {np.nanmedian(st):.3f} m")
print(f"=> 分离度 {np.nanmedian(st)/max(np.nanmedian(mv),1e-6):.2f}x")
