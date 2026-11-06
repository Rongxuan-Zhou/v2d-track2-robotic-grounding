import pathlib
"""Is foot-ground contact a strong enough anchor to correct the root drift?

Root error (16 mm / 10 deg) is shared by the whole body and lever-armed into
75 mm of foot world error. The ground plane is supplied per episode, so a planted
foot gives both a height constraint and a no-slip constraint. Measure whether the
signal is actually there before building the solver.
"""
import os,sys,json,glob,warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "libs" / "v2d_infra"))
from soma_fk import fk as _soma_fk
import numpy as np, torch, pyarrow.parquet as pq
from denoise_v1 import col, meta, T1
ASSETS=str(pathlib.Path(__file__).resolve().parents[2] / "third_party/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/body_models/soma")
from soma import SOMALayer
layer=SOMALayer(data_root=ASSETS,identity_model_type="mhr",device="cpu",enable_procedural_transforms=False)
names=list(layer.rig_data["joint_names"])[1:]
LF=[i for i,n in enumerate(names) if n in ("LeftFoot","LeftToeBase","LeftToeEnd")]
RF=[i for i,n in enumerate(names) if n in ("RightFoot","RightToeBase","RightToeEnd")]
def fk(t):   # 委托给 libs/v2d_infra/soma_fk（FK 唯一入口，带 bone_length_flexibles）
    return _soma_fk(t, flex=True)

R_CV=np.diag([1.0,-1.0,-1.0])
rows=[]
for f1 in sorted(glob.glob(T1+"/data/chunk-000/*.parquet"))[:10]:
    f2=f1.replace("tier_1_multiview_caption","tier_2_synthetic_noise")
    t1,t2=pq.read_table(f1),pq.read_table(f2)
    ei=int(np.asarray(col(t1,"episode_index")).ravel()[0])
    gp=json.load(open(os.path.join(T1,meta[ei]["ground_plane"])))["plane"]
    n=np.array(gp[:3]); d=gp[3]; nn=np.linalg.norm(n); n=n/nn; d=d/nn
    J1,J2=fk(t1),fk(t2)
    # FK 在 SOMA 系，平面在 OpenCV 系 -> 把 FK 转回 OpenCV 比较
    def to_cv(J): return J@R_CV.T
    for tag,J in (("t1",to_cv(J1)),("t2",to_cv(J2))):
        h=(J[:,LF+RF,:]@n)+d                       # 各脚点到平面的有符号距离
        lo=h.min(axis=1)                           # 每帧最低脚点
        if tag=="t1": h1,lo1=h,lo
        else: h2,lo2=h,lo
    # 支撑判据：脚点接近平面且速度小
    v=np.linalg.norm(np.diff(to_cv(J1)[:,LF+RF,:],axis=0),axis=-1)
    v=np.r_[v[:1],v]
    planted=(np.abs(h1)<0.06)&(v<0.004)
    rows.append((ei, 100*planted.any(1).mean(), np.median(lo1), np.median(lo2),
                 np.median(np.abs(lo1-lo2))))
print(f"{'ep':>3} {'有支撑脚%':>9} {'t1最低脚高':>11} {'t2最低脚高':>11} {'t1-t2差':>9}")
for r in rows: print(f"{r[0]:>3} {r[1]:>8.0f}% {r[2]:>10.3f}m {r[3]:>10.3f}m {r[4]:>8.3f}m")
a=np.array([r[1] for r in rows]); dd=np.array([r[4] for r in rows])
print(f"\n有支撑脚的帧占比 {a.mean():.0f}%")
print(f"t1/t2 最低脚高差异中位数 {1000*np.median(dd):.1f} mm  <- 这是根漂移在脚高上的投影")
