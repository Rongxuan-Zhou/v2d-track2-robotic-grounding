import pathlib
"""No-slip is where the root signal lives.

A planted foot should not translate in world. Tier 1 roughly honours that; the
Tier 2 root drift makes it appear to slide. That apparent slip is the observable,
and it is horizontal -- which is why the ground-plane height test only saw 7 mm
of a 71 mm error.
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
FEET=[i for i,n in enumerate(names) if n in ("LeftFoot","LeftToeBase","RightFoot","RightToeBase")]
def fk(t):   # 委托给 libs/v2d_infra/soma_fk（FK 唯一入口，带 bone_length_flexibles）
    return _soma_fk(t, flex=True)

S1=[];S2=[];COV=[]
for f1 in sorted(glob.glob(T1+"/data/chunk-000/*.parquet"))[:10]:
    f2=f1.replace("tier_1_multiview_caption","tier_2_synthetic_noise")
    J1,J2=fk(pq.read_table(f1)),fk(pq.read_table(f2))
    F1,F2=J1[:,FEET,:],J2[:,FEET,:]
    # 支撑判据只用 tier1 定义真值；推理时改用 tier2 自身(见下)
    v1=np.r_[np.zeros((1,len(FEET))),np.linalg.norm(np.diff(F1,axis=0),axis=-1)]
    planted=v1<0.003
    COV.append(planted.any(1).mean())
    # 支撑期间的逐帧位移 = 表观滑动
    d1=np.where(planted, v1, np.nan)
    v2=np.r_[np.zeros((1,len(FEET))),np.linalg.norm(np.diff(F2,axis=0),axis=-1)]
    d2=np.where(planted, v2, np.nan)
    S1.append(d1[~np.isnan(d1)]); S2.append(d2[~np.isnan(d2)])
S1,S2=np.concatenate(S1)*1000,np.concatenate(S2)*1000
print(f"支撑帧覆盖率 {100*np.mean(COV):.0f}%\n")
print(f"{'支撑期逐帧脚位移':20}{'mean':>9}{'p50':>9}{'p95':>9}  (mm/帧)")
print(f"{'  Tier 1 (真值)':20}{S1.mean():>9.3f}{np.median(S1):>9.3f}{np.percentile(S1,95):>9.3f}")
print(f"{'  Tier 2 (被污染)':20}{S2.mean():>9.3f}{np.median(S2):>9.3f}{np.percentile(S2,95):>9.3f}")
print(f"\n=> Tier2 的表观滑动是 Tier1 的 {S2.mean()/max(S1.mean(),1e-9):.1f} 倍")
print(f"   这个比值就是根漂移可观测的信噪比")
