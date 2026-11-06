import pathlib
"""Exactly how many joints does SOMALayer return, and in what order?"""
import os,sys,json,warnings; warnings.filterwarnings("ignore")
import numpy as np, torch, pyarrow.parquet as pq
ASSETS=str(pathlib.Path(__file__).resolve().parents[2] / "third_party/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/body_models/soma")
T1=str(pathlib.Path(__file__).resolve().parents[2] / "data/challenge/track_2/tier_1_multiview_caption")
def col(t,c):
    v=t.column(c).to_numpy(zero_copy_only=False)
    return np.stack(v) if v.dtype==object else np.asarray(v)
from soma import SOMALayer
layer=SOMALayer(data_root=ASSETS,identity_model_type="mhr",device="cpu",enable_procedural_transforms=False)
rig=list(layer.rig_data["joint_names"])
print("rig_data['joint_names'] 长度:", len(rig))
print("  前 3:", rig[:3])
t=pq.read_table(T1+"/data/chunk-000/episode_000000.parquet")
p=col(t,"observation.human.pose")[:5]; tr=col(t,"observation.human.translation")[:5]
idc=col(t,"observation.human.identity_coeffs")[:5]; sc=col(t,"observation.human.scale_params")[:5]
o=layer(torch.tensor(p.reshape(5,77,3),dtype=torch.float32),torch.tensor(idc,dtype=torch.float32),
        scale_params=torch.tensor(sc,dtype=torch.float32),transl=torch.tensor(tr,dtype=torch.float32),apply_correctives=False)
print("\nlayer 返回类型:", type(o).__name__)
if isinstance(o,(tuple,list)):
    print("  元素数:", len(o))
    for i,e in enumerate(o[:4]):
        print(f"  [{i}] {type(e).__name__} shape={getattr(e,'shape',None)}")
    j=o[0]
else:
    j=o
if hasattr(j,"joints"): print("  .joints shape:", j.joints.shape); j=j.joints
J=np.asarray(j.detach())
print("\n关节张量 shape:", J.shape)
print(f"=> 返回 {J.shape[1]} 个关节；rig 有 {len(rig)} 个")
if J.shape[1]==len(rig):
    print("   ⇒ 含 Root，应当 [:,1:,:] 丢掉 -> 77")
elif J.shape[1]==len(rig)-1:
    print("   ⇒ 已不含 Root，**不该再丢**，直接用 -> 77")
# 用解剖学校验对齐：头应当最高，脚应当最低
for drop in (0,1):
    K=J[:,drop:,:]; nm=rig[drop:] if drop else rig
    if K.shape[1]!=len(nm): continue
    z=K[0,:,1]   # SOMA 系 Y 轴
    hi=nm[int(np.argmax(np.abs(z)))]
    idx={n:i for i,n in enumerate(nm)}
    print(f"\n  drop={drop}: 幅值最大的关节 = {hi}")
    for probe in ("Head","LeftFoot","LeftHand"):
        if probe in idx: print(f"     {probe:10s} -> {K[0,idx[probe]]}")
