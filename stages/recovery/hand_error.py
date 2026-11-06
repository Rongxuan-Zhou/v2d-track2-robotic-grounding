import pathlib
"""How dirty is the hand in WORLD frame?

The 76 non-root joints are byte-identical between tiers, but they are clean only
RELATIVE TO THE ROOT. The root carries 16 mm translation + 10 deg rotation error,
which is lever-armed by limb length. If the resulting hand world error exceeds the
object's own 30 mm, then anchoring the object on the hand is anchoring a cleaner
quantity to a dirtier one.
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
H=[i for i,n in enumerate(names) if (n.endswith("End") and "Hand" in n) or n in ("LeftHand","RightHand")]
FT=[i for i,n in enumerate(names) if n in ("LeftFoot","RightFoot","LeftToeBase","RightToeBase")]
def fk(t):   # 委托给 libs/v2d_infra/soma_fk（FK 唯一入口，带 bone_length_flexibles）
    return _soma_fk(t, flex=True)

DH=[];DF=[];DR=[]
for f1 in sorted(glob.glob(T1+"/data/chunk-000/*.parquet"))[:10]:
    f2=f1.replace("tier_1_multiview_caption","tier_2_synthetic_noise")
    t1,t2=pq.read_table(f1),pq.read_table(f2)
    J1,J2=fk(t1),fk(t2)
    DH.append(np.linalg.norm(J1[:,H,:]-J2[:,H,:],axis=-1).mean(1))
    DF.append(np.linalg.norm(J1[:,FT,:]-J2[:,FT,:],axis=-1).mean(1))
    DR.append(np.linalg.norm(col(t1,"observation.human.translation")-col(t2,"observation.human.translation"),axis=1))
DH,DF,DR=[np.concatenate(x)*1000 for x in (DH,DF,DR)]
print(f"{'':16}{'mean':>9}{'p50':>9}{'p95':>9}{'max':>9}  (mm)")
for nm,v in (("根平移误差",DR),("手部世界误差",DH),("脚部世界误差",DF)):
    print(f"{nm:16}{v.mean():>9.1f}{np.median(v):>9.1f}{np.percentile(v,95):>9.1f}{v.max():>9.1f}")
print(f"\n对比: 物体自身平移误差 18.7 mm / 旋转 21.4 deg")
print(f"=> 手部世界误差是物体平移误差的 {DH.mean()/18.7:.1f} 倍")
