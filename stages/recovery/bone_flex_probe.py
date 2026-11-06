import pathlib
"""bone_flex_probe: 量化 observation.human.bone_length_flexibles 对 FK 的影响。

背景：挑战 parquet 给了 (T,6) 的 bone_length_flexibles，SOMA 的 identity_model
把它注入 pose_params[:,130:136]，用途是改 spine/neck/shoulder/arm/hip/leg 六处的
骨长（identity_model.py:331-341）。

但两处都没传它：
  - 我们 recovery 里的 fk()
  - 官方 read_soma._reconstruct（read_soma.py:507-518），且 _load_npz 的必填键里没有

实测值非零(±0.48)且序列内逐帧变化。本脚本测它改动关节位置的量级，判断：
  (a) 我们报的手部 37.0 mm / 脚部 71.3 mm 世界误差是否需要重算
  (b) tools/challenge_to_soma.py 是否必须把它带进 soma_params.npz
"""
import os,sys,glob,warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "libs" / "v2d_infra"))
from soma_fk import fk as _soma_fk
import numpy as np, torch, pyarrow.parquet as pq
from denoise_v1 import col, meta, T1
ASSETS=str(pathlib.Path(__file__).resolve().parents[2] / "third_party/video_to_data/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/body_models/soma")
from soma import SOMALayer
layer=SOMALayer(data_root=ASSETS,identity_model_type="mhr",device="cpu",enable_procedural_transforms=False)
names=list(layer.rig_data["joint_names"])[1:]
HAND=[i for i,n in enumerate(names) if n in ("LeftHand","RightHand")]
FOOT=[i for i,n in enumerate(names) if n in ("LeftFoot","RightFoot")]
HEAD=[i for i,n in enumerate(names) if n=="Head"]

def fk(t, use_flex):   # 委托给 libs/v2d_infra/soma_fk（FK 唯一入口）
    return _soma_fk(t, flex=use_flex)

print(f"{'ep':>3} {'物体':<20}{'全关节p50':>10}{'全关节max':>10}{'手p50':>8}{'手max':>8}{'脚p50':>8}{'头max':>8}  (mm)")
agg={"all":[],"hand":[],"foot":[]}
for f1 in sorted(glob.glob(T1+"/data/chunk-000/*.parquet"))[:10]:
    t1=pq.read_table(f1)
    ei=int(np.asarray(col(t1,"episode_index")).ravel()[0])
    a=fk(t1,False); b=fk(t1,True)
    d=np.linalg.norm(a-b,axis=-1)*1000.0            # (T,77) mm
    dh=d[:,HAND].ravel(); df=d[:,FOOT].ravel(); dhead=d[:,HEAD].ravel()
    agg["all"].append(d.ravel()); agg["hand"].append(dh); agg["foot"].append(df)
    print(f"{ei:>3} {meta[ei]['object'][:20]:<20}{np.median(d):>10.1f}{d.max():>10.1f}"
          f"{np.median(dh):>8.1f}{dh.max():>8.1f}{np.median(df):>8.1f}{dhead.max():>8.1f}")

A=np.concatenate(agg["all"]); H=np.concatenate(agg["hand"]); F=np.concatenate(agg["foot"])
print(f"\n带/不带 bone_length_flexibles 的关节位置差 (10 条序列, mm)")
print(f"{'':<10}{'mean':>9}{'p50':>9}{'p95':>9}{'max':>9}")
for nm,v in (("全部关节",A),("手部",H),("脚部",F)):
    print(f"{nm:<10}{v.mean():>9.1f}{np.median(v):>9.1f}{np.percentile(v,95):>9.1f}{v.max():>9.1f}")
print(f"\n对照: 绝对位置偏移 vs 物体自身平移误差 18.7 mm、手-物接触距离(运动时)12 mm")

# ── B 段：t1-vs-t2 的误差本身会不会变 ──────────────────────────────
# bone_length_flexibles 在两档之间逐字节相同，两边同样偏移可能大部分相消。
# 直接算，不推理。
print(f"\n=== B 段：手/脚世界误差 |FK(t1)-FK(t2)| 带与不带 flex 的对比 ===")
print(f"{'':<12}{'不带-手':>9}{'带-手':>9}{'不带-脚':>9}{'带-脚':>9}  (mm, 均值)")
eh0=[];eh1=[];ef0=[];ef1=[]
for f1 in sorted(glob.glob(T1+"/data/chunk-000/*.parquet"))[:10]:
    f2=f1.replace("tier_1_multiview_caption","tier_2_synthetic_noise")
    t1=pq.read_table(f1); t2=pq.read_table(f2)
    for use,(eh,ef) in ((False,(eh0,ef0)),(True,(eh1,ef1))):
        a=fk(t1,use); b=fk(t2,use)
        e=np.linalg.norm(a-b,axis=-1)*1000.0
        eh.append(e[:,HAND].ravel()); ef.append(e[:,FOOT].ravel())
EH0=np.concatenate(eh0);EH1=np.concatenate(eh1);EF0=np.concatenate(ef0);EF1=np.concatenate(ef1)
print(f"{'均值':<12}{EH0.mean():>9.1f}{EH1.mean():>9.1f}{EF0.mean():>9.1f}{EF1.mean():>9.1f}")
print(f"{'中位':<12}{np.median(EH0):>9.1f}{np.median(EH1):>9.1f}{np.median(EF0):>9.1f}{np.median(EF1):>9.1f}")
print(f"{'p95':<12}{np.percentile(EH0,95):>9.1f}{np.percentile(EH1,95):>9.1f}"
      f"{np.percentile(EF0,95):>9.1f}{np.percentile(EF1,95):>9.1f}")
print(f"\n=> 若 B 段两列接近，说明 flex 在 t1-t2 差分里相消，已报的 37.0/71.3 仍有效；")
print(f"   但 A 段的绝对偏移仍然要求转换器把 flex 带进 soma_params.npz。")
