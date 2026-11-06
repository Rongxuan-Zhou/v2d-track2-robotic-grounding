import pathlib
"""bone_flex_physics: 带 flex 与不带 flex，哪个才是出题方的人体？

bone_flex_probe 证明了 flex 会让关节偏 13.5 mm。但"带上更准"不能想当然：官方
read_soma 就没带它。如果出题方算评测用的人手时也没带，我们带上反而离参考更远。
本脚本用两组独立证据裁决，只读 Tier 1（分析用途，不进 Tier 2/3 管线）：

A. 复现出题方的 foot_plane_dist_stats。每条序列的 ground_plane.json 里有它，
   是出题方用他们的人体算的脚离地距离；平面本身是 RANSAC 拟场景点云得到的，
   不依赖人体。哪个版本能复现这组数，出题方用的就是哪个。
   脚点的确切定义未公开，所以试几种候选定义，看哪种在哪个版本下对得最准。
B. 物理一致性：物体在动时一定有手碰着它，手-物最近距离应当接近 0。
"""
import os,sys,json,glob,warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "libs" / "v2d_infra"))
import numpy as np, pyarrow.parquet as pq, add_metrics as A
from soma_fk import fk, joint_names, col, R_CV
from paths import TIER1
from denoise_v3 import to_soma
from static_window import gt_still

T1=str(TIER1)
meta={json.loads(l)["episode_index"]:json.loads(l) for l in open(T1+"/meta/episodes_metadata.jsonl")}
names=joint_names()
idx=lambda ns:[names.index(n) for n in ns]
FOOT_ALL=idx(["LeftFoot","LeftToeBase","LeftToeEnd","RightFoot","RightToeBase","RightToeEnd"])
ANKLE=idx(["LeftFoot","RightFoot"])
TOE=idx(["LeftToeBase","LeftToeEnd","RightToeBase","RightToeEnd"])
HAND=[i for i,n in enumerate(names) if (n.endswith("End") and "Hand" in n) or n in ("LeftHand","RightHand")]

# 候选脚点定义：(名字, 取哪些关节, 每帧如何汇总)
DEFS=[("全脚点逐帧最低",FOOT_ALL,"min"),("脚踝逐帧最低",ANKLE,"min"),
      ("脚趾逐帧最低",TOE,"min"),("全脚点全部入池",FOOT_ALL,"pool"),("脚踝全部入池",ANKLE,"pool")]

off={d[0]:{0:[],1:[]} for d in DEFS}         # 每种定义下 |我们的中位 - 官方中位|
ours={d[0]:{0:[],1:[]} for d in DEFS}; offi=[]
hand={0:[],1:[]}; hand_near={0:[],1:[]}
for f1 in sorted(glob.glob(T1+"/data/chunk-000/*.parquet")):
    t1=pq.read_table(f1)
    ei=int(np.asarray(col(t1,"episode_index")).ravel()[0])
    g=json.load(open(os.path.join(T1,meta[ei]["ground_plane"])))
    n=np.array(g["plane"][:3]); d=g["plane"][3]; s=np.linalg.norm(n); n,d=n/s,d/s
    offi.append(g["foot_plane_dist_stats"]["median"])
    vis=np.asarray(col(t1,"observation.object.visible")).astype(bool).ravel()
    p1=col(t1,"observation.object.pose")[vis]
    ov=A.transform_vertices(A.sample_mesh_vertices(os.path.join(T1,meta[ei]["mesh"]),n=300,seed=ei),to_soma(p1))
    moving=~gt_still(p1)
    for fl in (0,1):
        J=fk(t1,flex=bool(fl))
        Jcv=J@R_CV.T                                   # 平面在 OpenCV 系
        for nm,ji,how in DEFS:
            h=np.abs(Jcv[:,ji,:]@n+d)
            v=h.min(axis=1) if how=="min" else h.ravel()
            m=float(np.median(v)); ours[nm][fl].append(m)
            off[nm][fl].append(abs(m-offi[-1]))
        dist=np.linalg.norm(J[vis][:,HAND,None,:]-ov[:,None,:,:],axis=-1).min(axis=(1,2))
        if moving.any():
            hand[fl].append(np.median(dist[moving])); hand_near[fl].append((dist[moving]<0.02).mean())

offi=np.array(offi)
print("A. 复现官方 foot_plane_dist_stats.median（30 条序列，官方中位数的中位 "
      f"{1000*np.median(offi):.1f} mm）")
print(f"{'候选定义':<14}{'不带flex 偏差':>13}{'带flex 偏差':>12}{'不带 相关':>10}{'带 相关':>9}  (偏差=|我们-官方| 的中位, mm)")
for nm,_,_ in DEFS:
    e0,e1=np.median(off[nm][0])*1000,np.median(off[nm][1])*1000
    r0=np.corrcoef(ours[nm][0],offi)[0,1]; r1=np.corrcoef(ours[nm][1],offi)[0,1]
    print(f"{nm:<14}{e0:>13.1f}{e1:>12.1f}{r0:>10.3f}{r1:>9.3f}")

print("\nB. 物体在动时的手-物最近距离（越接近 0 越符合物理）")
for fl,tag in ((0,"不带 flex"),(1,"带 flex")):
    print(f"  {tag:<8} 中位 {1000*np.median(hand[fl]):5.1f} mm   <2cm 的帧占比 {100*np.mean(hand_near[fl]):4.0f}%")
