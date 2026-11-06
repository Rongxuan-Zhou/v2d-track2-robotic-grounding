import pathlib
"""Can static segments be detected from the CORRUPTED trace alone?
Inference time only has Tier 2. The drift is slow (8 s correlation), real
motion is faster, so a per-frame velocity threshold should still separate them.
Compare T2-detected segments against T1-detected ground truth.
"""
import sys, os, pathlib, json, glob
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "libs" / "v2d_infra"))
import numpy as np, pyarrow.parquet as pq

ROOT=str(pathlib.Path(__file__).resolve().parents[2] / "data/challenge/track_2"); T1=ROOT+"/tier_1_multiview_caption"
def col(t,c):
    v=t.column(c).to_numpy(zero_copy_only=False)
    return np.stack(v) if v.dtype==object else np.asarray(v)
meta={json.loads(l)["episode_index"]:json.loads(l) for l in open(T1+"/meta/episodes_metadata.jsonl")}

def still_mask(p, tr_thr, rot_thr):
    dt=np.r_[0,np.linalg.norm(np.diff(p[:,:3],axis=0),axis=1)]
    q=p[:,3:7]/(np.linalg.norm(p[:,3:7],axis=1,keepdims=True)+1e-12)
    dq=np.r_[0,2*np.arccos(np.clip(np.abs((q[:-1]*q[1:]).sum(1)),0,1))]
    # 中值滤波抑制单帧毛刺
    k=5; pad=k//2
    def med(x):
        xp=np.pad(x,pad,mode="edge"); return np.median(np.lib.stride_tricks.sliding_window_view(xp,k),axis=-1)
    return (med(dt)<tr_thr)&(med(dq)<rot_thr)

TR, ROT = 0.0015, np.radians(0.25)
tp=fp=fn=tn=0; per=[]
for f1 in sorted(glob.glob(T1+"/data/chunk-000/*.parquet")):
    f2=f1.replace("tier_1_multiview_caption","tier_2_synthetic_noise")
    t1,t2=pq.read_table(f1),pq.read_table(f2)
    ei=int(np.asarray(col(t1,"episode_index")).ravel()[0])
    vis=np.asarray(col(t1,"observation.object.visible")).astype(bool).ravel()
    p1,p2=col(t1,"observation.object.pose")[vis],col(t2,"observation.object.pose")[vis]
    g=still_mask(p1,TR,ROT); d=still_mask(p2,TR,ROT)
    tp+=int((g&d).sum()); fp+=int((~g&d).sum()); fn+=int((g&~d).sum()); tn+=int((~g&~d).sum())
    if g.sum()>0: per.append((ei,meta[ei]["object"],100*g.mean(),100*d.mean(),
                              100*(g&d).sum()/max(d.sum(),1), 100*(g&d).sum()/max(g.sum(),1)))
print(f"{'ep':>3} {'物体':<21} {'T1静止%':>7} {'T2检出%':>7} {'精确率':>6} {'召回率':>6}")
for r in sorted(per,key=lambda r:-r[2])[:12]:
    print(f"{r[0]:>3} {r[1]:<21} {r[2]:>6.0f}% {r[3]:>6.0f}% {r[4]:>5.0f}% {r[5]:>5.0f}%")
prec=100*tp/max(tp+fp,1); rec=100*tp/max(tp+fn,1)
print(f"\n全局: 精确率 {prec:.1f}%  召回率 {rec:.1f}%  F1 {2*prec*rec/max(prec+rec,1e-9):.1f}")
print(f"       TP={tp} FP={fp} FN={fn} TN={tn}")
