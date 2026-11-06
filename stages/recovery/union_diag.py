import pathlib
"""union_diag: 逐序列拆开 geom / contact / union，找出 union 相对 geom 崩掉的那批。

N5 修正后的观察：union 的 AUC 中位数 0.662 明显高于 geom 的 0.590，但均值反而低
(0.580 vs 0.593)。说明接触信息对多数序列有效，是少数序列被拖垮。积分法对错误锚点
没有容错（一个错锚污染其后全部累积），所以尾部损失极重。

本脚本逐序列输出 AUC 与几个可能的解释变量，目的是判断"崩掉的序列"是否可事先识别。
若可识别，union 就是白捡的 +0.07 中位数。
"""
import os,sys,json,glob,warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "libs" / "v2d_infra"))
import numpy as np, torch, pyarrow.parquet as pq, add_metrics as A
from denoise_v1 import window_rot, window_tr, nq, qmul, qinv, col, meta, T1
from denoise_v3 import fk, to_soma, integrate, H
from static_window import gt_still   # 真静止的规范定义：逐帧位移<1.5mm 且转角<0.25°

rows=[]
for f1 in sorted(glob.glob(T1+"/data/chunk-000/*.parquet")):
    f2=f1.replace("tier_1_multiview_caption","tier_2_synthetic_noise")
    t1,t2=pq.read_table(f1),pq.read_table(f2)
    ei=int(np.asarray(col(t1,"episode_index")).ravel()[0])
    vis=np.asarray(col(t1,"observation.object.visible")).astype(bool).ravel()
    p1=col(t1,"observation.object.pose")[vis]; p2=col(t2,"observation.object.pose")[vis]
    v=A.sample_mesh_vertices(os.path.join(T1,meta[ei]["mesh"]),n=500,seed=ei)
    J=fk(t2)[vis]
    ov=A.transform_vertices(A.sample_mesh_vertices(os.path.join(T1,meta[ei]["mesh"]),n=200,seed=ei), to_soma(p2))
    d=np.linalg.norm(J[:,H,None,:]-ov[:,None,:,:],axis=-1).min(axis=(1,2))
    geom=(window_rot(p2,91)<23.5)&(window_tr(p2,91)<0.06)
    cont=d>0.15
    auc=lambda m: A.auc_from_add(A.add_per_frame(v,p1,integrate(p2,m)))[0]
    raw=A.auc_from_add(A.add_per_frame(v,p1,p2))[0]

    true_static=gt_still(p1)
    # contact 相对真值的精确率：判静止且确实静止
    prec_c = float(true_static[cont].mean()) if cont.sum() else float("nan")
    prec_g = float(true_static[geom].mean()) if geom.sum() else float("nan")

    rows.append(dict(ep=ei, obj=meta[ei]["object"][:20],
        radius=float(np.linalg.norm(v,axis=1).max()),
        raw=raw, geom=auc(geom), cont=auc(cont), union=auc(geom|cont),
        f_static=float(true_static.mean()), f_geom=float(geom.mean()),
        f_cont=float(cont.mean()), f_union=float((geom|cont).mean()),
        prec_g=prec_g, prec_c=prec_c,
        disp=float(np.linalg.norm(p1[:,:3]-p1[0,:3],axis=1).max())))

rows.sort(key=lambda r: r["union"]-r["geom"])
print(f"{'ep':>3} {'物体':<20}{'半径':>6}{'raw':>7}{'geom':>7}{'cont':>7}{'union':>7}{'Δu-g':>7}"
      f"{'真静%':>7}{'geom%':>7}{'cont%':>7}{'精g':>6}{'精c':>6}")
for r in rows:
    print(f"{r['ep']:>3} {r['obj']:<20}{r['radius']:>6.2f}{r['raw']:>7.3f}{r['geom']:>7.3f}"
          f"{r['cont']:>7.3f}{r['union']:>7.3f}{r['union']-r['geom']:>+7.3f}"
          f"{r['f_static']*100:>6.0f}%{r['f_geom']*100:>6.0f}%{r['f_cont']*100:>6.0f}%"
          f"{r['prec_g']*100:>5.0f}%{r['prec_c']*100:>5.0f}%")

import numpy as _np
bad=[r for r in rows if r["union"]<r["geom"]-0.01]
good=[r for r in rows if r["union"]>r["geom"]+0.01]
print(f"\n变差 {len(bad)} 条 / 变好 {len(good)} 条 / 持平 {len(rows)-len(bad)-len(good)} 条")
for nm,grp in (("变差",bad),("变好",good)):
    if not grp: continue
    print(f"  {nm}: 半径中位 {_np.median([r['radius'] for r in grp]):.2f}  "
          f"真静止 {_np.median([r['f_static'] for r in grp])*100:.0f}%  "
          f"cont覆盖 {_np.median([r['f_cont'] for r in grp])*100:.0f}%  "
          f"cont精确率 {_np.median([r['prec_c'] for r in grp])*100:.0f}%  "
          f"geom精确率 {_np.median([r['prec_g'] for r in grp])*100:.0f}%")

d_ug=_np.array([r["union"]-r["geom"] for r in rows])
for k in ("radius","f_static","f_cont","f_geom","prec_c","prec_g","disp","raw"):
    x=_np.array([r[k] for r in rows])
    print(f"  Pearson(Δu-g, {k:<8}) = {_np.corrcoef(x,d_ug)[0,1]:+.3f}")
