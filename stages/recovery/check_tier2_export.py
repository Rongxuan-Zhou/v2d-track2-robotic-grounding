import pathlib
"""核对 export_tier2.py 的落盘产物：在可见帧上与 Tier 1 真值比 ADD-AUC，应复现 v5.2 0.734 / v5.3 0.748。
用法：python stages/recovery/check_tier2_export.py [tier2_v52|tier2_v53]

⚠️ 读 Tier 1 真值，仅用于分析；与导出脚本分开，导出本身不碰 Tier 1。
同时检查：坐标系往返（产物是 OpenCV 系）、全长帧数、第 0 帧与原观测一致（d(0)≈0）。
"""
import os,sys,glob,warnings; warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "libs" / "v2d_infra")); sys.path.insert(0, str(HERE))
import numpy as np, pyarrow.parquet as pq, add_metrics as A
from soma_fk import col
from paths import RUNS
from denoise_v1 import meta, T1

SUB = sys.argv[1] if len(sys.argv) > 1 else "tier2_v53"
EXPECT = {"tier2_v52": 0.734, "tier2_v53": 0.748, "tier2_v54": "净室版，首次评测"}[SUB]
# 第 0 帧 d≈0，v5.2 不应改动它（≤5 mm）。v5.3 的不穿地约束会抬起第 0 帧就低于地面 ε 以上的物体
# （实测只有 ep17 交通锥，观测低于地面 23.5 mm → 抬 13.5 mm），所以放宽到 20 mm 并打印实际量。
F0_TOL = {"tier2_v52": 0.005, "tier2_v53": 0.020, "tier2_v54": 0.020}[SUB]
f0 = []
aucs, raw = [], []
for f1 in sorted(glob.glob(T1 + "/data/chunk-000/*.parquet")):
    t1 = pq.read_table(f1); f2 = f1.replace("tier_1_multiview_caption", "tier_2_synthetic_noise"); t2 = pq.read_table(f2)
    ei = int(np.asarray(col(t1, "episode_index")).ravel()[0])
    z = np.load(RUNS / "recovery" / SUB / f"episode_{ei:06d}.npz")
    q = z["object_pose"]; vis = z["visible"]
    assert len(q) == t1.num_rows, f"ep{ei} 帧数不符"
    p1 = col(t1, "observation.object.pose")[vis]; p2 = col(t2, "observation.object.pose")[vis]
    v = A.sample_mesh_vertices(os.path.join(T1, meta[ei]["mesh"]), n=500, seed=ei)
    aucs.append(A.auc_from_add(A.add_per_frame(v, p1, q[vis]))[0]); raw.append(A.auc_from_add(A.add_per_frame(v, p1, p2))[0])
    dev = np.linalg.norm(q[vis][0, :3] - p2[0, :3]); f0.append((dev, ei))
    assert dev < F0_TOL, f"ep{ei} 第 0 帧平移偏离观测 {1000*dev:.1f} mm > {1000*F0_TOL:.0f} mm"
print(f"{SUB}（OpenCV 系，直接与 Tier 1 parquet 比）: raw {np.mean(raw):.3f} → {np.mean(aucs):.3f}（应为 0.423 → {EXPECT}）")
print("第 0 帧平移偏离 >1 mm 的序列:", [(e, round(1000 * v, 1)) for v, e in sorted(f0, reverse=True) if v > 0.001] or "无")
