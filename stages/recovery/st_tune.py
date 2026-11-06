import pathlib
"""st_tune: 静止检测阈值重调，防过拟合协议。

⚠️ 读 Tier 1 真值生成新噪声与评测，仅用于分析。
overfit_audit P2 显示 st_rot 默认 23.5° 偏保守（30° 时真实 0.711、新噪声 0.723）。
协议：只在 3 份**另外的**全新噪声（种子与 audit 不重叠）上选 (st_rot, st_tr, st_win)，
选定后在真实 Tier 2 上**只评估一次**。
判据（写死）：真实 Tier 2 上提升 ≥ +0.005 且调参噪声上提升 ≥ +0.005 → 采纳。
"""
import os,sys,glob,json,itertools,warnings; warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "libs" / "v2d_infra")); sys.path.insert(0, str(HERE))
import numpy as np, pyarrow.parquet as pq, add_metrics as A
from soma_fk import col, R_CV
from paths import TIER2
from denoise_v1 import meta, T1
from denoise_v4 import to_soma
from v51 import denoise_v51, surface_tops, stable_ups, DEFAULT
from noise_gen import corrupt

E = []
for f1 in sorted(glob.glob(T1 + "/data/chunk-000/*.parquet")):
    f2 = f1.replace("tier_1_multiview_caption", "tier_2_synthetic_noise")
    t1, t2 = pq.read_table(f1), pq.read_table(f2)
    ei = int(np.asarray(col(t1, "episode_index")).ravel()[0])
    vis = np.asarray(col(t1, "observation.object.visible")).astype(bool).ravel()
    p1 = to_soma(col(t1, "observation.object.pose")[vis]); p2 = to_soma(col(t2, "observation.object.pose")[vis])
    g = json.load(open(TIER2 / meta[ei]["ground_plane"]))["plane"]
    s_ = np.linalg.norm(g[:3]); n = R_CV @ (np.array(g[:3]) / s_); d = g[3] / s_
    if np.median(p2[:, :3] @ n + d) < 0: n, d = -n, -d
    E.append(dict(ei=ei, p1=p1, p2=p2, n=n, d=d, v=A.sample_mesh_vertices(os.path.join(T1, meta[ei]["mesh"]), n=500, seed=ei),
                  tops=surface_tops(TIER2 / meta[ei]["support_surface"]), ups=stable_ups(str(TIER2 / meta[ei]["mesh"])),
                  tune=[corrupt(p1, 7_000_000 + 1000 * k + ei) for k in range(3)]))
score = lambda P, which: np.mean([A.auc_from_add(A.add_per_frame(e["v"], e["p1"],
        denoise_v51(q, e["n"], e["d"], e["tops"], e["ups"], P)))[0] for e in E for q in which(e)])
tune = lambda e: e["tune"]; real = lambda e: [e["p2"]]

base_t = score({}, tune)
grid = list(itertools.product([23.5, 26, 28, 30, 33, 36], [0.06, 0.08, 0.10]))
res = {(r, t): score({"st_rot": r, "st_tr": t}, tune) for r, t in grid}
print(f"调参噪声（3 份）上 v5.1 均值，默认 (23.5°, 0.06 m, 91 帧) = {base_t:.3f}")
print("   st_rot\\st_tr " + "".join(f"{t:>8}" for t in (0.06, 0.08, 0.10)))
for r in (23.5, 26, 28, 30, 33, 36):
    print(f"   {r:>12} " + "".join(f"{res[(r, t)]:>8.3f}" for t in (0.06, 0.08, 0.10)))
br, bt = max(res, key=res.get)
wins = {w: score({"st_rot": br, "st_tr": bt, "st_win": w}, tune) for w in (61, 91, 121)}
print(f"   在 ({br}°, {bt} m) 下扫窗口: " + "  ".join(f"{w}:{s:.3f}" for w, s in wins.items()))
bw = max(wins, key=wins.get); best = {"st_rot": br, "st_tr": bt, "st_win": bw}
gain_t = wins[bw] - base_t
real_def, real_best = score({}, real), score(best, real)
print(f"\n选定 {best}：调参噪声 {wins[bw]:.3f}（{gain_t:+.3f}）")
print(f"真实 Tier 2（只评估这一次）：默认 {real_def:.3f} → 选定 {real_best:.3f}（{real_best - real_def:+.3f}）")
print("判据：" + ("采纳" if real_best - real_def >= 0.005 and gain_t >= 0.005 else "不采纳，保持默认"))
