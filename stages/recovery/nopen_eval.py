import pathlib
"""nopen_eval: v5.3 不穿地约束的评估（lift / rot / rot_s 对照 v5.2）。

⚠️ 读 Tier 1 真值做评测、自检与生成全新噪声，仅用于分析。修正本身只用 Tier 2 的地面与网格。
判据（写死）：满足全部才采纳，取真实 AUC 最高者：
  ① 穿透 >20 mm 帧占比 ≤ Tier 1 水平 + 0.5%  ② 真实 AUC ≥ 0.731
  ③ 新噪声 AUC ≥ 同噪声 v5.2 − 0.003          ④ 抖动 ≤ 1.2 × v5.2
ε 固定 10 mm；5/20 mm 只报敏感性，不用于选择。
"""
import os,sys,glob,json,warnings; warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "libs" / "v2d_infra")); sys.path.insert(0, str(HERE))
import numpy as np, pyarrow.parquet as pq, add_metrics as A
from scipy.spatial.transform import Rotation as Rot
from soma_fk import col, R_CV
from paths import TIER2
from denoise_v1 import meta, T1
from denoise_v4 import to_soma
from v51 import V52, denoise_v51, surface_tops, stable_ups
from nopen import enforce, hull_vertices, min_height
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
    E.append(dict(ei=ei, obj=meta[ei]["object"][:18], p1=p1, p2=p2, n=n, d=d,
                  v=A.sample_mesh_vertices(os.path.join(T1, meta[ei]["mesh"]), n=500, seed=ei),
                  V=hull_vertices(str(TIER2 / meta[ei]["mesh"])),
                  tops=surface_tops(TIER2 / meta[ei]["support_surface"]), ups=stable_ups(str(TIER2 / meta[ei]["mesh"]))))
auc = lambda e, q: A.auc_from_add(A.add_per_frame(e["v"], e["p1"], q))[0]
v52 = lambda e, p: denoise_v51(p, e["n"], e["d"], e["tops"], e["ups"], V52)


def jitter(p):
    Rm = Rot.from_quat(p[:, [4, 5, 6, 3]])
    w = (Rm[1:] * Rm[:-1].inv()).as_rotvec()
    return np.degrees(np.median(np.linalg.norm(np.diff(w, axis=0), axis=1)))


MODES = ["v5.2", "lift", "rot", "rot_s"]
def variants(e, p):
    b = v52(e, p); out = {"v5.2": b}
    for m in MODES[1:]: out[m] = enforce(b, e["V"], e["n"], e["d"], mode=m)
    return out

# ── 真实 Tier 2 ──
R = {m: {"auc": [], "mh": [], "jit": []} for m in MODES + ["Tier1"]}
per = []
for e in E:
    vs = variants(e, e["p2"])
    for m, q in vs.items():
        R[m]["auc"].append(auc(e, q)); R[m]["mh"].append(min_height(q, e["V"], e["n"], e["d"])); R[m]["jit"].append(jitter(q))
    R["Tier1"]["mh"].append(min_height(e["p1"], e["V"], e["n"], e["d"])); R["Tier1"]["jit"].append(jitter(e["p1"]))
    per.append((e["ei"], e["obj"], *[R[m]["auc"][-1] for m in MODES], 1000 * R["v5.2"]["mh"][-1].min(), 1000 * R["rot_s"]["mh"][-1].min()))

def pen(m):
    mh = np.concatenate(R[m]["mh"]); return 100 * (mh < -0.02).mean(), 1000 * mh.min()
t1p, t1w = pen("Tier1")
print("真实 Tier 2（30 条）")
print(f"{'':<8}{'AUC':>7}{'穿透>2cm帧%':>12}{'最深mm':>9}{'抖动°':>8}")
print(f"{'Tier1':<8}{'':>7}{t1p:>12.2f}{t1w:>9.0f}{np.median(R['Tier1']['jit']):>8.3f}")
for m in MODES:
    pp, pw = pen(m); print(f"{m:<8}{np.mean(R[m]['auc']):>7.3f}{pp:>12.2f}{pw:>9.0f}{np.median(R[m]['jit']):>8.3f}")

# ── Tier 1 自检：约束加在真值上应几乎不动 ──
chg = []; mod = []
for e in E:
    q = enforce(e["p1"], e["V"], e["n"], e["d"], mode="rot_s")
    chg.append(1000 * A.add_per_frame(e["v"], e["p1"], q).mean()); mod.append((np.abs(q - e["p1"]).max(1) > 1e-6).mean())
print(f"\nTier 1 自检（rot_s 加在真值上）：平均 ADD 改动 {np.mean(chg):.2f} mm，最大 {np.max(chg):.2f} mm；被改动帧占比 {100*np.mean(mod):.1f}%")

# ── ε 敏感性（只报告）──
print("\nε 敏感性（rot_s，真实 AUC / 穿透>2cm帧%）: " + "  ".join(
    f"ε={eps*1000:.0f}mm:{np.mean([auc(e, enforce(v52(e, e['p2']), e['V'], e['n'], e['d'], 'rot_s', eps=eps)) for e in E]):.3f}"
    for eps in (0.005, 0.010, 0.020)))

# ── 全新噪声 5 份 ──
F = {m: [] for m in MODES}
for s in range(5):
    acc = {m: [] for m in MODES}
    for e in E:
        for m, q in variants(e, corrupt(e["p1"], 9_000_000 + 1000 * s + e["ei"])).items(): acc[m].append(auc(e, q))
    for m in MODES: F[m].append(np.mean(acc[m]))
    print(f"新噪声种子 {s}: " + "  ".join(f"{m} {np.mean(acc[m]):.3f}" for m in MODES))
print("5 份均值: " + "  ".join(f"{m} {np.mean(F[m]):.3f}" for m in MODES))

print(f"\n逐序列（按 v5.2 最深穿透排序，只列最深 <−20 mm 的）")
print(f"{'ep':>2} {'物体':<18}" + "".join(f"{m:>8}" for m in MODES) + f"{'v5.2最深':>9}{'rot_s最深':>10}")
for r in sorted(per, key=lambda r: r[6]):
    if r[6] < -20: print(f"{r[0]:>2} {r[1]:<18}" + "".join(f"{x:>8.3f}" for x in r[2:6]) + f"{r[6]:>9.0f}{r[7]:>10.0f}")

# ── 判据 ──
ok = []
jb = np.median(R["v5.2"]["jit"]); fb = np.mean(F["v5.2"])
for m in MODES[1:]:
    c1 = pen(m)[0] <= t1p + 0.5; c2 = np.mean(R[m]["auc"]) >= 0.731; c3 = np.mean(F[m]) >= fb - 0.003; c4 = np.median(R[m]["jit"]) <= 1.2 * jb
    print(f"判据 {m:<6}: ①穿透 {'✓' if c1 else '✗'} ②真实AUC {'✓' if c2 else '✗'} ③新噪声 {'✓' if c3 else '✗'} ④抖动 {'✓' if c4 else '✗'}")
    if c1 and c2 and c3 and c4: ok.append(m)
best = max(ok, key=lambda m: np.mean(R[m]["auc"])) if ok else None
print("结论：" + (f"采纳 {best} 为 v5.3" if best else "都不满足，保持 v5.2"))
