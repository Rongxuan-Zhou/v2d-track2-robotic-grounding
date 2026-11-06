import pathlib
"""support_bridge_v2: 让支撑面约束支持"翻面"与"换高度"。

v5（support_bridge.py 的 R_real）假设物体每次都放回同一个面、同一高度，参考取第 0 帧；
翻面或换支撑面的静止段被门限拦下、不修。它到上限 O_3dof 还差 0.043。

本版每个静止段从一组候选假设里挑一个：
  面   ：物体凸包的所有稳定放置姿态（trimesh compute_stable_poses），外加第 0 帧参考
  高度 ：地面 0 + Tier 2 support_surface 里每个几何体的顶面（相对地面平面的高度，R1 检查已验证）
  中心高度 = 支撑面高度 + 沿"上"方向最低点到原点的距离（与偏航无关）
每个假设蕴含一个漂移（倾斜 θ、高度 Δh）。只保留在腐蚀上限内的，取归一化漂移最小者；
与第二名（非等价假设）拉不开差距时跳过——选错比不修更糟，错锚会经布朗桥污染两侧运动段。

只用 Tier 2 观测、Tier 2 的 support_surface / ground_plane 与物体网格。Tier 1 只用于诊断与评测。

倾斜修正的两种写法（同一目标 D·n = z*，差别只在绕 n 的偏航怎么处理）：
  min  ：C = 把 D·n 转到 z* 的最小旋转，左乘 D。会顺带改动偏航，倾斜修正越大改得越多。
  st   ：swing-twist 分解 D = Swing·Twist(绕 n)，保留原 Twist（偏航），只把 Swing 换成 n→z*。
第一轮只有 min，vacuum −0.17 / monitor −0.11 且选择全对，怀疑是 min 带偏了偏航，故加 st 对照。

第二轮（看过结果后的修改，30 条序列，有过拟合风险，单独立判据）：
  st 与 min 无差别（真值 0.700 vs 0.701），假设被否。vacuum 真值版 0.632 不掉，R_multi 选择全对却掉到 0.458
  ⇒ 损失来自"歧义跳过"丢掉了 v5 原本修过的段（ref 与相近稳定姿态打平）。
  分层（R_tier）：先按 v5 用 ref（v5 门限 35°/5 cm）；ref 过不了门限时才退到稳定姿态×多高度（紧门限），
  这些新假设之间歧义仍跳过。保证 v5 修过的段一个不丢。
  判据：R_tier ≥ R_multi、选错 0、vacuum 与 monitor 不低于 v5。

预设判据（第一轮）：R_multi ≥ 0.678（R_real + 0.02）且被修正段的选错率 < 5% → 采纳为 v5.1；否则保持 v5。
"""
import os,sys,re,glob,json,warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "libs" / "v2d_infra"))
import numpy as np, pyarrow.parquet as pq, add_metrics as A, trimesh
from soma_fk import col, R_CV
from paths import TIER2
from denoise_v1 import window_rot, window_tr, meta, T1
from denoise_v4 import to_soma, segments
from support_bridge import integrate, apply, correct

M = lambda q: A.quat_wxyz_to_matrix(q)
TILT_GATE, H_GATE = np.radians(33), 0.035          # 腐蚀上限 30° / 30 mm 各加一点余量
TILT_CAP, H_CAP, MARGIN = np.radians(30), 0.03, 0.5
EQ_ANG, EQ_H = np.radians(5), 0.01                  # 两个假设在此范围内视为同一个


def surface_tops(usda: pathlib.Path) -> list[float]:
    """解析 support usda（只有 Cylinder / Cube，无旋转）→ 各几何体顶面高度。"""
    tops = []
    for blk in re.split(r"\bdef (?=Cylinder|Cube)", usda.read_text())[1:]:
        z = float(re.search(r"xformOp:translate = \(([^)]*)\)", blk).group(1).split(",")[2])
        if blk.startswith("Cylinder"):
            tops.append(z + float(re.search(r"double height = ([\d.eE+-]+)", blk).group(1)) / 2)
        else:
            size = float(re.search(r"double size = ([\d.eE+-]+)", blk).group(1))
            sc = re.search(r"xformOp:scale = \(([^)]*)\)", blk)
            sz = float(sc.group(1).split(",")[2]) if sc else 1.0
            tops.append(z + size * sz / 2)
    return tops


def stable_ups(mesh_path):
    """凸包的稳定放置姿态 → [(物体系里朝上的单位向量 b, 原点到最低点的距离 δ)]。"""
    m = trimesh.load(mesh_path, force="scene")
    m = m.dump(concatenate=True) if isinstance(m, trimesh.Scene) else m
    hull = m.convex_hull
    T, _ = hull.compute_stable_poses()
    out = []
    for Tk in T:
        b = Tk[:3, :3].T @ np.array([0, 0, 1.0])
        out.append((b, -float((hull.vertices @ b).min())))
    return out, hull.vertices


def rot_between(a, b):
    v = np.cross(a, b); c = float(a @ b); s = np.linalg.norm(v)
    if s < 1e-12: return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + K + K @ K * ((1 - c) / s**2)


def swing_twist_full(DRm, Dxm, n, z_img, h_d):
    """保留 DRm 绕 n 的扭转，只把摆动换成 n→z_img；平移只改沿 n 的分量。返回 correct() 的 full 目标。"""
    tw = rot_between(n, DRm @ n).T @ DRm            # 绕 n 的扭转部分
    R_new = rot_between(n, z_img) @ tw
    x_new = Dxm + (h_d - n @ Dxm) * n
    return (R_new, x_new)


def same(h1, h2):
    return np.arccos(np.clip(h1[0] @ h2[0], -1, 1)) < EQ_ANG and abs(h1[1] - h2[1]) < EQ_H


if __name__ == "__main__":
    res = {k: [] for k in ("raw", "v1", "O_3dof_min", "O_3dof_st", "R_real", "R_multi", "R_multi_noref", "R_multi_st", "R_tier")}
    diag = {k: {"seg": 0, "fix": 0, "ok": 0, "bad": 0, "skip_amb": 0, "tilt_err": [], "h_err": []}
            for k in ("R_real", "R_multi", "R_multi_noref", "R_multi_st", "R_tier")}
    per_ep = []
    for f1 in sorted(glob.glob(T1 + "/data/chunk-000/*.parquet")):
        f2 = f1.replace("tier_1_multiview_caption", "tier_2_synthetic_noise")
        t1, t2 = pq.read_table(f1), pq.read_table(f2)
        ei = int(np.asarray(col(t1, "episode_index")).ravel()[0])
        vis = np.asarray(col(t1, "observation.object.visible")).astype(bool).ravel()
        p1 = to_soma(col(t1, "observation.object.pose")[vis]); p2 = to_soma(col(t2, "observation.object.pose")[vis])
        v = A.sample_mesh_vertices(os.path.join(T1, meta[ei]["mesh"]), n=500, seed=ei)
        auc = lambda q: A.auc_from_add(A.add_per_frame(v, p1, q))[0]

        g = json.load(open(TIER2 / meta[ei]["ground_plane"]))["plane"]
        s_ = np.linalg.norm(g[:3]); n = R_CV @ (np.array(g[:3]) / s_); d = g[3] / s_
        if np.median(p2[:, :3] @ n + d) < 0: n, d = -n, -d         # n 与 d 一起翻，保证"地面以上为正"
        height = lambda x: x @ n + d

        tops = [0.0] + surface_tops(TIER2 / meta[ei]["support_surface"])
        ups, _ = stable_ups(TIER2 / meta[ei]["mesh"])
        R0 = M(p2[:1, 3:7])[0]
        ref = (R0.T @ n, float(height(p2[0, :3])), "ref")
        poses = [(b, H + dl, f"face{k}@{H:.2f}") for k, (b, dl) in enumerate(ups) for H in tops]

        st = (window_rot(p2, 91) < 23.5) & (window_tr(p2, 91) < 0.06)
        segs = segments(st, 10)
        DR, Dx = integrate(p2, st)
        res["raw"].append(auc(p2)); res["v1"].append(auc(apply(p2, DR, Dx)))

        def make_target(hyps, tag, tilt_gate=TILT_GATE, h_gate=H_GATE, mode="min", tiered=False):
            def target(s, e):
                m = (s + e) // 2; Robs = M(p2[m:m+1, 3:7])[0]; ho = height(p2[m, :3])
                if tiered:                                     # 先试 v5：只看 ref、v5 门限
                    z_img = Robs @ ref[0]; dh = ho - ref[1]
                    if np.arccos(np.clip(z_img @ n, -1, 1)) < np.radians(35) and abs(dh) < 0.05:
                        cand = [(0.0, ref, z_img, dh)]
                        return pick(cand, tag, m, Robs, ho, mode)
                cand = []
                for h in hyps:
                    z_img = Robs @ h[0]
                    th = np.arccos(np.clip(z_img @ n, -1, 1)); dh = ho - h[1]
                    if th < tilt_gate and abs(dh) < h_gate:
                        cand.append((th / TILT_CAP + abs(dh) / H_CAP, h, z_img, dh))
                cand.sort(key=lambda c: c[0])
                return pick(cand, tag, m, Robs, ho, mode)
            return target

        def pick(cand, tag, m, Robs, ho, mode):
                D = diag[tag]; D["seg"] += 1
                if not cand: return (None, None, None)
                best = cand[0]
                rival = next((c for c in cand[1:] if not same(c[1], best[1])), None)
                if rival is not None and rival[0] - best[0] < MARGIN:
                    D["skip_amb"] += 1; return (None, None, None)
                # 诊断（真值）：选中的面/高度是否与真实一致
                Rt = M(p1[m:m+1, 3:7])[0]; bt = Rt.T @ n; ht = height(p1[m, :3])
                ok = np.arccos(np.clip(best[1][0] @ bt, -1, 1)) < EQ_ANG and abs(best[1][1] - ht) < EQ_H
                D["fix"] += 1; D["ok" if ok else "bad"] += 1
                true_zimg = (Robs @ Rt.T) @ n                             # 真实 D·n
                D["tilt_err"].append(np.degrees(np.arccos(np.clip(best[2] @ true_zimg, -1, 1))))
                D["h_err"].append(1000 * abs(best[3] - (ho - ht)))
                if mode == "st":
                    return (None, None, swing_twist_full(DR[m], Dx[m], n, best[2], best[3]))
                return (best[2], best[3], None)

        def oracle_target(mode):
            def target(s, e):
                m = (s + e) // 2
                Rt = M(p1[m:m+1, 3:7])[0]; Ro = M(p2[m:m+1, 3:7])[0]
                z_img = (Ro @ Rt.T) @ n; h_d = height(p2[m, :3]) - height(p1[m, :3])
                if mode == "st":
                    return (None, None, swing_twist_full(DR[m], Dx[m], n, z_img, h_d))
                return (z_img, h_d, None)
            return target

        # R_real 在本框架下重写：只有 ref 一个假设、v5 的原门限、无歧义规则（ref 无对手）
        res["R_real"].append(auc(apply(p2, *correct(DR, Dx, segs,
                                   make_target([ref], "R_real", np.radians(35), 0.05), n, True))))
        res["R_multi"].append(auc(apply(p2, *correct(DR, Dx, segs, make_target([ref] + poses, "R_multi"), n, True))))
        res["R_multi_noref"].append(auc(apply(p2, *correct(DR, Dx, segs, make_target(poses, "R_multi_noref"), n, True))))
        res["R_multi_st"].append(auc(apply(p2, *correct(DR, Dx, segs, make_target([ref] + poses, "R_multi_st", mode="st"), n, True))))
        res["R_tier"].append(auc(apply(p2, *correct(DR, Dx, segs, make_target(poses, "R_tier", tiered=True), n, True))))
        res["O_3dof_min"].append(auc(apply(p2, *correct(DR, Dx, segs, oracle_target("min"), n, True))))
        res["O_3dof_st"].append(auc(apply(p2, *correct(DR, Dx, segs, oracle_target("st"), n, True))))
        per_ep.append((ei, meta[ei]["object"][:18], len(ups), len(tops) - 1, res["R_real"][-1], res["R_multi"][-1],
                       res["R_multi_st"][-1], res["O_3dof_min"][-1], res["O_3dof_st"][-1], res["R_tier"][-1]))

    r = np.array(res["raw"])
    print(f"{'方案':<16}{'AUC mean':>9}{'中位':>8}{'改善':>8}")
    for k in res:
        a = np.array(res[k]); print(f"{k:<16}{a.mean():>9.3f}{np.median(a):>8.3f}{int((a > r).sum()) if k != 'raw' else 0:>6}/30")
    print(f"\n静止段假设选择诊断（用真值判对错：面 5° 内且高度 1 cm 内）")
    print(f"{'':<16}{'静止段':>6}{'修正':>6}{'选对':>6}{'选错':>6}{'歧义跳过':>9}{'残余倾斜°中位':>13}{'残余高度mm中位':>14}")
    for k, D in diag.items():
        print(f"{k:<16}{D['seg']:>6}{D['fix']:>6}{D['ok']:>6}{D['bad']:>6}{D['skip_amb']:>9}"
              f"{np.median(D['tilt_err']) if D['tilt_err'] else float('nan'):>13.2f}{np.median(D['h_err']) if D['h_err'] else float('nan'):>14.1f}")
    print(f"\n逐序列（按 R_multi − R_real 排序）")
    print(f"{'ep':>3} {'物体':<18}{'面':>4}{'撑':>3}{'R_real':>8}{'multi':>8}{'multi_st':>9}{'O_min':>7}{'O_st':>7}{'tier':>7}")
    for e in sorted(per_ep, key=lambda e: e[5] - e[4]):
        print(f"{e[0]:>3} {e[1]:<18}{e[2]:>4}{e[3]:>3}{e[4]:>8.3f}{e[5]:>8.3f}{e[6]:>9.3f}{e[7]:>7.3f}{e[8]:>7.3f}{e[9]:>7.3f}")
    rm, rr = np.mean(res["R_multi"]), np.mean(res["R_real"])
    D = diag["R_multi"]; bad_rate = D["bad"] / max(D["fix"], 1)
    print(f"\n判据：R_multi {rm:.3f}（需 ≥ 0.678），选错率 {100*bad_rate:.1f}%（需 < 5%）→ "
          + ("采纳为 v5.1" if rm >= 0.678 and bad_rate < 0.05 else "保持 v5（R_real）"))
    rt = np.mean(res["R_tier"]); Dt = diag["R_tier"]
    ep = {e[0]: e for e in per_ep}
    keep = all(ep[i][9] >= ep[i][4] - 1e-3 for i in (1, 6))
    print(f"第二轮判据：R_tier {rt:.3f}（需 ≥ R_multi {rm:.3f}），选错 {Dt['bad']}（需 0），"
          f"vacuum {ep[1][9]:.3f}/v5 {ep[1][4]:.3f}，monitor {ep[6][9]:.3f}/v5 {ep[6][4]:.3f} → "
          + ("采纳 R_tier 为 v5.1" if rt >= rm and Dt["bad"] == 0 and keep else "不采纳 R_tier"))
    print(f"自检：v1 {np.mean(res['v1']):.3f}（应为 0.593）；R_real {rr:.3f}（应为 0.658）")
