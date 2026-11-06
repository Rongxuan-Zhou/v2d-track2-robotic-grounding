"""Tier 3 误差预算之一：内参这一项让现有交付版重建在世界坐标系里偏了多少（Mac 上跑，要 numpy、pyarrow；只读交付版的物体位姿与
世界步参数，不读 Tier 1，不读 Tier 2，不与任何真值比）。

交付版 tier3_cari4d_f830_best_dr2s_dyn7 的主机位链路（t3_convert_ep.py，另一位用户的分支 track2/tier3-recon）：CARI4D 用
K_c = (fx 835 上下, 主点 (768, 576) 居中) 重建；焦距步骤把 fx 换成 830、深度乘 830/fx_c、主点不动；世界步把相机系绕原点转 R
（地面法向对到发布的 n）再乘 s（离地高度对到发布值），即 X_w = s·R·X_830。所以每帧物体中心可以倒推回像素与深度：
X_830 = Rᵀ X_w / s，p = K_830 X_830，Z = X_830 的 z。

世界坐标系就是主机位相机的坐标系（results/t3world/manhattan_calib_main9_boot.txt，判读“支持”）。真内参 K_t 下，同一个像素 p 的视线是
K_t⁻¹[p, 1]，深度按两种口径给上下界（看结果之前定）：
  口径一（视大小）：深度来自已知网格的视大小，随焦距成比例：X_t = (Z·f_t/830)·K_t⁻¹[p, 1]。
  口径二（距离不变）：沿视线的距离取流程最终的 ‖X_w‖（含世界步缩放 s），只改方向：X_t = ‖X_w‖·单位(K_t⁻¹[p, 1])。
差 ‖X_t − X_w‖ 是“只把内参改对、别的都不动”时这份重建要移动的量。K_t 列三套：吊顶方格拟合 G、曼哈顿拟合 B、曼哈顿拟合 A
（数值见下 CANDIDATES 与出处）；cx 还没定（agent-doc/experiments/2026-10-16-t3-world-frame.md“没解决”一节），所以三套都报。
另报只看方向的角度误差：流程的世界系视线 R·K_830⁻¹[p, 1] 与 K_t⁻¹[p, 1] 的夹角。

理论预期（看结果之前写下）：方格 K 下偏航约 (768 − 702)/866 ≈ 4.4°，即每米距离横向约 7.7 cm；口径一的深度多 866/830 − 1 ≈ 4.3%；
俯仰：流程转了约 3.7°，真主点比 576 高 (576 − 533)/866 ≈ 2.8°，剩约 0.9° 的俯仰差。
只绕 n 的偏航：R* 是 3 个自由度的转动，其中的俯仰、横滚部分会把流程按发布地面做的对齐（R_floor）拧回去，内参候选又不带横滚信息，那部分
是 Kabsch 吸收了重建的深度倾斜，不是内参的修正（Fable 复核 MAJOR 1）。cx 只决定偏航，所以 A/B 用只绕 n（发布的地面法向，朝上；
t3_manhattan_calib.floor_normal，属“场景信息”的解读）转一个角 ψ 的转动：ψ* = atan2(Σ (a_h × b_h)·n, Σ a_h·b_h)，a_h、b_h 是 X_w、X_t
去掉 n 方向分量后的水平部分（使 Σ‖R_n(ψ) X_w − X_t‖² 最小的闭式解）。报 ψ* 与只转偏航后剩下的位移，--write-rotations 另写 rot_<label>_yaw.json。
刚体部分：打包器用“评测器记录的参考 ↔ 我们交的 frame refs”之间的一个刚体变换把仿真里的物体位姿转到发布世界系（官方 kit
tracks/track_2/README.md“Input references for frame conversion”），所以 frame refs 整体绕相机中心转一个 R，打包出的轨迹也整体转 R，
不用重训。对每套 K_t 求所有可见帧上使 Σ‖R·X_w − X_t‖² 最小的转动 R*（Kabsch，绕原点，口径二），报 R* 的角度、其中绕 n 的偏航分量，
以及转完之后剩下的位移（非刚体部分，打包消不掉）。
只算主机位 24 集（第二机位 ep12、14、16、25、28 与侧机位 ep21 还叠着配准，另算）。
用法：uv run --no-project --python 3.12 --with numpy --with pyarrow python stages/recovery/t3_k_budget.py [--root runs/t3world/budget]
最后一行 K_BUDGET_DONE。
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
CANDIDATES = {   # (f, cx, cy)
    "方格 G（ceiling_grid_main3）": (866.2, 702.0, 533.3),
    "曼哈顿 B（manhattan_calib_main9）": (860.8, 790.0, 527.5),
    "曼哈顿 A（manhattan_calib_main9）": (862.2, 847.1, 530.5),
}


def K(f, cx, cy) -> np.ndarray:
    return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def episode(d: Path) -> dict:
    wc = json.loads((d / "world_calib.json").read_text())
    if wc.get("camera_register") or wc.get("camera_group") != 0:
        raise SystemExit(f"{d.name}：不是主机位（camera_group {wc.get('camera_group')}）或带了配准")
    oc = np.load(d / "object_cam.npz")
    kc = oc["intrinsics"]
    f830 = float(wc["focal"]["fx_new"])
    k830 = K(f830, float(kc[0, 2]), float(kc[1, 2]))
    R, s = np.array(wc["rotation"], float), float(wc["scale"])
    pq_file = next(d.glob("episode_*.parquet"))
    t = pq.read_table(pq_file, columns=["observation.object.pose", "observation.object.visible"])
    pose = np.stack(t.column("observation.object.pose").to_numpy(zero_copy_only=False)).astype(float)
    vis = np.asarray(t.column("observation.object.visible").to_numpy(zero_copy_only=False), bool)
    Xw = pose[vis, :3]
    X830 = Xw @ R / s          # Rᵀ X_w / s，按行
    p = X830 @ k830.T
    p = p[:, :2] / p[:, 2:3]
    return {"name": d.name, "Xw": Xw, "X830": X830, "p": p, "f830": f830, "k830": k830, "R": R, "s": s,
            "cxcy": (float(kc[0, 2]), float(kc[1, 2])), "frames": int(vis.sum())}


def budget(ep: dict, kt: tuple) -> dict:
    f, cx, cy = kt
    r = np.c_[ep["p"], np.ones(len(ep["p"]))] @ np.linalg.inv(K(f, cx, cy)).T
    Z = ep["X830"][:, 2]
    x1 = r * (Z * f / ep["f830"])[:, None]                                  # 口径一：视大小
    x2 = unit(r) * np.linalg.norm(ep["Xw"], axis=1)[:, None]                 # 口径二：距离取流程最终的 ‖X_w‖
    ray_pipe = unit(np.c_[ep["p"], np.ones(len(ep["p"]))] @ np.linalg.inv(ep["k830"]).T @ ep["R"].T)
    ang = np.degrees(np.arccos(np.clip((ray_pipe * unit(r)).sum(1), -1, 1)))
    d1, d2 = np.linalg.norm(x1 - ep["Xw"], axis=1), np.linalg.norm(x2 - ep["Xw"], axis=1)
    rng = np.linalg.norm(ep["Xw"], axis=1)
    return {"ang": float(np.median(ang)), "d1": float(np.median(d1)), "d2": float(np.median(d2)), "range": float(np.median(rng)),
            "d1_all": d1, "d2_all": d2}


def budget_x1(ep: dict, kt: tuple) -> np.ndarray:
    """口径一：深度随焦距成比例（z_t = z_830 · f_t / 830），像素不变。"""
    f, cx, cy = kt
    r = np.c_[ep["p"], np.ones(len(ep["p"]))] @ np.linalg.inv(K(f, cx, cy)).T
    return r * (ep["X830"][:, 2] * f / ep["f830"])[:, None]


def budget_x2(ep: dict, kt: tuple) -> np.ndarray:
    f, cx, cy = kt
    r = np.c_[ep["p"], np.ones(len(ep["p"]))] @ np.linalg.inv(K(f, cx, cy)).T
    return unit(r) * np.linalg.norm(ep["Xw"], axis=1)[:, None]


def rot_about(n: np.ndarray, psi: float) -> np.ndarray:
    """绕单位轴 n 转 psi（弧度），Rodrigues。"""
    K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
    return np.eye(3) + math.sin(psi) * K + (1 - math.cos(psi)) * K @ K


def yaw_fit(A: np.ndarray, B: np.ndarray, n: np.ndarray) -> float:
    """使 Σ‖R_n(ψ) a − b‖² 最小的 ψ（闭式解）：只看去掉 n 分量后的水平部分。"""
    ah, bh = A - np.outer(A @ n, n), B - np.outer(B @ n, n)
    return math.atan2(float((np.cross(ah, bh) @ n).sum()), float((ah * bh).sum()))


def kabsch(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """绕原点的最佳转动 R（使 Σ‖R a − b‖² 最小）。"""
    U, _, Vt = np.linalg.svd(A.T @ B)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def floor_up() -> np.ndarray:
    """主机位 24 集发布地面法向的平均（朝上），与 t3_manhattan_calib.floor_normal 相同。"""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import t3_manhattan_calib as mc
    return mc.floor_normal()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT / "runs/t3world/budget")
    ap.add_argument("--write-rotations", type=Path, help="把每套内参的 R* 按全精度写成 rotate_frefs_t3.py 用的 JSON（rot_<label>.json）")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "libs/v2d_infra"))
    from tier_guard import forbid_tier1
    forbid_tier1()   # 档位隔离：此后读 Tier 1 的任何文件都会抛错
    eps = [episode(d) for d in sorted(a.root.iterdir()) if d.is_dir()]
    print(f"== 主机位 {len(eps)} 集（{a.root}）；流程内参：fx {eps[0]['f830']:.0f}、主点 {eps[0]['cxcy']}；世界步缩放 "
          f"{min(e['s'] for e in eps):.4f}–{max(e['s'] for e in eps):.4f}")
    for tag, kt in CANDIDATES.items():
        print(f"== 真内参取 {tag}：f {kt[0]}、cx {kt[1]}、cy {kt[2]}")
        print("集\t可见帧\t距离中位(m)\t视线夹角中位(°)\t位移中位 口径一(cm)\t位移中位 口径二(cm)")
        all1, all2 = [], []
        for ep in eps:
            b = budget(ep, kt)
            all1.append(b["d1_all"])
            all2.append(b["d2_all"])
            print(f"{ep['name']}\t{ep['frames']}\t{b['range']:.2f}\t{b['ang']:.2f}\t{100 * b['d1']:.1f}\t{100 * b['d2']:.1f}")
        a1, a2 = np.concatenate(all1), np.concatenate(all2)
        Xw = np.concatenate([ep["Xw"] for ep in eps])
        Xt = np.concatenate([budget_x2(ep, kt) for ep in eps])
        Rs = kabsch(Xw, Xt)
        resid = np.linalg.norm(Xw @ Rs.T - Xt, axis=1)
        n_up = floor_up()
        yaw = math.degrees(math.atan2((np.cross([0, 0, 1.0], Rs @ [0, 0, 1.0])) @ n_up, (Rs @ [0, 0, 1.0]) @ [0, 0, 1.0]))
        print(f"  最佳刚体转动 R*（绕相机中心，24 集合拟）：转角 {math.degrees(np.arccos(np.clip((np.trace(Rs) - 1) / 2, -1, 1))):.2f}°，"
              f"光轴绕 n 的偏航 {yaw:+.2f}°；转完后剩下的位移中位 {100 * np.median(resid):.1f} cm、90% 分位 {100 * np.percentile(resid, 90):.1f} cm"
              f"（口径二；打包消不掉的非刚体部分）；R* 按行 {np.array2string(Rs, precision=5, separator=',').replace(chr(10), '')}")
        psi = yaw_fit(Xw, Xt, n_up)
        Ry = rot_about(n_up, psi)
        resid_y = np.linalg.norm(Xw @ Ry.T - Xt, axis=1)
        print(f"  只绕 n 的偏航 ψ* {math.degrees(psi):+.2f}°；只转偏航后剩下的位移中位 {100 * np.median(resid_y):.1f} cm、90% 分位 "
              f"{100 * np.percentile(resid_y, 90):.1f} cm（口径二）")
        Xt1 = np.concatenate([budget_x1(ep, kt) for ep in eps])   # 口径一：同一个偏航（A/B 交的那个）转完后还剩多少
        resid_y1 = np.linalg.norm(Xw @ Ry.T - Xt1, axis=1)
        dz1 = (np.linalg.norm(Xt1, axis=1) - np.linalg.norm(Xw, axis=1))
        print(f"  口径一下：用同一个偏航转完后剩下的位移中位 {100 * np.median(resid_y1):.1f} cm、90% 分位 {100 * np.percentile(resid_y1, 90):.1f} cm；"
              f"其中沿视线的距离差（口径一 − 流程）中位 {100 * np.median(dz1):+.1f} cm（相对 {100 * np.median(dz1 / np.linalg.norm(Xw, axis=1)):+.1f}%）")
        if a.write_rotations:
            label = {"方格 G": "gridG", "曼哈顿 B": "manhattanB", "曼哈顿 A": "manhattanA"}[tag.split("（")[0]]
            a.write_rotations.mkdir(parents=True, exist_ok=True)
            f = a.write_rotations / f"rot_{label}.json"
            f.write_text(json.dumps({"rotation": Rs.tolist(), "label": label,
                                     "source": f"stages/recovery/t3_k_budget.py R*（{tag}，f {kt[0]}、cx {kt[1]}、cy {kt[2]}；主机位 24 集、口径二）"},
                                    ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            print(f"  写 {f}")
            fy = a.write_rotations / f"rot_{label}_yaw.json"
            fy.write_text(json.dumps({"rotation": Ry.tolist(), "label": f"{label}_yaw", "yaw_deg": math.degrees(psi), "axis_n": n_up.tolist(),
                                      "source": f"stages/recovery/t3_k_budget.py 只绕 n 的偏航 ψ*（{tag}，f {kt[0]}、cx {kt[1]}、cy {kt[2]}；"
                                                f"主机位 24 集、口径二）"}, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            print(f"  写 {fy}")
        print(f"  全部帧：口径一位移中位 {100 * np.median(a1):.1f} cm、90% 分位 {100 * np.percentile(a1, 90):.1f} cm；"
              f"口径二位移中位 {100 * np.median(a2):.1f} cm、90% 分位 {100 * np.percentile(a2, 90):.1f} cm；"
              f"位移 < 9 cm（AUC 最宽的阈值）的帧占 口径一 {100 * np.mean(a1 < 0.09):.0f}%、口径二 {100 * np.mean(a2 < 0.09):.0f}%")
    print("K_BUDGET_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
