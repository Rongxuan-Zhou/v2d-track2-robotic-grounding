"""Tier 3 exo 相机的曼哈顿世界标定，同时检验“世界坐标系 = 主机位相机的坐标系”（Mac 上跑，要 numpy、scipy、opencv-python-headless；
只读 exo 视频与发布的地面平面，不读任何轨迹，不读 Tier 1）。计划见 agent-doc/experiments/2026-10-16-t3-world-frame.md“下一步”。

模型：针孔，正方形像素、无斜切，焦距 f、主点 (cx, cy)，两阶径向畸变 x_d = x_u (1 + k1 r² + k2 r⁴)（归一化坐标，畸变中心取主点）。
房间里的直线大多沿三个正交方向（曼哈顿假设：墙、地、天花板、门框、灯格、地毯条纹）。线段两端点去畸变、反投影成视线 r1、r2，
解释平面法向 m = r1 × r2（单位化）；线段若平行于方向 d，则 m·d = 0。残差取 m·d_k（k 是指派给它的方向），按线段长度加权，Cauchy 损失。
两种拟合，内参与畸变各视频共用（同一台相机）：
  A：每段视频的相机姿态 R（相机 ← 曼哈顿）自由，3 个参数；检验量是估出的竖直方向（与地面法向最接近的那一列）与发布地面平面法向 n 的夹角。
  B：竖直方向固定为 n（假设“世界 = 这台相机的坐标系”，地面法向在相机坐标系里就是 n），每段视频只剩绕 n 的偏航 1 个参数。
假设成立时：A 的竖直方向与 n 的夹角落在自助法区间的噪声内，B 的代价只比 A 略高；假设不成立（例如差 3° 俯仰）时，A 的夹角明显大于噪声，
B 的代价明显变高。判读按 bootstrap 区间（看数据之前登记）：A 的夹角 95% 区间上界 < 1.0° 记“支持”，下界 > 1.5° 记“否定”，其余“说不清”。
线段来自每段视频 31 帧的逐像素中位数背景上的 LSD（长度 ≥ 40 px）；初始指派用逐个 RANSAC 消失点（同 t3_horizon_check.py），
之后交替：固定指派做最小二乘、按新参数重新指派（夹角 < 2° 才指派，其余算离群），共 5 轮。
自检：`--selftest` 在已知参数的合成房间上跑两种拟合，参数要恢复到登记的容差内（见 selftest）。
错切检验：`--skew-cx 702.8` 把 cx 固定为吊顶方格给的值，比较三种拟合 B 在同一套指派下的代价：cx 自由、cx 固定不加错切、cx 固定加错切
（K 的 (0, 1) 元，u = f x + skew y + cx）。错切只让竖直线倾斜、不影响水平线；若加错切能把代价降回 cx 自由时的水平，墙上竖直线与方格的冲突就能由
约 0.3° 的错切解释（看结果之前写下）。
灵敏度：`--n-rot 横滚°,俯仰°` 把 n 绕相机 z 轴（横滚）、x 轴（俯仰）转一个小角再拟合，看 B 的 cx、cy、f 跟着怎么动。理论（看结果之前写下）：
俯仰约 3.2° 时竖直消失点在主点下方约 f/|n_z| ≈ 1.55 万像素，竖直线段的倾斜对“横滚”与“竖直消失点的横坐标”几乎等价，
所以 B 的 cx ≈ 真 cx − (f/|n_z|)·横滚误差（0.1° 约 27 px）；俯仰误差主要进 f 与 cy（地平线行号 ≈ cy − f·n_z/n_y 是测出来的）。
用法：UV_NO_CONFIG=1 UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple uv run --no-project --python 3.12 --with numpy --with scipy \
        --with opencv-python-headless python stages/recovery/t3_manhattan_calib.py [--selftest] [--boot 200] <episode> ...
最后一行 MANHATTAN_CALIB_DONE（自检是 MANHATTAN_SELFTEST_OK）。
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "data/challenge/track_2/tier_2_synthetic_noise"
VID = DS / "videos/chunk-000/observation.images.exo_camera"
SECOND, SIDE = {12, 14, 16, 25, 28}, {21}
W, H = 1536, 1152
ASSIGN_DEG, ROUNDS, SCALE = 2.0, 5, math.sin(math.radians(0.5))


def undistort(p: np.ndarray, f, cx, cy, k1, k2, skew: float = 0.0) -> np.ndarray:
    """像素 → 去畸变的归一化坐标（迭代求逆）。skew 是 K 的 (0, 1) 元（像素）：u = f x + skew y + cx。"""
    yd = (p[..., 1] - cy) / f
    xd = (p[..., 0] - cx - skew * yd) / f
    xu, yu = xd.copy(), yd.copy()
    for _ in range(12):
        r2 = xu * xu + yu * yu
        s = 1.0 + k1 * r2 + k2 * r2 * r2
        xu, yu = xd / s, yd / s
    return np.stack([xu, yu], axis=-1)


def distort(xu: np.ndarray, f, cx, cy, k1, k2) -> np.ndarray:
    r2 = (xu ** 2).sum(-1, keepdims=True)
    return xu * (1.0 + k1 * r2 + k2 * r2 * r2) * f + np.array([cx, cy])


def plane_normals(seg: np.ndarray, f, cx, cy, k1, k2, skew: float = 0.0) -> np.ndarray:
    a = undistort(seg[:, 0:2], f, cx, cy, k1, k2, skew)
    b = undistort(seg[:, 2:4], f, cx, cy, k1, k2, skew)
    r1 = np.c_[a, np.ones(len(a))]
    r2 = np.c_[b, np.ones(len(b))]
    m = np.cross(r1, r2)
    return m / np.linalg.norm(m, axis=1, keepdims=True)


def rot_b(n: np.ndarray, yaw: float) -> np.ndarray:
    """拟合 B 的姿态：第 2 列（竖直）固定为 n，第 0、1 列是 n 的正交补里转 yaw 的两根轴。"""
    n = n / np.linalg.norm(n)
    t = np.cross(n, [0.0, 0.0, 1.0])
    t /= np.linalg.norm(t)
    u = np.cross(n, t)
    a = math.cos(yaw) * t + math.sin(yaw) * u
    b = np.cross(n, a)
    return np.c_[a, b, n]


class Fit:
    """内参 (f, cx, cy, k1, k2) 共用，每段视频一个姿态。mode A：旋转向量 3 个参数；mode B：yaw 1 个参数。"""

    def __init__(self, segs: list, mode: str, n: np.ndarray = None, skew_cx: float = None, skew_free: bool = True):
        """skew_cx 给了数时：cx 固定为它；skew_free 为真时原来 cx 的那个参数改作错切 skew（像素），为假时那个参数不起作用、错切为 0。"""
        self.segs, self.mode, self.n, self.skew_cx, self.skew_free = segs, mode, n, skew_cx, skew_free
        self.lens = [np.hypot(s[:, 2] - s[:, 0], s[:, 3] - s[:, 1]) for s in segs]

    def unpack(self, x: np.ndarray):
        k = x[:5] if self.skew_cx is None else np.array([x[0], self.skew_cx, x[2], x[3], x[4]])
        poses = []
        for i in range(len(self.segs)):
            if self.mode == "A":
                poses.append(Rotation.from_rotvec(x[5 + 3 * i: 8 + 3 * i]).as_matrix())
            else:
                poses.append(rot_b(self.n, x[5 + i]))
        return k, poses

    def skew(self) -> float:
        return float(self._x[1]) if self.skew_cx is not None and self.skew_free else 0.0

    def residuals(self, x: np.ndarray, assign: list) -> np.ndarray:
        self._x = x
        (f, cx, cy, k1, k2), poses = self.unpack(x)
        out = []
        for seg, ln, R, asg in zip(self.segs, self.lens, poses, assign):
            m = plane_normals(seg, f, cx, cy, k1, k2, self.skew())
            ok = asg >= 0
            d = R[:, asg[ok]].T
            out.append(np.sqrt(ln[ok] / 100.0) * (m[ok] * d).sum(1))
        return np.concatenate(out)

    def assign(self, x: np.ndarray) -> list:
        self._x = x
        (f, cx, cy, k1, k2), poses = self.unpack(x)
        res = []
        for seg, R in zip(self.segs, poses):
            m = plane_normals(seg, f, cx, cy, k1, k2, self.skew())
            s = np.abs(m @ R)
            k = s.argmin(1)
            k[s.min(1) > math.sin(math.radians(ASSIGN_DEG))] = -1
            res.append(k)
        return res

    def solve(self, x0: np.ndarray, assign: list = None):
        x = x0.copy()
        asg = assign if assign is not None else self.assign(x)
        for _ in range(ROUNDS):
            r = least_squares(lambda z: self.residuals(z, asg), x, loss="cauchy", f_scale=SCALE, x_scale="jac", max_nfev=200)
            x = r.x
            asg = self.assign(x)
        r = least_squares(lambda z: self.residuals(z, asg), x, loss="cauchy", f_scale=SCALE, x_scale="jac", max_nfev=200)
        return r.x, asg, float(r.cost)


def vertical_column(R: np.ndarray, n: np.ndarray) -> np.ndarray:
    c = R[:, int(np.argmax(np.abs(R.T @ n)))]
    return c if c @ n > 0 else -c


def angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))))))


def init_rotation_from_vps(vps: list, f, cx, cy) -> np.ndarray:
    cols = []
    for v in vps[:3]:
        d = np.array([(v[0] / v[2] - cx) / f, (v[1] / v[2] - cy) / f, 1.0]) if abs(v[2]) > 1e-9 else np.array([v[0], v[1], 0.0])
        cols.append(d / np.linalg.norm(d))
    M = np.c_[cols[0], cols[1], cols[2]]
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        R[:, 2] *= -1
    return R


# ---------- 真数据 ----------

def load_segments(ep: int):
    import t3_horizon_check as hc   # 同目录：背景、LSD、RANSAC 消失点沿用前一步的实现
    bg = hc.background(VID / f"episode_{ep:06d}.mp4")
    seg = hc.segments(bg).astype(float)
    vps = [v["vp"] for v in hc.find_vps(seg, np.random.default_rng(0))]
    return seg, vps


def floor_normal() -> np.ndarray:
    ns = []
    for line in (DS / "meta/episodes_metadata.jsonl").read_text().splitlines():
        m = json.loads(line)
        if int(m["episode_index"]) in SECOND | SIDE:
            continue
        p = np.array(json.loads((DS / m["ground_plane"]).read_text())["plane"], float)
        p /= np.linalg.norm(p[:3])
        ns.append(-p[:3] if p[3] < 0 else p[:3])
    n = np.mean(ns, axis=0)
    return n / np.linalg.norm(n)


def x0_for(mode: str, segs: list, vps_all: list, n: np.ndarray, f0=860.0) -> np.ndarray:
    cx0, cy0 = (W - 1) / 2, (H - 1) / 2
    x = [f0, cx0, cy0, 0.0, 0.0]
    for vps in vps_all:
        R = init_rotation_from_vps(vps, f0, cx0, cy0)
        if mode == "A":
            x += list(Rotation.from_matrix(R).as_rotvec())
        else:
            v = vertical_column(R, n)
            others = [R[:, j] for j in range(3) if abs(R[:, j] @ v) < 0.9]
            t = np.cross(n, [0.0, 0.0, 1.0])
            t /= np.linalg.norm(t)
            u = np.cross(n, t)
            a = others[0] - (others[0] @ n) * n
            x.append(math.atan2(a @ u, a @ t))
    return np.array(x)


def report(tag: str, x: np.ndarray, fit: Fit, n: np.ndarray, cost: float, asg: list) -> dict:
    (f, cx, cy, k1, k2), poses = fit.unpack(x)
    used = sum(int((a >= 0).sum()) for a in asg)
    out = {"f": f, "cx": cx, "cy": cy, "k1": k1, "k2": k2, "cost": cost, "used": used}
    if fit.mode == "A":
        out["angles"] = [angle_deg(vertical_column(R, n), n) for R in poses]
        # 各视频的相机姿态两两夹角：同一台相机没动过时应在噪声内（曼哈顿三轴的符号与次序可能互换，取使夹角最小的那种对应）
        def rel(Ra, Rb):
            best = 180.0
            for perm in ((0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)):
                for sx in (1, -1):
                    for sy in (1, -1):
                        Pb = Rb[:, perm] * np.array([sx, sy, sx * sy * np.linalg.det(Rb[:, perm])])
                        best = min(best, math.degrees(Rotation.from_matrix(Ra @ Pb.T).magnitude()))
            return best
        out["pose_spread"] = max((rel(a, b) for i, a in enumerate(poses) for b in poses[i + 1:]), default=0.0)
    print(f"  {tag}：f {f:.1f}、cx {cx:.1f}、cy {cy:.1f}、k1 {k1:+.4f}、k2 {k2:+.4f}；代价 {cost:.2f}，用到线段 {used} 条"
          + (f"；竖直方向与 n 的夹角 {', '.join(f'{a:.2f}°' for a in out['angles'])}；各视频姿态两两最大夹角 {out['pose_spread']:.2f}°"
             if fit.mode == "A" else ""))
    return out


def run(eps: list, boot: int, n_rot: tuple = (0.0, 0.0), skew_cx: float = None) -> int:
    n = floor_normal()
    if any(n_rot):   # 灵敏度：n 绕相机 z 轴转 n_rot[0]°（横滚）、绕 x 轴转 n_rot[1]°（俯仰）
        n = Rotation.from_rotvec([math.radians(n_rot[1]), 0.0, 0.0]).apply(Rotation.from_rotvec([0.0, 0.0, math.radians(n_rot[0])]).apply(n))
        print(f"== 灵敏度：n 横滚 {n_rot[0]:+.3f}°、俯仰 {n_rot[1]:+.3f}°")
    data = [load_segments(e) for e in eps]
    segs, vps_all = [d[0] for d in data], [d[1] for d in data]
    print(f"== 视频 {eps}；线段 {[len(s) for s in segs]} 条；发布地面法向 n = ({n[0]:+.5f}, {n[1]:+.5f}, {n[2]:+.5f})")
    fa, fb = Fit(segs, "A", n), Fit(segs, "B", n)
    xa, asg_a, ca = fa.solve(x0_for("A", segs, vps_all, n))
    xb, asg_b, cb = fb.solve(x0_for("B", segs, vps_all, n))
    ra = report("拟合 A（姿态自由）", xa, fa, n, ca, asg_a)
    rb = report("拟合 B（竖直 = n）", xb, fb, n, cb, asg_b)
    print(f"  同一套指派（A 的）下的代价：A {fb_cost_with(fa, xa, asg_a):.3f}，B {fb_cost_with(fb, xb, asg_a):.3f}")
    if boot:
        rng = np.random.default_rng(1)
        stats = {"angle": [], "f": [], "cx": [], "cy": [], "k1": [], "cxB": [], "cyB": [], "fB": [], "k1B": []}
        for b in range(boot):
            idx = [rng.integers(0, len(s), len(s)) for s in segs]
            bs = [s[i] for s, i in zip(segs, idx)]
            fa_b = Fit(bs, "A", n)
            xab, _, _ = fa_b.solve(xa, [a[i] for a, i in zip(asg_a, idx)])
            (f, cx, cy, k1, _), poses = fa_b.unpack(xab)
            stats["angle"].append(np.mean([angle_deg(vertical_column(R, n), n) for R in poses]))
            stats["f"].append(f); stats["cx"].append(cx); stats["cy"].append(cy); stats["k1"].append(k1)
            fb_b = Fit(bs, "B", n)
            xbb, _, _ = fb_b.solve(xb, [a[i] for a, i in zip(asg_b, idx)])
            stats["fB"].append(xbb[0]); stats["cxB"].append(xbb[1]); stats["cyB"].append(xbb[2]); stats["k1B"].append(xbb[3])
        q = {k: np.percentile(v, [2.5, 50, 97.5]) for k, v in stats.items()}
        for k, (lo, mid, hi) in q.items():
            print(f"  自助法 {boot} 次 {k}：中位 {mid:.3f}，95% 区间 [{lo:.3f}, {hi:.3f}]")
        lo, hi = q["angle"][0], q["angle"][2]
        verdict = "支持：世界 = 这台相机的坐标系（转动）" if hi < 1.0 else "否定" if lo > 1.5 else "说不清"
        print(f"  判读（A 的竖直方向与 n 的夹角，按视频平均）：{verdict}")
    if skew_cx is not None:   # 错切检验：cx 固定为方格的值，看加一个错切能不能把代价降回 cx 自由时的水平
        x0 = xb.copy()
        x0[1] = 0.0
        f0s = Fit(segs, "B", n, skew_cx=skew_cx, skew_free=False)
        x_ns, _, _ = f0s.solve(x0, asg_b)
        fss = Fit(segs, "B", n, skew_cx=skew_cx, skew_free=True)
        x_s, _, _ = fss.solve(x0, asg_b)
        c_free, c_ns, c_s = fb_cost_with(fb, xb, asg_b), fb_cost_with(f0s, x_ns, asg_b), fb_cost_with(fss, x_s, asg_b)
        print(f"  错切检验（拟合 B，同一套指派）：cx 自由（{xb[1]:.1f}）代价 {c_free:.4f}；cx 固定 {skew_cx} 不加错切 代价 {c_ns:.4f}"
              f"（f {x_ns[0]:.1f}、cy {x_ns[2]:.1f}）；cx 固定 {skew_cx} 加错切 代价 {c_s:.4f}，错切 {x_s[1]:+.2f} px"
              f"（{math.degrees(math.atan2(x_s[1], x_s[0])):+.3f}°，f {x_s[0]:.1f}、cy {x_s[2]:.1f}）")
    print("MANHATTAN_CALIB_DONE")
    return 0


def fb_cost_with(fit: Fit, x: np.ndarray, asg: list) -> float:
    r = fit.residuals(x, asg)
    return float(np.sum(SCALE ** 2 / 2 * np.log1p((r / SCALE) ** 2)))


# ---------- 自检：合成房间 ----------

def synth_segments(K, R, rng, n_seg=600, noise_px=0.5) -> np.ndarray:
    f, cx, cy, k1, k2 = K
    segs = []
    while len(segs) < n_seg:
        axis = rng.integers(0, 3)
        p = rng.uniform([-6, -2.5, -6], [6, 1.4, 6])   # 曼哈顿坐标：第 2 轴竖直
        p[1] = rng.choice([-1.4, 1.6]) if axis != 1 and rng.random() < 0.6 else p[1]   # 多在地面、天花板上
        d = np.zeros(3); d[axis] = rng.uniform(0.3, 1.5)
        a, b = p, p + d
        ca, cb = R @ a + [0, 0, 0], R @ b
        ca[2] += 5.0; cb[2] += 5.0   # 相机在房间里，物体在前方
        if ca[2] < 0.5 or cb[2] < 0.5:
            continue
        ua = distort(np.array(ca[:2] / ca[2]), f, cx, cy, k1, k2) + rng.normal(0, noise_px, 2)
        ub = distort(np.array(cb[:2] / cb[2]), f, cx, cy, k1, k2) + rng.normal(0, noise_px, 2)
        if not (0 <= ua[0] < W and 0 <= ua[1] < H and 0 <= ub[0] < W and 0 <= ub[1] < H):
            continue
        if np.hypot(*(ua - ub)) < 40:
            continue
        segs.append([*ua, *ub])
    return np.array(segs)


def selftest() -> int:
    rng = np.random.default_rng(7)
    n_true = np.array([-0.0189, -0.9983, -0.0555]); n_true /= np.linalg.norm(n_true)
    K_true = (860.0, 770.0, 535.0, -0.08, 0.01)
    yaw = 0.6
    R_true = rot_b(n_true, yaw)
    # 曼哈顿第 2 轴是竖直：相机坐标里竖直 = R[:, 2] = n
    segs = [synth_segments(K_true, R_true, rng) for _ in range(2)]
    vps_all = []
    for _ in segs:
        vps = []
        for j in range(3):
            d = R_true[:, j]
            vps.append(np.array([K_true[0] * d[0] + K_true[1] * d[2], K_true[0] * d[1] + K_true[2] * d[2], d[2]]))
        vps_all.append(vps)
    fa, fb = Fit(segs, "A", n_true), Fit(segs, "B", n_true)
    xa, asg_a, ca = fa.solve(x0_for("A", segs, vps_all, n_true))
    xb, asg_b, cb = fb.solve(x0_for("B", segs, vps_all, n_true))
    ra = report("自检 A", xa, fa, n_true, ca, asg_a)
    rb = report("自检 B", xb, fb, n_true, cb, asg_b)
    ok = (abs(ra["f"] - 860) < 15 and abs(rb["f"] - 860) < 10 and abs(rb["cy"] - 535) < 6 and abs(rb["cx"] - 770) < 10
          and max(ra["angles"]) < 0.5 and abs(rb["k1"] + 0.08) < 0.02)
    # 反向对照：竖直方向故意错 3°，B 的代价应明显变高、A 的夹角应约 3°
    n_wrong = Rotation.from_rotvec([math.radians(3.0), 0, 0]).apply(n_true)
    fbw = Fit(segs, "B", n_wrong)
    xbw, asg_bw, cbw = fbw.solve(x0_for("B", segs, vps_all, n_wrong))
    faw = Fit(segs, "A", n_wrong)
    xaw, asg_aw, caw = faw.solve(xa)
    angw = [angle_deg(vertical_column(R, n_wrong), n_wrong) for R in faw.unpack(xaw)[1]]
    print(f"  反向对照（n 错 3°）：B 代价 {cbw:.2f}（对的 n 是 {cb:.2f}），A 的竖直方向与错 n 的夹角 {', '.join(f'{a:.2f}°' for a in angw)}，"
          f"B 估的 cy {xbw[2]:.1f}")
    ok = ok and min(angw) > 2.0
    print("MANHATTAN_SELFTEST_OK" if ok else "MANHATTAN_SELFTEST_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("eps", nargs="*", type=int)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--boot", type=int, default=0)
    ap.add_argument("--n-rot", default="0,0", help="灵敏度：横滚°,俯仰°")
    ap.add_argument("--skew-cx", type=float, default=None, help="错切检验：cx 固定为这个值，比较加不加错切的拟合 B 代价")
    a = ap.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.exit(selftest() if a.selftest else run(a.eps or [5, 11, 20], a.boot, tuple(float(v) for v in a.n_rot.split(",")), a.skew_cx))
