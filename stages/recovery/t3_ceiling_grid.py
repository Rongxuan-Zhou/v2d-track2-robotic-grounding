"""Tier 3 exo 相机的吊顶方格标定：用方格的正方形结构定主点横坐标 cx（Mac 上跑，要 numpy、scipy、opencv-python-headless；
只读 exo 视频与发布的地面平面，不读任何轨迹，不读 Tier 1）。

为什么做：曼哈顿标定（t3_manhattan_calib.py）定得住 f 与 cy，定不住 cx。主机位俯仰约 3.2°，竖直消失点在主点下方约 1.55 万像素，
竖直线段的倾斜对“横滚”与“竖直消失点的横坐标”几乎等价，cx 跟着横滚的误差走，放大约 1/sin(俯仰) ≈ 18 倍；合成房间自检里理想数据也偏 8 px，
真数据拟合 A 的 95% 区间 [782, 964]。cx 错 Δ 像素，等于整个场景在世界坐标系里绕相机竖轴转 Δ/f（30 px 约 2°，4 m 外约 14 cm）。

原理（看真数据之前写下）：吊顶是 2 英尺见方的 T 型龙骨方格（灯板占 2×4 英尺），所在平面与地面平行。平面上的度量结构决定它的两个
圆点在图像里的位置：两个圆点在地平线上，实部是主点到地平线的垂足 F，虚部是 ρ = √(D² + f²)（D 是主点到地平线的距离）。所以方格单独就能
钉住 F（主点沿地平线方向的位置，地平线近乎水平，即 cx），而 D 与 f 只受 D² + f² = ρ² 一条约束，还差一个量，要靠墙上的竖直线（拟合 G）
或发布的地面法向 n（拟合 GB）补上。换个说法：K 错时，只要两个水平消失点对得上，模型把吊顶反投回去得到的是真方格的轴对齐仿射
（两族线各自缩放、平移），所以方格在曼哈顿正交之外只多给一个方程：两族格距相等。这个方程定 cx 有多准，先在合成数据上量（自检）。

模型：针孔，主点 (cx, cy)，焦距 f（可选纵横比 asp，fy = asp·f），一阶径向畸变 k1（归一化坐标）。每段视频一个相机姿态 R（相机 ← 房间，
第 0、1 列是方格两个方向 e1、e2，第 2 列 e3 竖直向上）和一套方格：吊顶平面 e3·X = 1（以相机到吊顶的距离 h 为单位），格线 a = φa + iΔ
（沿 e2 的线）与 b = φb + jΔ（沿 e1 的线），Δ = 格距/h。线段两端点去畸变成视线，残差是端点到所指派格线的像距（像素）。
拟合 G：R 自由，加墙上竖直线段（解释平面要含 e3），不用 n；拟合 GB：e3 固定为 n，只剩偏航，不用竖直线。G 估出的 e3 与 n 的夹角是又一个
“世界 = 这台相机”的检验（用的是吊顶，不是墙）。指派与求解交替：先按矩形格（两族格距各自）定好每条线的格号，再加正方形约束放开内参。

判读（看真数据之前登记）：
  1. cx 定住了：拟合 G 的 cx 自助法 95% 区间宽 ≤ 20 px（偏航不确定度 ≤ ±0.7°）；否则记“cx 仍定不住”，报区间。
  2. G 与 GB 的 cx 之差落在 G 的区间内，记“两种补法一致”；否则两个都报，标出分歧。
  3. G 的 e3 与 n 的夹角：≤ 0.5° 记“吊顶与世界坐标系的竖直方向一致”。
  4. 只报不判：纵横比（asp 放开时）、k1、每段视频单独拟合的 cx 极差、h 按 0.6096 m 与 0.600 m 两种格距换成吊顶离相机的距离。
自检（`--selftest`）：合成吊顶方格 + 墙上竖直线，真值 K = (860, 805, 528)、k1 = 0；线段带 T 型龙骨两侧边缘（±12 mm）、灯板内缩边（±4 cm）、
0.4 px 端点噪声和 15% 离群；初值 cx 故意偏 +45 px、f 偏 −20 px（模仿曼哈顿的偏差）。10 个种子里 G 与 GB 的 |cx 误差| ≤ 5 px、|f 误差| ≤ 5 px
各至少 9 个，记 CEILING_SELFTEST_OK。另报两个反向对照：方格故意做成 1:1.03 的长方形（看 cx 偏多少），GB 的 n 故意错 0.1° 横滚（看 cx 偏多少）。
用法：UV_NO_CONFIG=1 UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple uv run --no-project --python 3.12 --with numpy --with scipy \
        --with opencv-python-headless python stages/recovery/t3_ceiling_grid.py [--selftest] [--boot 100] [--asp] [--overlay] <episode> ...
叠加图（--overlay）写到 runs/t3world/ceiling_grid_ep<N>.png（不进 git）。最后一行 CEILING_GRID_DONE。
"""
import argparse
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t3_manhattan_calib as mc  # noqa: E402  rot_b、floor_normal、曼哈顿拟合（初值）

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/t3world"
W, H = mc.W, mc.H
TILES_M = (0.6096, 0.600)
MIN_LEN = 25.0
FAM_DEG0, FAM_DEG = 3.0, 1.5
TOL_PX = (4.0, 3.0, 2.0, 1.5)
SCALE_PX = 1.0
UP_MIN = math.sin(math.radians(1.0))
D_RANGE = (0.25, 0.70)
D_FLOOR = 1.3877  # 主机位 24 集地面平面 d 的中位（results/t3world/plane_facts.txt 第 7 列），只用于地面俯视图


# ---------- 几何 ----------

def rays(p: np.ndarray, k: tuple) -> np.ndarray:
    """像素 → 去畸变的视线 (x, y, 1)。k = (f, cx, cy, k1, asp)。"""
    f, cx, cy, k1, asp = k
    xd, yd = (p[:, 0] - cx) / f, (p[:, 1] - cy) / (f * asp)
    xu, yu = xd.copy(), yd.copy()
    for _ in range(10):
        s = 1.0 + k1 * (xu * xu + yu * yu)
        xu, yu = xd / s, yd / s
    return np.c_[xu, yu, np.ones(len(xu))]


def project(X: np.ndarray, k: tuple) -> np.ndarray:
    f, cx, cy, k1, asp = k
    xu, yu = X[:, 0] / X[:, 2], X[:, 1] / X[:, 2]
    s = 1.0 + k1 * (xu * xu + yu * yu)
    return np.c_[f * xu * s + cx, f * asp * yu * s + cy]


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def line_dist(m: np.ndarray, r: np.ndarray, f: float) -> np.ndarray:
    """视线 r（z = 1）到解释平面法向为 m 的像线的距离，换成像素（乘 f）。"""
    return f * (m * r).sum(1) / np.linalg.norm(m[:, :2], axis=1)


# ---------- 拟合 ----------

class GridFit:
    """内参共用；每段视频一个姿态与一套方格。mode G：姿态为相对初值 R0 的旋转向量（3 个参数）并用竖直线；GB：e3 = n，只剩偏航。
    square=False 时两族格距各自独立（只用来定格号）；fix_k=True 时内参不动。"""

    def __init__(self, vids: list, mode: str, n: np.ndarray, R0s: list, asp: bool = False, fixed_delta: float = None):
        self.vids, self.mode, self.n, self.R0s, self.asp, self.fixed_delta = vids, mode, n, R0s, asp, fixed_delta

    def nk(self) -> int:
        return 5 if self.asp else 4

    def npv(self, square: bool) -> int:
        return (3 if self.mode == "G" else 1) + (3 if square else 4)

    def unpack(self, x: np.ndarray, square: bool):
        f, cx, cy, k1 = x[:4]
        k = (f, cx, cy, k1, x[4] if self.asp else 1.0)
        o, per = self.nk(), []
        for v in range(len(self.vids)):
            if self.mode == "G":
                R = Rotation.from_rotvec(x[o:o + 3]).as_matrix() @ self.R0s[v]
                o += 3
            else:
                R = mc.rot_b(self.n, x[o])
                o += 1
            if square:
                da = db = self.fixed_delta if self.fixed_delta else math.exp(x[o])   # fixed_delta：格距固定（参数留着、不起作用）
                pa, pb = x[o + 1], x[o + 2]
                o += 3
            else:
                da, db = math.exp(x[o]), math.exp(x[o + 1])
                pa, pb = x[o + 2], x[o + 3]
                o += 4
            per.append((R, da, db, pa, pb))
        return k, per

    def lattice_coords(self, v: int, k: tuple, R: np.ndarray, seg: np.ndarray):
        e1, e2, e3 = R[:, 0], R[:, 1], R[:, 2]
        ra, rb = rays(seg[:, 0:2], k), rays(seg[:, 2:4], k)
        m = unit(np.cross(ra, rb))
        upa, upb = (ra @ e3) / np.linalg.norm(ra, axis=1), (rb @ e3) / np.linalg.norm(rb, axis=1)
        rm = rays(np.c_[(seg[:, 0] + seg[:, 2]) / 2, (seg[:, 1] + seg[:, 3]) / 2], k)
        up_m = rm @ e3
        with np.errstate(divide="ignore", invalid="ignore"):
            a, b = (rm @ e1) / up_m, (rm @ e2) / up_m
        return m, (upa > UP_MIN) & (upb > UP_MIN), a, b, ra, rb

    def assign(self, x: np.ndarray, square: bool, tol: float, fam_deg: float = FAM_DEG) -> list:
        k, per = self.unpack(x, square)
        out = []
        for v, (R, da, db, pa, pb) in enumerate(per):
            e1, e2, e3 = R[:, 0], R[:, 1], R[:, 2]
            seg = self.vids[v]["ceil"]
            m, ceil, a, b, ra, rb = self.lattice_coords(v, k, R, seg)
            s = math.sin(math.radians(fam_deg))
            res = {}
            for fam, along, other, coord, dd, ph in (("A", e2, e1, a, da, pa), ("B", e1, e2, b, db, pb)):
                cand = ceil & (np.abs(m @ along) < s) & np.isfinite(coord)
                idx = np.nonzero(cand)[0]
                j = np.round((coord[idx] - ph) / dd)
                lv = ph + j * dd
                mm = np.cross(e3[None, :] + lv[:, None] * other[None, :], along[None, :])
                d1, d2 = line_dist(mm, ra[idx], k[0]), line_dist(mm, rb[idx], k[0])
                ok = (np.abs(d1) < tol) & (np.abs(d2) < tol)
                res[fam] = (idx[ok], j[ok])
            vs = self.vids[v]["vert"]
            if self.mode == "G" and len(vs):
                mv = unit(np.cross(rays(vs[:, 0:2], k), rays(vs[:, 2:4], k)))
                lv = np.hypot(vs[:, 2] - vs[:, 0], vs[:, 3] - vs[:, 1])
                steep = np.abs(vs[:, 3] - vs[:, 1]) > 2.0 * np.abs(vs[:, 2] - vs[:, 0])
                res["V"] = np.nonzero(steep & (np.abs(math.sqrt(2) * lv / 2 * (mv @ e3)) < tol))[0]
            else:
                res["V"] = np.zeros(0, int)
            out.append(res)
        return out

    def residuals(self, x: np.ndarray, square: bool, asg: list) -> np.ndarray:
        k, per = self.unpack(x, square)
        out = []
        for v, ((R, da, db, pa, pb), A) in enumerate(zip(per, asg)):
            e1, e2, e3 = R[:, 0], R[:, 1], R[:, 2]
            seg = self.vids[v]["ceil"]
            for fam, along, other, dd, ph in (("A", e2, e1, da, pa), ("B", e1, e2, db, pb)):
                idx, j = A[fam]
                if not len(idx):
                    continue
                lv = ph + j * dd
                mm = np.cross(e3[None, :] + lv[:, None] * other[None, :], along[None, :])
                out.append(line_dist(mm, rays(seg[idx, 0:2], k), k[0]))
                out.append(line_dist(mm, rays(seg[idx, 2:4], k), k[0]))
            if len(A["V"]):
                vs = self.vids[v]["vert"][A["V"]]
                mv = unit(np.cross(rays(vs[:, 0:2], k), rays(vs[:, 2:4], k)))
                lv = np.hypot(vs[:, 2] - vs[:, 0], vs[:, 3] - vs[:, 1])
                out.append(math.sqrt(2) * lv / 2 * (mv @ e3))
        return np.concatenate(out)

    def bounds(self, square: bool):
        lo = [600.0, 500.0, 300.0, -0.3] + ([0.9] if self.asp else [])
        hi = [1200.0, 1036.0, 850.0, 0.3] + ([1.1] if self.asp else [])
        for _ in self.vids:
            lo += [-0.5, -0.5, -0.5] if self.mode == "G" else [-math.pi * 4]
            hi += [0.5, 0.5, 0.5] if self.mode == "G" else [math.pi * 4]
            nd = 1 if square else 2
            lo += [math.log(0.15)] * nd + [-50.0, -50.0]
            hi += [math.log(1.5)] * nd + [50.0, 50.0]
        return np.array(lo), np.array(hi)

    def x_square(self, x_rect: np.ndarray) -> np.ndarray:
        """矩形格的解 → 正方形格的初值（格距取两族的几何平均）。"""
        out, o = list(x_rect[:self.nk()]), self.nk()
        npose = 3 if self.mode == "G" else 1
        for _ in self.vids:
            out += list(x_rect[o:o + npose])
            la, lb, pa, pb = x_rect[o + npose:o + npose + 4]
            out += [(la + lb) / 2, pa, pb]
            o += npose + 4
        return np.array(out)

    def solve(self, x_rect0: np.ndarray, verbose: bool = False):
        """三段。一：矩形格（两族格距各自），内参放开，逐轮收紧容差定格号；K 错时模型反投的是真方格的轴对齐仿射，矩形格能贴上，
        格号就不会错。二：换成正方形格，沿用第一段的格号与内点，内参放开，Cauchy 尺度 5 → 2 → 1 px。三：在收敛的模型下重新指派
        （格号重新取整），容差 2 → 1.5 px 各解一次。返回 (x, asg, cost, 用到的观测数)。"""
        x = x_rect0.copy()
        for r, tol in enumerate((8.0, 4.0, 3.0, 2.0)):
            asg = self.assign(x, False, tol, FAM_DEG0 if r == 0 else FAM_DEG)
            x = least_squares(lambda z: self.residuals(z, False, asg), x, loss="cauchy", f_scale=2.0, x_scale="jac",
                              max_nfev=400, bounds=self.bounds(False)).x
            if verbose:
                print(f"    矩形格第 {r + 1} 轮（容差 {tol} px）：f {x[0]:.1f} cx {x[1]:.1f} cy {x[2]:.1f}；内点 {self.count(asg)}")
        asg = self.assign(x, False, 2.0)
        xs = self.x_square(x)
        for sc in (5.0, 2.0, 1.0):
            xs = least_squares(lambda z: self.residuals(z, True, asg), xs, loss="cauchy", f_scale=sc, x_scale="jac", max_nfev=600,
                               bounds=self.bounds(True)).x
            if verbose:
                print(f"    正方形格（沿用格号，尺度 {sc} px）：f {xs[0]:.1f} cx {xs[1]:.1f} cy {xs[2]:.1f}")
        for tol in (2.0, 1.5):
            asg = self.assign(xs, True, tol)
            sol = least_squares(lambda z: self.residuals(z, True, asg), xs, loss="cauchy", f_scale=SCALE_PX, x_scale="jac",
                                max_nfev=600, bounds=self.bounds(True))
            xs = sol.x
            if verbose:
                print(f"    重新指派（容差 {tol} px）：f {xs[0]:.1f} cx {xs[1]:.1f} cy {xs[2]:.1f}；内点 {self.count(asg)}")
        return xs, asg, float(sol.cost), self.count(asg)

    @staticmethod
    def count(asg: list) -> int:
        return sum(len(a["A"][0]) + len(a["B"][0]) + len(a["V"]) for a in asg)

    def refit(self, x: np.ndarray, asg: list) -> np.ndarray:
        """自助法用：固定指派，从 x 出发重解。"""
        return least_squares(lambda z: self.residuals(z, True, asg), x, loss="cauchy", f_scale=SCALE_PX,
                             x_scale="jac", max_nfev=400, bounds=self.bounds(True)).x


def diagnose(fit: GridFit, x: np.ndarray, eps: list) -> None:
    """最终模型下每族全部候选线段（吊顶区域、方向在 1.5° 内）的格坐标：宽范围周期图的前三个峰，以及 (坐标 − φ)/Δ 小数部分的直方图
    （10 格，−0.5 到 0.5）。格认对了，小数部分在 0 附近尖峰，最强周期就是拟合用的 Δ。"""
    k, per = fit.unpack(x, True)
    for v, (R, da, _, pa, pb) in enumerate(per):
        m, ceil, a, b, _, _ = fit.lattice_coords(v, k, R, fit.vids[v]["ceil"])
        seg = fit.vids[v]["ceil"]
        ln = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
        sn = math.sin(math.radians(FAM_DEG))
        for fam, along, coord, ph in (("A", R[:, 1], a, pa), ("B", R[:, 0], b, pb)):
            c = ceil & (np.abs(m @ along) < sn) & np.isfinite(coord) & (np.abs(coord) < 20)
            ds = np.arange(0.20, 1.00, 0.0005)
            z = np.abs((ln[c][None, :] * np.exp(2j * np.pi * coord[c][None, :] / ds[:, None])).sum(1)) / ln[c].sum()
            peaks = [i for i in range(1, len(ds) - 1) if z[i] >= z[i - 1] and z[i] >= z[i + 1]]
            top = sorted(peaks, key=lambda i: -z[i])[:3]
            frac = ((coord[c] - ph) / da + 0.5) % 1.0 - 0.5
            hist = np.histogram(frac, bins=10, range=(-0.5, 0.5), weights=ln[c])[0]
            print(f"  诊断 ep{eps[v]} {fam} 族：候选 {int(c.sum())} 段；拟合 Δ {da:.4f}；宽范围周期图前三峰 "
                  f"{', '.join(f'{ds[i]:.4f}（{z[i]:.2f}）' for i in top)}；小数部分直方图（按长度加权，%）"
                  f"{' '.join(f'{100 * h / max(hist.sum(), 1e-9):.0f}' for h in hist)}")


def cx_profile(fit: GridFit, x: np.ndarray, asg: list, cxs: list) -> list:
    """cx 固定在给定值、其余参数全放开（同一套指派），返回每个 cx 的 Cauchy 代价与 f、cy。"""
    out = []
    for c in cxs:
        z0 = np.r_[x[:1], x[2:]]
        lo, hi = fit.bounds(True)
        lo, hi = np.r_[lo[:1], lo[2:]], np.r_[hi[:1], hi[2:]]
        sol = least_squares(lambda z: fit.residuals(np.r_[z[:1], c, z[1:]], True, asg), z0, loss="cauchy", f_scale=SCALE_PX,
                            x_scale="jac", max_nfev=800, bounds=(lo, hi))
        out.append((c, float(sol.cost), sol.x[0], sol.x[1]))
    return out


def rectify(ep: int, bg: np.ndarray, fit: GridFit, x: np.ndarray, v: int, tag: str) -> None:
    """吊顶俯视图（不进 git）：按模型把吊顶平面上 1 cm 一格的点投回画面取样，叠上每个格距一条灰线，写 ceiling_rect_ep<N>_<tag>.png。
    模型对时瓷砖是正方形、灯板是 1:2 的长方形、龙骨远近都落在灰线上。"""
    import cv2
    k, per = fit.unpack(x, True)
    R, da, _, pa, pb = per[v]
    e1, e2, e3 = R[:, 0], R[:, 1], R[:, 2]
    h = TILES_M[0] / da
    seg = fit.vids[v]["ceil"]
    _, ceil, a, b, _, _ = fit.lattice_coords(v, k, R, seg)
    ok = ceil & np.isfinite(a) & np.isfinite(b) & (np.abs(a) < 12) & (np.abs(b) < 12)
    (a0, a1), (b0, b1) = np.percentile(a[ok], [1, 99]) * h, np.percentile(b[ok], [1, 99]) * h
    res = 0.01
    A, B = np.meshgrid(np.arange(a0, a1, res), np.arange(b0, b1, res))
    X = e3[None, None, :] + (A / h)[..., None] * e1 + (B / h)[..., None] * e2
    p = project(X.reshape(-1, 3), k).reshape(A.shape + (2,)).astype(np.float32)
    front = (X[..., 2] > 0.2)
    p[~front] = -1
    img = cv2.remap(bg, p[..., 0], p[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    for j in range(int(math.floor((a0 / h - pa) / da)), int(math.ceil((a1 / h - pa) / da)) + 1):
        col = int(round(((pa + j * da) * h - a0) / res))
        if 0 <= col < img.shape[1]:
            img[:, col] = (img[:, col] * 0.5 + np.array([0, 160, 0]) * 0.5).astype(np.uint8)
    for j in range(int(math.floor((b0 / h - pb) / da)), int(math.ceil((b1 / h - pb) / da)) + 1):
        row = int(round(((pb + j * da) * h - b0) / res))
        if 0 <= row < img.shape[0]:
            img[row, :] = (img[row, :] * 0.5 + np.array([0, 0, 200]) * 0.5).astype(np.uint8)
    cv2.imwrite(str(OUT / f"ceiling_rect_ep{ep}_{tag}.png"), img)


def vertical_tilt(fit: GridFit, x: np.ndarray, eps: list, tag: str) -> None:
    """模型下全部陡线段（|dy| > 2|dx|、长 ≥ 40 px，不按容差挑）相对 e3 的带符号倾斜：线段从上往下定向，解释平面法向 m = r_上 × r_下，
    m·e3 的反正弦（度）。按长度加权的中位数与四分位：竖直结构与 e3 一致时中位数在 0 附近。符号（推导见 agent-doc/experiments/
    2026-10-16-t3-world-frame.md）：线段下端相对模型竖直方向往 +x 偏时为负，即竖直线实际汇聚在模型竖直消失点右边；按“竖直消失点横坐标
    = cx + f·e3_x/e3_z”折成竖直线自己要的 cx 比模型大约 −θ·f/|e3_z|（θ 取弧度）。只用 |m·e3| < 1° 的线段算（排除斜撑、三脚架腿）。"""
    k, per = fit.unpack(x, True)
    for v, (R, *_) in enumerate(per):
        vs = fit.vids[v]["vert"]
        vs = vs[np.abs(vs[:, 3] - vs[:, 1]) > 2.0 * np.abs(vs[:, 2] - vs[:, 0])]
        top = np.where((vs[:, 1] <= vs[:, 3])[:, None], vs[:, 0:2], vs[:, 2:4])
        bot = np.where((vs[:, 1] <= vs[:, 3])[:, None], vs[:, 2:4], vs[:, 0:2])
        m = unit(np.cross(rays(top, k), rays(bot, k)))
        t = np.degrees(np.arcsin(np.clip(m @ R[:, 2], -1, 1)))
        ln = np.hypot(vs[:, 2] - vs[:, 0], vs[:, 3] - vs[:, 1])
        ok = np.abs(t) < 1.0
        o = np.argsort(t[ok])
        cw = np.cumsum(ln[ok][o]) / ln[ok].sum()
        q = [t[ok][o][np.searchsorted(cw, c)] for c in (0.25, 0.5, 0.75)]
        print(f"  竖直线倾斜 [{tag}] ep{eps[v]}：{int(ok.sum())} 段（|倾斜| < 1°），按长度加权的四分位 {q[0]:+.3f}° / {q[1]:+.3f}° / {q[2]:+.3f}°；"
              f"竖直线自己要的 cx 比模型大约 {-q[1] * math.pi / 180 * k[0] / abs(R[2, 2]):+.0f} px")


def rectify_floor(ep: int, bg: np.ndarray, k: tuple, R: np.ndarray, n: np.ndarray, d: float, tag: str) -> None:
    """地面俯视图（不进 git）：地面在相机坐标系里是 n·X = −d（发布的平面，米制），e1、e2 取方格方向并对 n 正交化；按 1 cm 一格投回画面
    取样，叠 1 m 一条的灰线，写 floor_rect_ep<N>_<tag>.png。与吊顶无关的独立检验：K 对时地毯块、胶带标记、墙脚线成直角、尺寸是整齐的米制数。"""
    import cv2
    e1 = unit(R[:, 0] - (R[:, 0] @ n) * n)
    e2 = np.cross(n, e1)
    pix = np.stack(np.meshgrid(np.arange(0, W, 8), np.arange(560, H, 8)), -1).reshape(-1, 2).astype(float)
    r = rays(pix, k)
    down = -(r @ n)
    r, down = r[down > 0.02], down[down > 0.02]
    X = r * (d / down)[:, None]
    a, b = X @ e1, X @ e2
    (a0, a1), (b0, b1) = np.percentile(a, [1, 99]), np.percentile(b, [1, 99])
    a0, a1, b0, b1 = max(a0, -8), min(a1, 8), max(b0, -12), min(b1, 12)
    res = 0.01
    A, B = np.meshgrid(np.arange(a0, a1, res), np.arange(b0, b1, res))
    P = -d * n[None, None, :] + A[..., None] * e1 + B[..., None] * e2
    q = project(P.reshape(-1, 3), k).reshape(A.shape + (2,)).astype(np.float32)
    q[P[..., 2] < 0.2] = -1
    img = cv2.remap(bg, q[..., 0], q[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    for t in np.arange(math.ceil(a0), a1, 1.0):
        img[:, int(round((t - a0) / res))] = (120, 120, 120)
    for t in np.arange(math.ceil(b0), b1, 1.0):
        img[int(round((t - b0) / res)), :] = (120, 120, 120)
    cv2.imwrite(str(OUT / f"floor_rect_ep{ep}_{tag}.png"), img)


def periodogram(coord: np.ndarray, w: np.ndarray) -> tuple:
    """一族格线的格距与相位：在 D_RANGE 里扫周期，取加权相干和最大的那个。"""
    ds = np.arange(D_RANGE[0], D_RANGE[1], 0.0005)
    z = (w[None, :] * np.exp(2j * np.pi * coord[None, :] / ds[:, None])).sum(1)
    i = int(np.argmax(np.abs(z)))
    return ds[i], (np.angle(z[i]) / (2 * np.pi)) * ds[i], float(np.abs(z[i]) / w.sum())


def init_x(fit: GridFit, k0: tuple, poses0: list) -> np.ndarray:
    """内参与姿态取曼哈顿拟合的结果；每段视频的两族格距与相位用周期图起步。"""
    x = list(k0[:4]) + ([1.0] if fit.asp else [])
    for v, R in enumerate(poses0):
        x += [0.0, 0.0, 0.0] if fit.mode == "G" else [poses0_yaw(R, fit.n)]
        m, ceil, a, b, _, _ = fit.lattice_coords(v, (*k0[:4], 1.0), R, fit.vids[v]["ceil"])
        seg = fit.vids[v]["ceil"]
        ln = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
        s = math.sin(math.radians(FAM_DEG0))
        ia = ceil & (np.abs(m @ R[:, 1]) < s) & np.isfinite(a) & (np.abs(a) < 20)
        ib = ceil & (np.abs(m @ R[:, 0]) < s) & np.isfinite(b) & (np.abs(b) < 20)
        da, pa, qa = periodogram(a[ia], ln[ia])
        db, pb, qb = periodogram(b[ib], ln[ib])
        fit.vids[v]["init"] = (da, db, qa, qb, int(ia.sum()), int(ib.sum()))
        x += [math.log(da), math.log(db), pa, pb]
    return np.array(x)


def poses0_yaw(R: np.ndarray, n: np.ndarray) -> float:
    t = np.cross(n, [0.0, 0.0, 1.0])
    t /= np.linalg.norm(t)
    u = np.cross(n, t)
    a = R[:, 0] - (R[:, 0] @ n) * n
    return math.atan2(a @ u, a @ t)


def summary(tag: str, fit: GridFit, x: np.ndarray, asg: list, cost: float, used: int) -> dict:
    k, per = fit.unpack(x, True)
    ang = [mc.angle_deg(R[:, 2], fit.n) for R, *_ in per]
    hs = [[t / da for t in TILES_M] for _, da, _, _, _ in per]
    nA = [len(a["A"][0]) for a in asg]
    nB = [len(a["B"][0]) for a in asg]
    nV = [len(a["V"]) for a in asg]
    r = fit.residuals(x, True, asg)
    print(f"  {tag}：f {k[0]:.1f}、cx {k[1]:.1f}、cy {k[2]:.1f}、k1 {k[3]:+.4f}" + (f"、纵横比 {k[4]:.4f}" if fit.asp else "")
          + f"；代价 {cost:.1f}，观测 {used} 个（每段视频 A 族/B 族/竖直：{', '.join(f'{p}/{q}/{s}' for p, q, s in zip(nA, nB, nV))}），"
          f"残差中位 {np.median(np.abs(r)):.2f} px")
    print(f"    竖直方向 e3 与 n 的夹角 {', '.join(f'{a:.2f}°' for a in ang)}；吊顶离相机（格距 0.6096 / 0.600 m）"
          f"{', '.join(f'{h[0]:.3f}/{h[1]:.3f}' for h in hs)} m")
    return {"f": k[0], "cx": k[1], "cy": k[2], "k1": k[3], "asp": k[4], "angles": ang, "h": hs}


# ---------- 真数据 ----------

def lsd(bg: np.ndarray, min_len: float) -> np.ndarray:
    import cv2
    gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    lines = cv2.createLineSegmentDetector().detect(gray)[0].reshape(-1, 4).astype(float)
    ln = np.hypot(lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1])
    return lines[ln >= min_len]


def overlay(ep: int, bg: np.ndarray, fit: GridFit, x: np.ndarray, asg: list, v: int) -> None:
    """两张图（不进 git）：ceiling_raw_ep<N>.png 是背景的吊顶区域原图（第 0–520 行），ceiling_grid_ep<N>.png 在同一区域画拟合的格线
    （绿：A 族，红：B 族；只画在画面内、地平线以上的连续段）和用上的线段（黄：A 族，蓝：B 族）。"""
    import cv2
    k, per = fit.unpack(x, True)
    R, da, _, pa, pb = per[v]
    e1, e2, e3 = R[:, 0], R[:, 1], R[:, 2]
    top = bg[:520].copy()
    OUT.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT / f"ceiling_raw_ep{ep}.png"), top)
    img = top.copy()
    t = np.linspace(-25, 25, 4000)
    for j in range(-60, 61):
        for base, along, col in ((e3 + (pa + j * da) * e1, e2, (0, 255, 0)), (e3 + (pb + j * da) * e2, e1, (0, 0, 255))):
            X = base[None, :] + t[:, None] * along[None, :]
            good = X[:, 2] > 0.2
            p = np.full((len(X), 2), np.nan)
            p[good] = project(X[good], k)
            good &= (p[:, 0] > 0) & (p[:, 0] < W) & (p[:, 1] > 0) & (p[:, 1] < 520)
            for i in range(len(X) - 1):
                if good[i] and good[i + 1]:
                    cv2.line(img, (int(p[i, 0]), int(p[i, 1])), (int(p[i + 1, 0]), int(p[i + 1, 1])), col, 1)
    seg = fit.vids[v]["ceil"]
    for fam, col in (("A", (0, 220, 255)), ("B", (255, 140, 0))):
        for i in asg[v][fam][0]:
            cv2.line(img, tuple(int(q) for q in seg[i, 0:2]), tuple(int(q) for q in seg[i, 2:4]), col, 2)
    cv2.imwrite(str(OUT / f"ceiling_grid_ep{ep}.png"), img)


def run(eps: list, boot: int, asp: bool, want_overlay: bool, profile: bool = False) -> int:
    import t3_horizon_check as hc
    n = mc.floor_normal()
    bgs = [hc.background(mc.VID / f"episode_{e:06d}.mp4") for e in eps]
    man = [mc.load_segments(e) for e in eps]
    segs_m, vps_all = [d[0] for d in man], [d[1] for d in man]
    fb = mc.Fit(segs_m, "B", n)
    xb, _, _ = fb.solve(mc.x0_for("B", segs_m, vps_all, n))
    (f0, cx0, cy0, k10, _), poses0 = fb.unpack(xb)
    print(f"== 视频 {eps}；初值取曼哈顿拟合 B：f {f0:.1f}、cx {cx0:.1f}、cy {cy0:.1f}、k1 {k10:+.4f}；n = ({n[0]:+.5f}, {n[1]:+.5f}, {n[2]:+.5f})")
    vids = [{"ceil": lsd(bg, MIN_LEN), "vert": s} for bg, s in zip(bgs, segs_m)]
    results = {}
    for mode in ("G", "GB"):
        fit = GridFit(vids, mode, n, poses0, asp=asp and mode == "G")
        x0 = init_x(fit, (f0, cx0, cy0, k10), poses0)
        for v, e in enumerate(eps):
            da, db, qa, qb, na, nb = vids[v]["init"]
            print(f"  [{mode}] ep{e} 周期图起步：A 族 {na} 段，格距 {da:.4f}（相干 {qa:.2f}）；B 族 {nb} 段，格距 {db:.4f}（相干 {qb:.2f}）")
        x, asg, cost, used = fit.solve(x0, verbose=True)
        results[mode] = (fit, x, asg, summary(f"拟合 {mode}（{'姿态自由 + 竖直线' if mode == 'G' else '竖直 = n'}）", fit, x, asg, cost, used))
        diagnose(fit, x, eps)
        vertical_tilt(fit, x, eps, mode)
    if want_overlay:
        fit, x, asg, _ = results["G"]
        for v, e in enumerate(eps):
            overlay(e, bgs[v], fit, x, asg, v)
            rectify(e, bgs[v], fit, x, v, "G")
            rectify(e, bgs[v], results["GB"][0], results["GB"][1], v, "GB")
            kgb, pergb = results["GB"][0].unpack(results["GB"][1], True)
            rectify_floor(e, bgs[v], kgb, pergb[v][0], n, D_FLOOR, "GB")
            rectify_floor(e, bgs[v], (f0, cx0, cy0, k10, 1.0), poses0[v], n, D_FLOOR, f"manhattanB_cx{cx0:.0f}")
        print(f"  叠加图：{OUT}/ceiling_grid_ep*.png 与 ceiling_raw_ep*.png（绿、红是拟合的两族格线，黄、蓝是用上的线段）")
    if profile:
        fit, x, asg, _ = results["G"]
        prof = cx_profile(fit, x, asg, list(range(660, 881, 20)))
        c_best = min(prof, key=lambda r: r[1])
        print("  cx 剖面（拟合 G，同一套指派，cx 固定、其余放开；代价是 Cauchy 代价，括号里是比最低处高多少、f、cy）：")
        for c, cost, f_, cy_ in prof:
            print(f"    cx {c}：{cost:.1f}（+{cost - c_best[1]:.1f}）f {f_:.1f} cy {cy_:.1f}")
        if want_overlay:   # cx 固定在曼哈顿拟合 B 的值时的俯视图，对照用
            z0 = np.r_[x[:1], x[2:]]
            lo, hi = fit.bounds(True)
            zb = least_squares(lambda z: fit.residuals(np.r_[z[:1], cx0, z[1:]], True, asg), z0, loss="cauchy", f_scale=SCALE_PX,
                               x_scale="jac", max_nfev=800, bounds=(np.r_[lo[:1], lo[2:]], np.r_[hi[:1], hi[2:]])).x
            xb_ = np.r_[zb[:1], cx0, zb[1:]]
            for v, e in enumerate(eps):
                rectify(e, bgs[v], fit, xb_, v, f"cx{cx0:.0f}")
    if len(eps) > 1:   # 每段视频单独拟合：cx 的极差
        cxs = []
        for v, e in enumerate(eps):
            fit1 = GridFit([dict(vids[v])], "G", n, [poses0[v]])
            x1, _, _, _ = fit1.solve(init_x(fit1, (f0, cx0, cy0, k10), [poses0[v]]))
            cxs.append(x1[1])
        print(f"  每段视频单独拟合 G 的 cx：{', '.join(f'{c:.1f}' for c in cxs)}；极差 {max(cxs) - min(cxs):.1f} px")
    if boot:
        rng = np.random.default_rng(2)
        for mode in ("G", "GB"):
            fit, x, asg, _ = results[mode]
            st = {"f": [], "cx": [], "cy": [], "k1": [], "ang": []}
            for _ in range(boot):
                bv, ba = [], []
                for v, a in zip(fit.vids, asg):
                    nv = {"ceil": v["ceil"], "vert": v["vert"]}
                    na = {}
                    for fam in ("A", "B"):
                        idx, j = a[fam]
                        pick = rng.integers(0, len(idx), len(idx)) if len(idx) else np.zeros(0, int)
                        na[fam] = (idx[pick], j[pick])
                    na["V"] = a["V"][rng.integers(0, len(a["V"]), len(a["V"]))] if len(a["V"]) else a["V"]
                    bv.append(nv)
                    ba.append(na)
                fb_ = GridFit(bv, mode, n, poses0, asp=fit.asp)
                xb_ = fb_.refit(x, ba)
                k, per = fb_.unpack(xb_, True)
                st["f"].append(k[0]); st["cx"].append(k[1]); st["cy"].append(k[2]); st["k1"].append(k[3])
                st["ang"].append(float(np.mean([mc.angle_deg(R[:, 2], n) for R, *_ in per])))
            q = {kk: np.percentile(vv, [2.5, 50, 97.5]) for kk, vv in st.items()}
            for kk, (lo, mid, hi) in q.items():
                print(f"  自助法 {boot} 次 [{mode}] {kk}：中位 {mid:.3f}，95% 区间 [{lo:.3f}, {hi:.3f}]，宽 {hi - lo:.3f}")
            results[mode] = results[mode] + (q,)
        qg, qgb = results["G"][4], results["GB"][4]
        w = qg["cx"][2] - qg["cx"][0]
        print(f"  判读 1：{'cx 定住了' if w <= 20 else 'cx 仍定不住'}（G 的 cx 区间宽 {w:.1f} px，规则 ≤ 20 px）")
        cg, cgb = results["G"][3]["cx"], results["GB"][3]["cx"]
        print(f"  判读 2：{'两种补法一致' if qg['cx'][0] <= cgb <= qg['cx'][2] else '两种补法有分歧'}（G {cg:.1f}，GB {cgb:.1f}）")
        ang = qg["ang"][1]
        print(f"  判读 3：{'吊顶与世界坐标系的竖直方向一致' if qg['ang'][2] <= 0.5 else '吊顶竖直方向与 n 不一致或说不清'}（夹角中位 {ang:.2f}°，"
              f"95% 上界 {qg['ang'][2]:.2f}°，规则上界 ≤ 0.5°）")
    print("CEILING_GRID_DONE")
    return 0


def reorder_up(R: np.ndarray, n: np.ndarray) -> np.ndarray:
    """曼哈顿拟合 A 的姿态（三列次序、符号任意）→ 第 2 列竖直朝上（与 n 同侧），第 0 列取剩下两列的第一列，第 1 列 = e3 × e0，右手系。"""
    j = int(np.argmax(np.abs(R.T @ n)))
    up = R[:, j] * np.sign(R[:, j] @ n)
    e0 = R[:, [k for k in range(3) if k != j][0]]
    e0 = unit(e0 - (e0 @ up) * up)
    return np.c_[e0, np.cross(up, e0), up]


def pose_from_vps(seg: np.ndarray, f: float, cx: float, cy: float) -> np.ndarray:
    """RANSAC 消失点（t3_horizon_check.find_vps）里，支持线段最平的两个当水平消失点，它们的连线是地平线 l，朝上方向 ∝ Kᵀl（取朝 −y 的一侧）；
    两个里支持线段总长大的当第一个水平方向，正交化成 [e0, e3 × e0, e3]。不用竖直消失点：非主机位支持它的线段少（ep12 只有 14 条），
    估出的位置离谱（第 1832 行，折合俯仰 34°）。"""
    import t3_horizon_check as hc
    vps = hc.find_vps(seg, np.random.default_rng(0))
    for v in vps:
        q = v["vp"][:2] / v["vp"][2] if abs(v["vp"][2]) > 1e-9 else np.array([np.inf, np.inf])
        print(f"    消失点：像素 ({q[0]:.0f}, {q[1]:.0f})，支持线段 {v['n']} 条、总长 {v['len']:.0f} px，支持线段相对水平的中位角 {v['steep']:.1f}°")
    Ki = np.linalg.inv(np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]]))
    ja, jb = sorted(range(len(vps)), key=lambda j: vps[j]["steep"])[:2]
    Kt = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    up = unit(Kt.T @ np.cross(vps[ja]["vp"], vps[jb]["vp"]))
    up = up if up[1] < 0 else -up
    jh = ja if vps[ja]["len"] >= vps[jb]["len"] else jb
    e0 = Ki @ vps[jh]["vp"]
    e0 = unit(e0 - (e0 @ up) * up)
    print(f"    初值姿态：朝上 {np.array2string(up, precision=4)}、第一水平方向 {np.array2string(e0, precision=4)}")
    return np.c_[e0, np.cross(up, e0), up]


def run_second(eps: list, boot: int, want_overlay: bool, fix_delta: float = None) -> int:
    """第二机位、侧机位：发布的地面法向在世界（主机位）坐标系里，不是这台相机的，所以初值取曼哈顿拟合 A（姿态自由），方格只跑拟合 G。
    n（主机位的地面法向）只用来挑哪一列是竖直方向（这些相机也大致水平，−y 朝上），e3 与 n 的夹角对它们没有检验意义。
    另报这台相机自己的俯仰、横滚（由方格估的 e3），以及方格方向 e1、e2、e3 在它坐标系里的表示，供两台相机经方格求相对转动。"""
    import t3_horizon_check as hc
    n = mc.floor_normal()
    bgs = [hc.background(mc.VID / f"episode_{e:06d}.mp4") for e in eps]
    man = [mc.load_segments(e) for e in eps]
    segs_m, vps_all = [d[0] for d in man], [d[1] for d in man]
    # 初值：内参取画面中心与 f 860（不用主机位的结果，免得把它带过去）；姿态由 RANSAC 消失点直接算（单段视频的曼哈顿拟合 A 在 ep12 上
    # 发散到 f → 0，不能做初值）
    f0, cx0, cy0, k10 = 860.0, (W - 1) / 2, (H - 1) / 2, 0.0
    poses0 = [pose_from_vps(sg, f0, cx0, cy0) for sg in segs_m]
    print(f"== 视频 {eps}（非主机位）；初值：f {f0:.1f}、主点 ({cx0:.1f}, {cy0:.1f})、k1 0，姿态由消失点算")
    vids = [{"ceil": lsd(bg, MIN_LEN), "vert": s} for bg, s in zip(bgs, segs_m)]
    fit = GridFit(vids, "G", n, poses0, fixed_delta=fix_delta)
    if fix_delta:
        print(f"  格距固定为 {fix_delta}（两台相机等高时离吊顶的距离相等，取主机位的值；只用在正方形格阶段）")
    x0 = init_x(fit, (f0, cx0, cy0, k10), poses0)
    for v, e in enumerate(eps):
        da, db, qa, qb, na, nb = vids[v]["init"]
        print(f"  [G] ep{e} 周期图起步：A 族 {na} 段，格距 {da:.4f}（相干 {qa:.2f}）；B 族 {nb} 段，格距 {db:.4f}（相干 {qb:.2f}）")
    x, asg, cost, used = fit.solve(x0, verbose=True)
    summary("拟合 G（姿态自由 + 竖直线；e3 与 n 的夹角对非主机位无检验意义）", fit, x, asg, cost, used)
    diagnose(fit, x, eps)
    vertical_tilt(fit, x, eps, "G")
    k, per = fit.unpack(x, True)
    for v, e in enumerate(eps):
        R = per[v][0]
        up = R[:, 2]
        print(f"  ep{e} 相机自己的姿态（由方格的 e3）：俯仰 {math.degrees(math.asin(-up[2])):+.2f}°（光轴在水平面下为正），"
              f"横滚 {math.degrees(math.atan2(up[0], -up[1])):+.2f}°；e1 {np.array2string(R[:, 0], precision=5)}、"
              f"e2 {np.array2string(R[:, 1], precision=5)}、e3 {np.array2string(up, precision=5)}；格距 {per[v][1]:.4f}、相位 ({per[v][3]:.4f}, {per[v][4]:.4f})")
    if want_overlay:
        for v, e in enumerate(eps):
            overlay(e, bgs[v], fit, x, asg, v)
            rectify(e, bgs[v], fit, x, v, "G")
    if len(eps) > 1:
        cxs = []
        for v in range(len(eps)):
            fit1 = GridFit([dict(vids[v])], "G", n, [poses0[v]])
            cxs.append(fit1.solve(init_x(fit1, (f0, cx0, cy0, k10), [poses0[v]]))[0][1])
        print(f"  每段视频单独拟合 G 的 cx：{', '.join(f'{c:.1f}' for c in cxs)}；极差 {max(cxs) - min(cxs):.1f} px")
    if boot:
        rng = np.random.default_rng(3)
        st = {"f": [], "cx": [], "cy": []}
        for _ in range(boot):
            ba = []
            for a in asg:
                na = {}
                for fam in ("A", "B"):
                    idx, j = a[fam]
                    pick = rng.integers(0, len(idx), len(idx)) if len(idx) else np.zeros(0, int)
                    na[fam] = (idx[pick], j[pick])
                na["V"] = a["V"][rng.integers(0, len(a["V"]), len(a["V"]))] if len(a["V"]) else a["V"]
                ba.append(na)
            xb_ = fit.refit(x, ba)
            st["f"].append(xb_[0]); st["cx"].append(xb_[1]); st["cy"].append(xb_[2])
        for kk, vv in st.items():
            lo, mid, hi = np.percentile(vv, [2.5, 50, 97.5])
            print(f"  自助法 {boot} 次 [G] {kk}：中位 {mid:.3f}，95% 区间 [{lo:.3f}, {hi:.3f}]，宽 {hi - lo:.3f}")
    print("CEILING_GRID_DONE")
    return 0


# ---------- 自检：合成吊顶 ----------

def synth(rng, k: tuple, R: np.ndarray, h: float, tile: tuple, phase: tuple, n_out: float = 0.15, edge: float = 0.012,
          inset: float = 0.04, noise: float = 0.4, n_ceil: int = 400, n_vert: int = 150) -> dict:
    """吊顶方格线段按画面吊顶区域均匀撒点（第 0–420 行，水平距离 ≤ 15 m），每点吸附到最近一条格线，沿格线取 0.3–1.2 m 一段；
    偏移含 T 型龙骨两侧边缘（±edge）、30% 灯板内缩（±inset）；端点噪声 noise px；另加 n_out 比例的离群与 n_vert 条墙上竖直线。"""
    e1, e2, e3 = R[:, 0], R[:, 1], R[:, 2]
    ceil, tries = [], 0
    while len(ceil) < n_ceil and tries < 200000:
        tries += 1
        r = rays(np.array([[rng.uniform(0, W), rng.uniform(0, 420)]]), k)[0]
        if r @ e3 <= UP_MIN * np.linalg.norm(r):
            continue
        X = h * r / (r @ e3)
        a, b = X @ e1, X @ e2
        if math.hypot(a, b) > 15.0:
            continue
        off = rng.choice([-edge, edge]) + (rng.choice([-inset, inset]) if rng.random() < 0.3 else 0.0)
        if rng.random() < 0.5:   # A 族：沿 e2，a = φa + i·格距
            c = phase[0] + round((a - phase[0]) / tile[0]) * tile[0]
            P, along = h * e3 + (c + off) * e1 + b * e2, e2
        else:
            c = phase[1] + round((b - phase[1]) / tile[1]) * tile[1]
            P, along = h * e3 + a * e1 + (c + off) * e2, e1
        ln = rng.uniform(0.3, 1.2)
        P0, P1 = P - ln / 2 * along, P + ln / 2 * along
        if P0[2] < 0.3 or P1[2] < 0.3:
            continue
        p = project(np.vstack([P0, P1]), k) + rng.normal(0, noise, (2, 2))
        if p[:, 0].min() < 0 or p[:, 0].max() >= W or p[:, 1].min() < 0 or p[:, 1].max() >= H or np.hypot(*(p[1] - p[0])) < MIN_LEN:
            continue
        ceil.append(p.ravel())
    ceil = np.array(ceil)
    bad = []
    while len(bad) < int(n_out * len(ceil)):   # 离群：吊顶区域里任意方向的线段
        q = np.c_[rng.uniform(0, W, 2), rng.uniform(0, 420, 2)]
        if np.hypot(*(q[1] - q[0])) >= MIN_LEN:
            bad.append(q.ravel())
    vert = []
    while len(vert) < n_vert:   # 墙上竖直线：离相机 4–9 m
        base = rng.uniform(4, 9) * unit(rng.normal(0, 1, 3) * [1, 0, 1] + [0, 0, 2])
        base = base - (base @ e3) * e3 + rng.uniform(-1.3, 1.2) * e3
        P0, P1 = base, base + rng.uniform(0.3, 1.5) * e3
        if P0[2] < 0.5 or P1[2] < 0.5:
            continue
        q = project(np.vstack([P0, P1]), k) + rng.normal(0, noise, (2, 2))
        if q[:, 0].min() < 0 or q[:, 0].max() >= W or q[:, 1].min() < 0 or q[:, 1].max() >= H or np.hypot(*(q[1] - q[0])) < 40:
            continue
        vert.append(q.ravel())
    return {"ceil": np.vstack([ceil, np.array(bad).reshape(-1, 4)]), "vert": np.array(vert)}


def vp_matched(R: np.ndarray, k_true: tuple, k_init: tuple) -> np.ndarray:
    """真姿态 R 在真内参下的三个消失点，换成初值内参下的方向，再正交化（SVD 取最近的旋转）。"""
    Kt = np.array([[k_true[0], 0, k_true[1]], [0, k_true[0], k_true[2]], [0, 0, 1.0]])
    Ki = np.array([[k_init[0], 0, k_init[1]], [0, k_init[0], k_init[2]], [0, 0, 1.0]])
    M = unit((np.linalg.inv(Ki) @ Kt @ R).T).T
    U, _, Vt = np.linalg.svd(M)
    Q = U @ Vt
    return Q if np.linalg.det(Q) > 0 else -Q


def selftest_debug() -> int:
    """逐项打开扰动看哪一步出问题：真值初值 + 无噪声 → 初值偏 → 加 T 型龙骨边缘 → 加灯板内缩 → 加噪声与离群。"""
    n_true = unit(np.array([-0.0189, -0.9983, -0.0555]))
    k_true = (860.0, 805.0, 528.0, 0.0, 1.0)
    R_true = mc.rot_b(n_true, 0.35)
    for tag, k_init, kw in (("真值初值、理想线段", (860.0, 805.0, 528.0, 0.0), dict(edge=0.0, inset=0.0, noise=0.0, n_out=0.0)),
                            ("初值偏（cx +45、f −20）、理想线段", (840.0, 850.0, 528.0, 0.0), dict(edge=0.0, inset=0.0, noise=0.0, n_out=0.0)),
                            ("初值偏、龙骨两侧边缘", (840.0, 850.0, 528.0, 0.0), dict(edge=0.012, inset=0.0, noise=0.0, n_out=0.0)),
                            ("初值偏、边缘 + 灯板内缩", (840.0, 850.0, 528.0, 0.0), dict(edge=0.012, inset=0.04, noise=0.0, n_out=0.0)),
                            ("初值偏、全部扰动", (840.0, 850.0, 528.0, 0.0), dict(edge=0.012, inset=0.04, noise=0.4, n_out=0.15))):
        rng = np.random.default_rng(100)
        vids = [synth(rng, k_true, R_true, 1.35, (0.6096, 0.6096), (0.13, 0.41), **kw)]
        for mode in ("G", "GB"):
            R0 = vp_matched(R_true, k_true, k_init)
            fit = GridFit([dict(v) for v in vids], mode, n_true, [R0])
            x0 = init_x(fit, k_init, [R0])
            da, db, qa, qb, na, nb = fit.vids[0]["init"]
            print(f"== {tag} [{mode}]：吊顶线段 {len(vids[0]['ceil'])}、竖直 {len(vids[0]['vert'])}；周期图 A {na} 段 Δ {da:.4f}（相干 {qa:.2f}），"
                  f"B {nb} 段 Δ {db:.4f}（相干 {qb:.2f}）；真 Δ {0.6096 / 1.35:.4f}")
            x, asg, cost, used = fit.solve(x0, verbose=True)
            print(f"   结果：cx 误差 {x[1] - 805:+.2f}，f 误差 {x[0] - 860:+.2f}，cy 误差 {x[2] - 528:+.2f}，内点 {used}")
    return 0


def selftest() -> int:
    n_true = unit(np.array([-0.0189, -0.9983, -0.0555]))
    k_true = (860.0, 805.0, 528.0, 0.0, 1.0)
    R_true = mc.rot_b(n_true, 0.35)
    k_init = (840.0, 850.0, 528.0, 0.0)
    rows = {"G": [], "GB": []}
    for seed in range(10):
        rng = np.random.default_rng(100 + seed)
        vids = [synth(rng, k_true, R_true, 1.35, (0.6096, 0.6096), (0.13, 0.41))]
        for mode in ("G", "GB"):
            # 初值的姿态：让初值内参下三个消失点与真值的消失点重合（曼哈顿拟合就是这样），正交化后再加 0.2° 扰动
            R0 = Rotation.from_rotvec(rng.normal(0, math.radians(0.2), 3)).as_matrix() @ vp_matched(R_true, k_true, k_init)
            fit = GridFit([dict(v) for v in vids], mode, n_true, [R0])
            x, asg, cost, used = fit.solve(init_x(fit, k_init, [R0]))
            k, per = fit.unpack(x, True)
            rows[mode].append((k[0] - 860.0, k[1] - 805.0, k[2] - 528.0, used))
    ok = True
    for mode in ("G", "GB"):
        a = np.array(rows[mode])
        good = int(((np.abs(a[:, 1]) <= 5) & (np.abs(a[:, 0]) <= 5)).sum())
        ok = ok and good >= 9
        print(f"  自检 {mode}：10 个种子，cx 误差 {', '.join(f'{v:+.1f}' for v in a[:, 1])}；f 误差 {', '.join(f'{v:+.1f}' for v in a[:, 0])}；"
              f"cy 误差中位 {np.median(a[:, 2]):+.1f}；|cx|、|f| ≤ 5 px 的 {good}/10")
    # 反向对照 1：长方形格（1:1.03）
    rng = np.random.default_rng(7)
    vids = [synth(rng, k_true, R_true, 1.35, (0.6096, 0.6096 * 1.03), (0.13, 0.41))]
    R0 = vp_matched(R_true, k_true, k_init)
    fit = GridFit(vids, "G", n_true, [R0])
    x, *_ = fit.solve(init_x(fit, k_init, [R0]))
    print(f"  反向对照（方格做成 1:1.03 的长方形，按正方形拟合）：G 的 cx 误差 {x[1] - 805:+.1f} px，f 误差 {x[0] - 860:+.1f} px")
    # 反向对照 2：GB 的 n 错 0.1° 横滚
    rng = np.random.default_rng(8)
    vids = [synth(rng, k_true, R_true, 1.35, (0.6096, 0.6096), (0.13, 0.41))]
    n_bad = Rotation.from_rotvec([0, 0, math.radians(0.1)]).apply(n_true)
    R0 = mc.rot_b(n_bad, poses0_yaw(vp_matched(R_true, k_true, k_init), n_bad))
    fit = GridFit(vids, "GB", n_bad, [R0])
    x, *_ = fit.solve(init_x(fit, k_init, [R0]))
    print(f"  反向对照（GB 的 n 错 0.1° 横滚）：GB 的 cx 误差 {x[1] - 805:+.1f} px，f 误差 {x[0] - 860:+.1f} px")
    print("CEILING_SELFTEST_OK" if ok else "CEILING_SELFTEST_FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("eps", nargs="*", type=int)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--debug", action="store_true", help="自检时逐项打开扰动")
    ap.add_argument("--boot", type=int, default=0)
    ap.add_argument("--asp", action="store_true", help="放开纵横比（只对拟合 G）")
    ap.add_argument("--overlay", action="store_true")
    ap.add_argument("--profile", action="store_true", help="cx 剖面：固定 cx、其余放开的代价曲线")
    ap.add_argument("--cam", choices=("main", "other"), default="main", help="other：第二机位、侧机位（初值取画面中心，只跑 G）")
    ap.add_argument("--fix-delta", type=float, default=None, help="非主机位：正方形格阶段把格距固定为这个值（以离吊顶的距离为单位）")
    a = ap.parse_args()
    if a.cam == "other" and not a.selftest:
        sys.exit(run_second(a.eps, a.boot, a.overlay, a.fix_delta))
    sys.exit((selftest_debug() if a.debug else selftest()) if a.selftest else run(a.eps or [5, 11, 20], a.boot, a.asp, a.overlay, a.profile))
