"""Tier 3 世界坐标系的事实之二：exo 画面里的实测地平线，与“世界坐标系就是这台相机的坐标系”所预测的地平线比（Mac 上跑，要 numpy、
opencv-python-headless；只读 exo 视频与发布的地面平面，不读任何轨迹，不读 Tier 1）。

前一步（stages/recovery/t3_world_frame_facts.py，results/t3world/plane_facts.txt）：30 集的地面平面在世界坐标系里是同一个平面，跨 3 个机位、
24 次标定，两两法向最大夹角 0.38°，离地 1.381–1.396 m。所以世界坐标系是整个数据集共用的一个；要判断的是它是不是主机位相机的坐标系。

假设甲（世界 = 这台相机的 OpenCV 坐标系，x 右、y 下、z 前，主点在画面中心、像素为正方形）下，地面平面 n·X + d = 0 的消失线是满足
n·K⁻¹[u, v, 1]ᵀ = 0 的像素：v = cy − (n_x (u − cx) + n_z f) / n_y。它在画面里的倾角与焦距无关，在中心列的高度随焦距变化很小。
实测：每段视频均匀取 31 帧，逐像素取中位数得到背景（人走动，背景不动）；LSD 检出长度 ≥ 40 px 的直线段；按序 RANSAC 找 3 个消失点
（候选来自两两线段的交点，线段方向与“线段中点到消失点”方向的夹角 < 1.5° 记为支持，按长度加权计票，再用加权最小二乘细化），
线段大致竖直的那组记竖直消失点，另两个是水平消失点；两水平消失点的连线就是实测地平线。两水平消失点若对应正交方向，
f² = −(v1 − c)·(v2 − c) 给出焦距；竖直消失点与地平线另给一个焦距（地平线是竖直消失点关于绝对二次曲线像的极线），两者互相核对。
判读（看数据之前登记）：主机位的实测地平线与预测（f 取实测值；实测不可用时取 800–950 px）在中心列相差 ≤ 10 px 且倾角相差 ≤ 0.5°，
记“支持：世界的转动与主机位相机一致”；相差 > 30 px 或倾角相差 > 2°，记“否定”；其余记“说不清”。第二机位是对照：它的预测线也对得上时，
这个检验分不开不同的相机，结论改看相机高度那一步。
输出逐段视频的消失点、实测地平线、预测地平线与判读，叠加图写到 runs/t3world/horizon_<ep>.png（不进 git）。
用法：UV_NO_CONFIG=1 UV_DEFAULT_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple uv run --no-project --python 3.12 --with numpy \
        --with opencv-python-headless python stages/recovery/t3_horizon_check.py 5 11 20 12
最后一行 HORIZON_CHECK_DONE。
"""
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DS = ROOT / "data/challenge/track_2/tier_2_synthetic_noise"
VID = DS / "videos/chunk-000/observation.images.exo_camera"
OUT = ROOT / "runs/t3world"
SECOND, SIDE = {12, 14, 16, 25, 28}, {21}
N_FRAMES, MIN_LEN, INLIER_DEG, N_SAMPLES = 31, 40.0, 1.5, 6000
TOL_ROW_OK, TOL_TILT_OK, TOL_ROW_BAD, TOL_TILT_BAD = 10.0, 0.5, 30.0, 2.0


def background(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in np.linspace(0, n - 1, N_FRAMES).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    if len(frames) < N_FRAMES // 2:
        raise RuntimeError(f"{path.name}：只读到 {len(frames)} 帧")
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def segments(bg: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
    lines = cv2.createLineSegmentDetector().detect(gray)[0].reshape(-1, 4)
    length = np.hypot(lines[:, 2] - lines[:, 0], lines[:, 3] - lines[:, 1])
    return lines[length >= MIN_LEN]


def homog(seg: np.ndarray) -> np.ndarray:
    p1 = np.c_[seg[:, 0], seg[:, 1], np.ones(len(seg))]
    p2 = np.c_[seg[:, 2], seg[:, 3], np.ones(len(seg))]
    l = np.cross(p1, p2)
    return l / np.linalg.norm(l[:, :2], axis=1, keepdims=True)


def support(seg: np.ndarray, vp: np.ndarray) -> np.ndarray:
    """每条线段方向与“中点到消失点”方向的夹角（度）。vp 是齐次坐标，可在无穷远。"""
    mid = np.c_[(seg[:, 0] + seg[:, 2]) / 2, (seg[:, 1] + seg[:, 3]) / 2]
    d_seg = np.c_[seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1]]
    d_vp = vp[:2][None, :] - mid * vp[2]
    cos = np.abs((d_seg * d_vp).sum(1)) / (np.linalg.norm(d_seg, axis=1) * np.linalg.norm(d_vp, axis=1) + 1e-12)
    return np.degrees(np.arccos(np.clip(cos, 0.0, 1.0)))


def refine(L: np.ndarray, w: np.ndarray) -> np.ndarray:
    M = (L * w[:, None]).T @ L
    return np.linalg.eigh(M)[1][:, 0]


def find_vps(seg: np.ndarray, rng: np.random.Generator, k: int = 3) -> list:
    L, length = homog(seg), np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    left, out = np.ones(len(seg), bool), []
    for _ in range(k):
        idx = np.flatnonzero(left)
        if len(idx) < 2:
            break
        best, best_score = None, -1.0
        for _ in range(N_SAMPLES):
            i, j = rng.choice(idx, 2, replace=False)
            vp = np.cross(L[i], L[j])
            if np.linalg.norm(vp) < 1e-12:
                continue
            inl = support(seg[idx], vp) < INLIER_DEG
            score = length[idx][inl].sum()
            if score > best_score:
                best, best_score = vp, score
        inl_mask = np.zeros(len(seg), bool)
        inl_mask[idx[support(seg[idx], best) < INLIER_DEG]] = True
        vp = refine(L[inl_mask], length[inl_mask])
        inl_mask = left & (support(seg, vp) < INLIER_DEG)
        vp = refine(L[inl_mask], length[inl_mask])
        ang = np.degrees(np.arctan2(np.abs(seg[inl_mask, 3] - seg[inl_mask, 1]), np.abs(seg[inl_mask, 2] - seg[inl_mask, 0])))
        out.append({"vp": vp, "n": int(inl_mask.sum()), "len": float(length[inl_mask].sum()), "steep": float(np.median(ang))})
        left &= ~inl_mask
    return out


def plane_normal() -> np.ndarray:
    ns = []
    for line in (DS / "meta/episodes_metadata.jsonl").read_text().splitlines():
        m = json.loads(line)
        ep = int(m["episode_index"])
        if ep in SECOND or ep in SIDE:
            continue
        p = np.array(json.loads((DS / m["ground_plane"]).read_text())["plane"], float)
        p = p / np.linalg.norm(p[:3])
        ns.append(-p[:3] if p[3] < 0 else p[:3])
    n = np.mean(ns, axis=0)
    return n / np.linalg.norm(n)


def predicted_row(n: np.ndarray, f: float, cx: float, cy: float, u: float) -> float:
    return cy - (n[0] * (u - cx) + n[2] * f) / n[1]


def main(eps: list) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    n = plane_normal()
    tilt_pred = math.degrees(math.atan2(-n[0] / n[1], 1.0))   # dv/du = −n_x / n_y
    print(f"== 主机位平均地面法向 n = ({n[0]:+.5f}, {n[1]:+.5f}, {n[2]:+.5f})；假设甲下预测地平线倾角 {tilt_pred:+.2f}°（v 朝下为正）")
    rng = np.random.default_rng(0)
    for ep in eps:
        grp = "second" if ep in SECOND else "side" if ep in SIDE else "main"
        bg = background(VID / f"episode_{ep:06d}.mp4")
        h, w = bg.shape[:2]
        cx, cy = (w - 1) / 2, (h - 1) / 2
        seg = segments(bg)
        vps = find_vps(seg, rng)
        vert = max(vps, key=lambda v: v["steep"])
        hor = [v for v in vps if v is not vert]
        print(f"== ep{ep}（{grp}）{w}×{h}，线段 {len(seg)} 条")
        for v in vps:
            p = v["vp"]
            loc = f"({p[0] / p[2]:.0f}, {p[1] / p[2]:.0f})" if abs(p[2]) > 1e-9 else "无穷远"
            print(f"  消失点 {loc}：支持线段 {v['n']} 条、总长 {v['len']:.0f} px、线段倾角中位 {v['steep']:.1f}°{'（竖直）' if v is vert else ''}")
        hline = np.cross(hor[0]["vp"], hor[1]["vp"])
        row_meas = -(hline[0] * cx + hline[2]) / hline[1]
        tilt_meas = math.degrees(math.atan2(-hline[0] / hline[1], 1.0))
        v1 = hor[0]["vp"][:2] / hor[0]["vp"][2] - [cx, cy]
        v2 = hor[1]["vp"][:2] / hor[1]["vp"][2] - [cx, cy]
        f2 = -float(v1 @ v2)
        f_orth = math.sqrt(f2) if f2 > 0 else float("nan")
        vz = vert["vp"]
        f_pol = float("nan")
        if abs(vz[2]) > 1e-9:   # 地平线是竖直消失点的极线：h ∝ (vx − cx, vy − cy, f² + …)；用中心列的行号解 f
            pv = vz[:2] / vz[2] - [cx, cy]
            # 极线：pv·(x − c) + f² = 0；在 x = (0, row − cy) 处：pv_y (row − cy) + f² = 0
            f_sq = -pv[1] * (row_meas - cy) - pv[0] * 0.0
            f_pol = math.sqrt(f_sq) if f_sq > 0 else float("nan")
        f_use = f_orth if 700 <= f_orth <= 1100 else float("nan")
        rows_pred = {f: predicted_row(n, f, cx, cy, cx) for f in (800.0, 830.0, 950.0)}
        row_pred = predicted_row(n, f_use, cx, cy, cx) if not math.isnan(f_use) else rows_pred[830.0]
        d_row, d_tilt = row_meas - row_pred, tilt_meas - tilt_pred
        if abs(d_row) <= TOL_ROW_OK and abs(d_tilt) <= TOL_TILT_OK:
            verdict = "支持：世界的转动与这台相机一致"
        elif abs(d_row) > TOL_ROW_BAD or abs(d_tilt) > TOL_TILT_BAD:
            verdict = "否定：世界的转动与这台相机不一致"
        else:
            verdict = "说不清"
        print(f"  实测地平线：中心列第 {row_meas:.1f} 行，倾角 {tilt_meas:+.2f}°；焦距（两水平消失点正交）{f_orth:.0f} px，（竖直消失点极线）{f_pol:.0f} px")
        print(f"  预测地平线：中心列第 {row_pred:.1f} 行（f = {f_use if not math.isnan(f_use) else 830.0:.0f}；f 800/830/950 时 "
              f"{rows_pred[800.0]:.1f}/{rows_pred[830.0]:.1f}/{rows_pred[950.0]:.1f}），倾角 {tilt_pred:+.2f}°")
        print(f"  差：行 {d_row:+.1f} px，倾角 {d_tilt:+.2f}° → {verdict}")
        img = bg.copy()
        for x0, y0, x1, y1 in seg.astype(int):
            cv2.line(img, (x0, y0), (x1, y1), (255, 200, 0), 1)
        for row, tilt, col in ((row_pred, tilt_pred, (0, 0, 255)), (row_meas, tilt_meas, (0, 255, 0))):
            s = math.tan(math.radians(tilt))
            cv2.line(img, (0, int(round(row - s * cx))), (w - 1, int(round(row + s * (w - 1 - cx)))), col, 2)
        cv2.imwrite(str(OUT / f"horizon_ep{ep}.png"), img)
        cv2.imwrite(str(OUT / f"bg_ep{ep}.png"), bg)
    print("HORIZON_CHECK_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main([int(x) for x in sys.argv[1:]] or [5, 11, 20, 12]))
