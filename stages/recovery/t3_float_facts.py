"""物体静置时离发布地面多高、离相机多远：看 Tier 3 重建的悬空是常数（尺度、焦距或整体平移）还是随距离增长（倾角）。只读，不用 GPU。

问题、做法与执行前登记的判读：agent-doc/experiments/2026-10-09-t3-float-structure.md。
每集取第 0..F−1 帧里的可见帧（默认 F=5）：物体凸包最低点离发布地面的高度取中位 h。地面平面 [a, b, c, d] 归一化后取 d > 0，
即相机在地面上方，d 就是相机离地高度 H，法向 n 朝上。候选面是地面 0 与支撑面里各几何体的顶面（相对地面的高度），离 h 最近的
当作物体所在的面，间隙 g = h − 面高。物体原点的中位 c（相机系）分解成沿地面的前向距离 Z、横向 X 与离相机的直线距离 D。
只用静置的集（窗口内原点位移不超过 20 mm）拟合：常数 g = o；倾角 g = o + b·Z + e·X（地面上的集，至少 4 集）；尺度 g = s·L + o，
L = −c·n 是相机比物体原点高多少（全部静置集，至少 6 集且 L 的极差 ≥ 0.3 m）。物体原点沿视线缩到 k 倍、网格不变时 g = (1 − k)/k · L，
只缩竖直分量（焦距错、深度对）时也与 L 成正比，所以 k = 1/(1 + s)。凸包与最低点用 nopen.py 的 hull_vertices、min_height；位姿 [x, y, z, qw, qx, qy, qz]，相机（OpenCV）系。
输入两种布局：<目录>/meta/episodes_metadata.jsonl（官方档目录），或 <目录>/meta/episode_<N>.metadata.json（tier3_pack_remote.sh 的包）；
parquet 在 <目录>/data/chunk-000/。网格、地面、支撑面按元数据里的相对路径到 --tier2 目录找（默认 libs/v2d_infra/paths.py 的 TIER2）。
--support-root <目录>：支撑面改到这个目录找（计划 agent-doc/plan-t3-hm-p2-2026-10-12.md：Tier 3 不用官方支撑面，取交付目录转来的文件）；
  第一行另写明支撑面从哪个目录取。不加时与以前相同。
--control：对照用，按登记的对照判据打 CONTROL_OK / CONTROL_FAIL；否则按登记的 Tier 3 判读打一行“判读”。
--all-rest：事后加的（10/9 21:40，看过第一轮结果之后，未登记）：不按位移筛静置，全部集都进拟合；输出里标明，不打判读。
  加它的原因：CARI4D 开头 5 帧的抖动有 9–51 mm，位移判据把 20 集筛到 6 集。开头调 tier_guard.forbid_tier1()。
--offset-gate <mm>：10/10 执行前登记的判据（计划 agent-doc/plan-t3-pilot-v2-2026-10-10.md 第 4 节，复查同组成员的校正版）：
  地面上的集至少 8 集、常数模型 |o| ≤ 门槛打 OFFSET_GATE_PASS，否则 OFFSET_GATE_FAIL；不到 8 集打 OFFSET_GATE_NONE（不判读）。
  这一行同时给 o 的标准误与地面上各集间隙的标准差，两版之间怎么选见 stages/recovery/t3_pick_version.py。
  与 --all-rest 同用时，输出不再标“事后（未登记）”。
用法（训练机，仓库根）：scripts/dev/incontainer stages/recovery/t3_float_facts.py <目录> [--tier2 <目录>] [--frames 5] [--episodes 0,3,...] [--control]
                       [--all-rest] [--offset-gate <mm>] [--support-root <目录>]
测试：stages/recovery/test_t3_float_facts.py。
"""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path, PurePosixPath

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "libs" / "v2d_infra"))
sys.path.insert(0, str(HERE))
from nopen import hull_vertices, min_height  # noqa: E402
from paths import TIER2  # noqa: E402
from tier_guard import forbid_tier1  # noqa: E402

STATIC_MM = 20.0
WIDE = 30           # 只报：第 0–29 帧的间隙


def plane_up(plane):
    a = np.asarray(plane, dtype=np.float64)
    s = np.linalg.norm(a[:3])
    n, d = a[:3] / s, a[3] / s
    if d < 0:                       # 相机（原点）在地面上方：d 是相机离地高度
        n, d = -n, -d
    return n, d


def floor_axes(n):
    u = np.array([0.0, 0.0, 1.0]) - n[2] * n      # 相机前向投到地面上
    u /= np.linalg.norm(u)
    return u, np.cross(u, n)                         # 前向、右方（OpenCV 系里约为 +z、+x）


def surface_tops(text: str) -> list[float]:
    """support usda（只有 Cylinder / Cube，无旋转）→ 各几何体顶面高度。与 support_bridge_v2.surface_tops 相同；那个模块属于拿 Tier 1 做诊断的分析脚本，这里不 import。"""
    tops = []
    for blk in re.split(r"\bdef (?=Cylinder|Cube)", text)[1:]:
        z = float(re.search(r"xformOp:translate = \(([^)]*)\)", blk).group(1).split(",")[2])
        if blk.startswith("Cylinder"):
            tops.append(z + float(re.search(r"double height = ([\d.eE+-]+)", blk).group(1)) / 2)
        else:
            size = float(re.search(r"double size = ([\d.eE+-]+)", blk).group(1))
            sc = re.search(r"xformOp:scale = \(([^)]*)\)", blk)
            tops.append(z + size * (float(sc.group(1).split(",")[2]) if sc else 1.0) / 2)
    return tops


def episode_facts(p, vis, V, plane, tops, frames=5):
    """p (T,7) 相机系位姿，vis (T,) 可见，V 凸包顶点，plane [a,b,c,d]，tops 支撑面顶面高度。窗口里没有可见帧时返回 None。"""
    idx = np.flatnonzero(np.asarray(vis[:frames], dtype=bool))
    if idx.size == 0:
        return None
    n, H = plane_up(plane)
    h = float(np.median(min_height(p[idx], V, n, H)))
    c = np.median(p[idx, :3], axis=0)
    top = min([0.0] + list(tops), key=lambda t: abs(h - t))
    u, r = floor_axes(n)
    wide = np.flatnonzero(np.asarray(vis[:WIDE], dtype=bool))
    return {"h": h, "top": top, "gap": h - top, "H": float(H), "Z": float(c @ u), "X": float(c @ r),
            "D": float(np.linalg.norm(c)), "L": float(-(c @ n)), "disp_mm": float(np.linalg.norm(p[idx, :3] - p[idx[0], :3], axis=1).max() * 1000),
            "n_frames": int(idx.size), "on_floor": top == 0.0,
            "gap_wide": float(np.median(min_height(p[wide], V, n, H))) - top}


def ols(y, cols):
    A = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ beta
    s2 = float(res @ res) / max(1, len(y) - A.shape[1])
    return beta, np.sqrt(np.clip(np.diag(s2 * np.linalg.pinv(A.T @ A)), 0, None)), float(res @ res)


def fit(rows, static_mm=STATIC_MM):
    st = [r for r in rows if r["disp_mm"] <= static_mm]
    fl = [r for r in st if r["on_floor"]]
    out = {"n_static": len(st), "n_floor": len(fl), "n_support": len(st) - len(fl)}
    if len(fl) >= 4:
        g = np.array([r["gap"] for r in fl])
        one = np.ones_like(g)
        out["const"] = ols(g, [one])
        out["tilt"] = ols(g, [one, np.array([r["Z"] for r in fl]), np.array([r["X"] for r in fl])])
        out["floor_abs_median"] = float(np.median(np.abs(g)))
        out["floor_sd"] = float(np.std(g, ddof=1))
    L = np.array([r["L"] for r in st])
    if len(st) >= 6 and L.max() - L.min() >= 0.3:
        g = np.array([r["gap"] for r in st])
        out["scale"] = ols(g, [L, np.ones_like(g)])
        out["L_median"] = float(np.median(L))
    return out


def tilt_deg(out):
    (_, b, e), _, _ = out["tilt"]
    return float(np.degrees(np.arctan(b))), float(np.degrees(np.arctan(e)))


def reading(out):
    """登记的 Tier 3 判读：倾角型 / 常数型 / 说不清；拟合了尺度模型时另给 尺度 / 整体平移 / 分不开。"""
    if "tilt" not in out:
        return "地面上静置的集不到 4 集，不判读", None
    (o0,), _, ss0 = out["const"]
    _, _, ss1 = out["tilt"]
    pitch, roll = tilt_deg(out)
    drop = 1 - ss1 / ss0 if ss0 > 0 else 0.0
    if (abs(pitch) >= 1 or abs(roll) >= 1) and drop >= 0.5:
        main = "倾角型"
    elif abs(pitch) < 1 and abs(roll) < 1 and abs(o0) >= 0.05:
        main = "常数型"
    else:
        main = "说不清"
    sub = None
    if "scale" in out:
        (s, o), _, _ = out["scale"]
        sh = abs(s) * out["L_median"]
        sub = "尺度" if sh >= 0.05 and abs(o) < 0.03 else "整体平移" if abs(o) >= 0.05 and sh < 0.03 else "分不开"
    return main, sub


def control_ok(out):
    """登记的对照判据：地面上静置的集 |g| 中位 ≤ 15 mm；倾角模型俯仰、横滚都 ≤ 0.5°，|o| ≤ 15 mm。"""
    if "tilt" not in out:
        return False
    pitch, roll = tilt_deg(out)
    (o, _, _), _, _ = out["tilt"]
    return out["floor_abs_median"] <= 0.015 and abs(pitch) <= 0.5 and abs(roll) <= 0.5 and abs(o) <= 0.015


def offset_gate(out, gate_mm):
    """10/10 登记的判据：地面上的集至少 8 集，常数模型 |o| ≤ gate_mm。返回 None（不判读）、True（过）或 False（不过）。"""
    if out["n_floor"] < 8 or "const" not in out:
        return None
    (o0,), _, _ = out["const"]
    return bool(abs(o0) * 1000 <= gate_mm)   # numpy 的比较结果是 np.bool_，`is True` 判不出


def safe(root: Path, rel: str) -> Path:
    p = PurePosixPath(rel)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise SystemExit(f"t3_float_facts: 元数据里的相对路径不安全：{rel}")
    return root / p


def load_meta(root: Path) -> dict:
    jl = root / "meta" / "episodes_metadata.jsonl"
    rows = ([json.loads(x) for x in jl.read_text().splitlines() if x.strip()] if jl.is_file()
            else [json.loads(f.read_text()) for f in sorted((root / "meta").glob("episode_*.metadata.json"))])
    if not rows:
        raise SystemExit(f"t3_float_facts: {root}/meta 下没有元数据")
    return {int(r["episode_index"]): r for r in rows}


def main(argv=None) -> int:
    forbid_tier1()
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--tier2", type=Path, default=TIER2)
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--episodes", default="")
    ap.add_argument("--control", action="store_true")
    ap.add_argument("--all-rest", action="store_true")
    ap.add_argument("--offset-gate", type=float, default=None, metavar="MM")
    ap.add_argument("--support-root", type=Path, default=None)
    a = ap.parse_args(argv)
    meta = load_meta(a.root)
    eps = sorted(meta) if not a.episodes else [int(x) for x in a.episodes.split(",")]
    sroot = a.support_root if a.support_root is not None else a.tier2
    where = f"；支撑面 {a.support_root}" if a.support_root is not None else ""
    print(f"目录 {a.root}；官方 Tier 2 {a.tier2}{where}；窗口第 0–{a.frames - 1} 帧；静置判据位移 ≤ {STATIC_MM:.0f} mm；{len(eps)} 集")
    print("ep\t物体\t帧\t位移mm\tH_m\th_mm\t面高_m\t间隙mm\t间隙30mm\tZ_m\tX_m\tD_m\tL_m\tk\tparquet_sha256")
    rows = []
    for ep in eps:
        m = meta[ep]
        f = a.root / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
        t = pq.read_table(f)
        p = np.asarray(t.column("observation.object.pose").to_pylist(), dtype=np.float64)
        vis = (np.asarray(t.column("observation.object.visible").to_pylist(), dtype=bool).ravel()
               if "observation.object.visible" in t.column_names else np.ones(len(p), bool))
        plane = json.loads(safe(a.tier2, m["ground_plane"]).read_text())["plane"]
        tops = surface_tops(safe(sroot, m["support_surface"]).read_text()) if m.get("support_surface") else []
        r = episode_facts(p, vis, hull_vertices(str(safe(a.tier2, m["mesh"]))), plane, tops, a.frames)
        sha = hashlib.sha256(f.read_bytes()).hexdigest()[:12]
        if r is None:
            print(f"{ep}\t{m.get('object', '?')}\t0\t（窗口里没有可见帧）\t\t\t\t\t\t\t\t\t\t\t{sha}")
            continue
        r["ep"] = ep
        rows.append(r)
        k = r["L"] / (r["L"] + r["gap"])
        print(f"{ep}\t{m.get('object', '?')}\t{r['n_frames']}\t{r['disp_mm']:.1f}\t{r['H']:.3f}\t{r['h'] * 1000:+.1f}\t{r['top']:.3f}\t"
              f"{r['gap'] * 1000:+.1f}\t{r['gap_wide'] * 1000:+.1f}\t{r['Z']:.2f}\t{r['X']:+.2f}\t{r['D']:.2f}\t{r['L']:.3f}\t{k:.3f}\t{sha}")
    out = fit(rows, float("inf") if a.all_rest else STATIC_MM)
    if a.all_rest:
        print("不按位移筛静置，全部集都进拟合；只按 10/10 登记的 --offset-gate 判读" if a.offset_gate is not None
              else "事后（未登记）：不按位移筛静置，全部集都进拟合；不打判读")
    print(f"静置 {out['n_static']} 集：在地面上 {out['n_floor']}，在支撑面上 {out['n_support']}")
    if "tilt" in out:
        (o0,), (se0,), ss0 = out["const"]
        (o, b, e), (so, sb, se), ss1 = out["tilt"]
        pitch, roll = tilt_deg(out)
        print(f"地面上的集 |间隙| 中位 {out['floor_abs_median'] * 1000:.1f} mm")
        print(f"常数模型：o = {o0 * 1000:+.1f} ± {se0 * 1000:.1f} mm，残差平方和 {ss0 * 1e6:.0f} mm²")
        print(f"倾角模型：o = {o * 1000:+.1f} ± {so * 1000:.1f} mm，b = {b * 1000:+.1f} ± {sb * 1000:.1f} mm/m（俯仰约 {pitch:+.2f}°），"
              f"e = {e * 1000:+.1f} ± {se * 1000:.1f} mm/m（横滚约 {roll:+.2f}°），残差平方和 {ss1 * 1e6:.0f} mm²"
              f"（比常数模型少 {100 * (1 - ss1 / ss0) if ss0 > 0 else 0:.0f}%）")
    if "scale" in out:
        (s, o), (ss_, so), _ = out["scale"]
        print(f"尺度模型（全部静置集）：s = {s:+.4f} ± {ss_:.4f}（k = 1/(1 + s) = {1 / (1 + s):.4f}，|s|·L 中位 = {abs(s) * out['L_median'] * 1000:.1f} mm），"
              f"o = {o * 1000:+.1f} ± {so * 1000:.1f} mm")
    if a.offset_gate is not None:
        v = offset_gate(out, a.offset_gate)
        if v is None:
            print(f"OFFSET_GATE_NONE（地面上 {out['n_floor']} 集，不到 8 集，不判读）")
        else:
            (o0,), (se0,), _ = out["const"]
            print(f"{'OFFSET_GATE_PASS' if v else 'OFFSET_GATE_FAIL'}（地面上 {out['n_floor']} 集，常数模型 o = {o0 * 1000:+.1f} ± {se0 * 1000:.1f} mm，"
                  f"每集标准差 {out['floor_sd'] * 1000:.1f} mm，门槛 |o| ≤ {a.offset_gate:g} mm）")
    if a.all_rest:
        pass
    elif a.control:
        print("CONTROL_OK" if control_ok(out) else "CONTROL_FAIL")
    else:
        main_, sub = reading(out)
        print(f"判读：{main_}" + (f"；尺度模型：{sub}" if sub else ""))
    print("T3_FLOAT_FACTS_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
