"""k-fix 判据 4b：新表达的相机系里，人脚有没有物体事件拟出的那约 1.2° 横滚（ml40 的 v2d_cari4d 镜像里跑，只用 CPU；只读影子根、Tier 3 input
与另一位用户的代码，不读真值，不读 Tier 1、Tier 2 轨迹）。登记与判读规则见 agent-doc/plan-t3-kfix-2026-10-16.md 判据 4b。

逐集（floor_fit_kt.json 里第 0 组、convert 目录里有 soma_params_focal.npz 的集）：SOMA 正向得到相机系顶点（另一位用户 t3_world_probe.soma_cv），
逐帧取两只脚鞋底顶点（蒙皮权重落在 Foot、ToeBase、ToeEnd 上 ≥ 0.5，同 t3_feet_diag.py）里对发布地面最低的那个，记高度 h 与它在发布地面上的
横向坐标（相机 x 轴投到地面上）、纵深坐标（地面上与横向垂直）。每集一个截距，合起来用 Huber（2 cm）迭代重加权拟两个斜率；按集自助抽样给 95% 区间。
对照量：n_ours 隐含的斜率 s = −(n_ours·e)/(n_ours·n_released)（e 为横向或纵深单位向量；推导见计划）。
用法（容器内）：python /k/t3_kfix_feet_roll.py <影子根> <floor_fit_kt.json> <SOMA 资产> <输出 .json> [--code-dir /c/stages/recovery/tier3]
     自检（Mac 上，只要 numpy）：python stages/recovery/t3_kfix_feet_roll.py --selftest
最后一行 FEET_ROLL_DONE <判读> 或 FEET_ROLL_SELFTEST_OK。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HUBER_M = 0.02
BOOT = 2000


def plane(n, d) -> tuple:
    """单位法向、d > 0（相机在正侧），同 t3_feet_diag.py。"""
    n, d = np.asarray(n, np.float64), float(d)
    k = np.linalg.norm(n)
    n, d = n / k, d / k
    return (n, d) if d > 0 else (-n, -d)


def floor_axes(n: np.ndarray) -> tuple:
    """发布地面上的横向（相机 x 轴的投影）与纵深（地面内与横向垂直、朝光轴方向）单位向量。"""
    ex, ez = np.array([1.0, 0, 0]), np.array([0, 0, 1.0])
    lat = ex - (ex @ n) * n
    lat /= np.linalg.norm(lat)
    dep = ez - (ez @ n) * n - (ez @ lat) * lat
    return lat, dep / np.linalg.norm(dep)


def implied_slope(n_o: np.ndarray, n_r: np.ndarray, e: np.ndarray) -> float:
    """落在拟合地面 (n_o) 上的点，对发布地面 (n_r) 的高度沿 e 的变化率（m/m）。"""
    return float(-(n_o @ e) / (n_o @ n_r))


def huber_fit(h: np.ndarray, x: np.ndarray, ep: np.ndarray, delta: float = HUBER_M, iters: int = 30) -> np.ndarray:
    """h = a_集 + x·b，迭代重加权的 Huber 回归；返回 b（x 的列数个斜率）。"""
    eps, inv = np.unique(ep, return_inverse=True)
    A = np.c_[np.eye(len(eps))[inv], x]
    w, prev = np.ones(len(h)), None
    for _ in range(iters):
        sw = np.sqrt(w)
        coef = np.linalg.lstsq(A * sw[:, None], h * sw, rcond=None)[0]
        if prev is not None and np.abs(coef - prev).max() < 1e-7:
            break
        prev = coef
        r = np.abs(h - A @ coef)
        w = np.where(r <= delta, 1.0, delta / np.maximum(r, 1e-12))
    return coef[len(eps):]


def boot_ci(h, x, ep, rng, n=BOOT) -> np.ndarray:
    """按集有放回抽样（同一集抽到几次就重复几次，每次给新的集号），返回 (n, 斜率数)。"""
    eps = np.unique(ep)
    idx = {e: np.nonzero(ep == e)[0] for e in eps}
    out = []
    for _ in range(n):
        pick = rng.choice(eps, len(eps), replace=True)
        rows = np.concatenate([idx[e] for e in pick])
        lab = np.concatenate([np.full(len(idx[e]), k) for k, e in enumerate(pick)])
        out.append(huber_fit(h[rows], x[rows], lab))
    return np.array(out)


def selftest() -> int:
    rng = np.random.default_rng(0)
    # 1. 隐含斜率：把发布地面法向绕纵深轴转 1.2°，横向斜率应为 ±tan(1.2°)，纵深斜率应为 0
    n_r, _ = plane([0.0, -0.999, 0.04], 1.39)
    lat, dep = floor_axes(n_r)
    a = np.radians(1.2)
    k = dep
    rot = np.eye(3) + np.sin(a) * np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]]) \
        + (1 - np.cos(a)) * np.outer(k, k) - (1 - np.cos(a)) * np.eye(3)
    n_o = rot @ n_r
    sl, sd = implied_slope(n_o, n_r, lat), implied_slope(n_o, n_r, dep)
    assert abs(abs(sl) - np.tan(a)) < 1e-9 and abs(sd) < 1e-9, (sl, sd)
    assert abs(lat @ n_r) < 1e-12 and abs(dep @ n_r) < 1e-12 and abs(lat @ dep) < 1e-12
    # 2. 稳健回归：24 集各自截距，真斜率 横向 +0.021、纵深 −0.006，1 cm 噪声加 10% 离群，应拟回来
    ep = np.repeat(np.arange(24), 500)
    x = np.c_[rng.uniform(-1.5, 1.5, len(ep)), rng.uniform(1.5, 5.0, len(ep))]
    h = rng.normal(0, 0.03, 24)[ep] + x @ np.array([0.021, -0.006]) + rng.normal(0, 0.01, len(ep))
    out = rng.random(len(ep)) < 0.1
    h[out] += rng.uniform(0.05, 0.3, out.sum())
    b = huber_fit(h, x, ep)
    assert abs(b[0] - 0.021) < 0.002 and abs(b[1] + 0.006) < 0.002, b
    ci = np.percentile(boot_ci(h, x, ep, rng, n=200), [2.5, 97.5], axis=0)
    assert ci[0, 0] < 0.021 < ci[1, 0], ci
    # 3. 反向对照：没有斜率时区间应含 0
    h0 = rng.normal(0, 0.03, 24)[ep] + rng.normal(0, 0.01, len(ep))
    ci0 = np.percentile(boot_ci(h0, x, ep, rng, n=200), [2.5, 97.5], axis=0)
    assert ci0[0, 0] < 0 < ci0[1, 0], ci0
    print(f"  隐含斜率：绕纵深轴转 1.2° → 横向 {100 * sl:+.3f} cm/m（tan 1.2° = {100 * np.tan(a):.3f}），纵深 {100 * sd:+.1e}")
    print(f"  稳健回归：拟出 横向 {100 * b[0]:+.2f}、纵深 {100 * b[1]:+.2f} cm/m（真值 +2.10、−0.60）；横向 95% 区间 [{100 * ci[0, 0]:+.2f}, {100 * ci[1, 0]:+.2f}]")
    print(f"  反向对照：无斜率时横向 95% 区间 [{100 * ci0[0, 0]:+.2f}, {100 * ci0[1, 0]:+.2f}] 含 0")
    print("FEET_ROLL_SELFTEST_OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path, nargs="?")
    ap.add_argument("fit", type=Path, nargs="?")
    ap.add_argument("assets", nargs="?")
    ap.add_argument("out", type=Path, nargs="?")
    ap.add_argument("--code-dir", type=Path, default=Path("/c/stages/recovery/tier3"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    sys.path.insert(0, str(a.code_dir))
    import t3_inputs
    from soma import SOMALayer
    from soma.units import Unit
    from t3_world_probe import soma_cv

    fit = json.loads(a.fit.read_text())
    g0 = next(g for g in fit["groups"] if g["group"] == 0)
    metas = {f"ep{m['episode_index']:02d}_{m['object']}": m
             for m in map(json.loads, (a.root / "input/meta/episodes_metadata.jsonl").read_text().splitlines()) if m}
    layer = SOMALayer(data_root=a.assets, identity_model_type="mhr", device="cpu", output_unit=Unit.METERS,
                      enable_procedural_transforms=False)
    names = [str(x) for x in layer._public_joint_names]
    w = np.asarray(layer.skinning_weights, np.float64)
    sole = np.concatenate([np.nonzero(w[:, [names.index(f"{s}Foot"), names.index(f"{s}ToeBase"), names.index(f"{s}ToeEnd")]].sum(1) >= 0.5)[0]
                           for s in ("Left", "Right")])
    keys = ("poses", "transl", "identity_coeffs", "scale_params", "bone_length_flexibles")
    n_o, d_o = plane(g0["n_ours"], g0["d_ours_m"])
    H, X, E, rows = [], [], [], []
    for conv in sorted((a.root / "runs/convert").glob("ep*")):
        name = conv.name
        entry = fit["episodes"].get(name)
        if entry is None or entry["group"] != 0 or not (conv / "soma_params_focal.npz").exists():
            continue
        gp = np.asarray(t3_inputs.ground_plane(a.root, metas[name]), np.float64)
        n_r, d_r = plane(gp[:3], gp[3])
        lat, dep = floor_axes(n_r)
        # 只取 keys 里的数值数组，不需要 allow_pickle（npz 里另有字符串数组，不去读它）
        z = np.load(conv / "soma_params_focal.npz")
        sp = {k: z[k] for k in keys}
        _, verts = soma_cv({k: sp[k] for k in keys}, layer)
        v = verts[:, sole]                                   # (T, 鞋底顶点, 3)，相机系
        hr = v @ n_r + d_r
        i = hr.argmin(1)
        low = v[np.arange(len(v)), i]                        # 每帧对发布地面最低的鞋底顶点
        h = hr[np.arange(len(v)), i]
        ho = (low @ n_o + d_o)
        H.append(h)
        X.append(np.c_[low @ lat, low @ dep])
        E.append(np.full(len(h), len(rows)))
        rows.append({"episode": name, "frames": len(h), "feet_released_m": round(float(np.median(h)), 4),
                     "feet_ours_m": round(float(np.median(ho)), 4), "lat_range_m": round(float(np.ptp(low @ lat)), 2)})
        print(json.dumps(rows[-1]), flush=True)
    h, x, ep = np.concatenate(H), np.concatenate(X), np.concatenate(E)
    np.savez(a.out.with_suffix(".frames.npz"), h=h, x=x, ep=ep, episodes=np.array([r["episode"] for r in rows]))   # 自助抽样可在别处重做
    n_r0, _ = plane(g0["n_released"], g0["d_released_m"])
    lat0, dep0 = floor_axes(n_r0)
    s_lat, s_dep = implied_slope(n_o, n_r0, lat0), implied_slope(n_o, n_r0, dep0)
    b = huber_fit(h, x, ep)
    bs = boot_ci(h, x, ep, np.random.default_rng(0))
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    has0, hass = lo[0] <= 0 <= hi[0], lo[0] <= s_lat <= hi[0]
    verdict = ("人脚也有同样的横滚" if (not has0 and hass) else "人脚没有这个横滚" if (has0 and not hass) else "说不清")
    res = {"episodes": rows, "slope_lat": b[0], "slope_dep": b[1], "ci_lat": [lo[0], hi[0]], "ci_dep": [lo[1], hi[1]],
           "implied_lat": s_lat, "implied_dep": s_dep, "verdict": verdict}
    a.out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"{len(rows)} 集 {len(h)} 帧。人脚高度斜率：横向 {100 * b[0]:+.2f} cm/m（95% 区间 [{100 * lo[0]:+.2f}, {100 * hi[0]:+.2f}]），"
          f"纵深 {100 * b[1]:+.2f} cm/m（[{100 * lo[1]:+.2f}, {100 * hi[1]:+.2f}]）")
    print(f"物体事件地面隐含的斜率：横向 {100 * s_lat:+.2f} cm/m，纵深 {100 * s_dep:+.2f} cm/m")
    med = lambda k: 100 * float(np.median([abs(r[k]) for r in rows]))
    print(f"每集人脚高度中位的绝对值，再取中位：对发布地面（不校正）{med('feet_released_m'):.2f} cm，对 n_ours、d_ours {med('feet_ours_m'):.2f} cm（只报不判）")
    print(f"判读（横向）：区间{'含' if has0 else '不含'} 0，{'含' if hass else '不含'} s_横向 → {verdict}")
    print(f"FEET_ROLL_DONE {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
