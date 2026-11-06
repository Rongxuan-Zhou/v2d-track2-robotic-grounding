"""Tier 3 k-fix 方案 1：把 CARI4D 相机系的重建按标定内参重新表达（ml40 的 v2d_cari4d 镜像里跑，只用 CPU；只读 CARI4D 的转换结果，
不读任何轨迹真值，不读 Tier 1、Tier 2）。计划 agent-doc/plan-t3-kfix-2026-10-16.md 方案 1。

沿用另一位用户 `stages/recovery/tier3/t3_convert_ep.py`（分支 track2/tier3-recon，部署在 ml40）的焦距步，只把“深度乘 k、主点不动”推广成一般的
像素不变线性映射 M（K_c 是 CARI4D 的内参，K_t 是标定内参）：
  口径 F（深度随焦距成比例）：M = (f_t/fx_c)·K_t⁻¹ K_c（fx_c = fy_c 时即 [[1, 0, (cx_c − cx_t)/f_c], [0, 1, (cy_c − cy_t)/f_c], [0, 0, f_t/f_c]]）；
  口径 Z（深度不变）：M = K_t⁻¹ K_c。
任一点 X 换成 M X 后，在 K_t 下的像素与原来在 K_c 下的相同（`pixel_check` 逐帧核物体原点与人体骨盆，差 ≤ 1e-6 px）。物体：位置 M t，
朝向随视线转（她的 `_ray_turn(t, M t)`）；人体：骨盆到 M·骨盆，根随视线转，其余关节刚体跟随（她的 `_move_body`，逐帧核关节，误差上限 1 mm）。
随视线转是对剪切的一阶近似：中心精确，0.5 m 深的物体边缘约 (Δcx/f_c)·0.5 m ≈ 4 cm。

输出写成焦距步的文件名：object_cam_focal.npz（T_cam_glb、intrinsics = K_t）、soma_params_focal.npz、focal_calib.json，另抄 residual_report.json。
这样她的地面拟合（经影子根目录）、世界步与装配原样可用。focal_calib.json 带她的世界步要读的键（fx_used_by_cari4d、fx_new、depth_ratio、
focal_spec_sha256），另记 K_c、K_t、口径、M、输入文件的 sha256、她代码的部署 commit。输出目录已存在就拒绝。
用法（容器内）：python /k/t3_kfix.py <convert 目录> <输出目录> <kt.json> <SOMA 资产目录> --mode F|Z [--code-dir /c/stages/recovery/tier3]
kt.json：{"f": …, "cx": …, "cy": …, "source": "…"}。最后一行 KFIX_EXPRESS_OK <集名>。
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def kmat(f: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])


def mapping(kc: np.ndarray, kt: np.ndarray, mode: str) -> np.ndarray:
    """像素不变的线性映射 M = k·K_t⁻¹K_c（OpenCV 相机系，作用在列向量上）；口径 F：k = f_t / fx_c，口径 Z：k = 1。
    用一般形式而不是按 fx = fy 展开的显式式子：CARI4D 的内参按 float32 存，fx 与 fy 末位可能不同（ep03 上显式式子差 6.9e-6 px，被
    pixel_check 拦下）。"""
    if mode == "Z":
        k = 1.0
    elif mode == "F":
        k = float(kt[0, 0]) / float(kc[0, 0])
    else:
        raise ValueError(f"口径只能是 F 或 Z，收到 {mode!r}")
    return k * np.linalg.inv(kt) @ kc


def project(k: np.ndarray, X: np.ndarray) -> np.ndarray:
    p = X @ k.T
    return p[:, :2] / p[:, 2:3]


def pixel_check(kc: np.ndarray, kt: np.ndarray, X: np.ndarray, M: np.ndarray, tol: float = 1e-6) -> float:
    """X（n,3）在 K_c 下的像素与 M X 在 K_t 下的像素之差的最大值；超过 tol 就抛错。"""
    d = float(np.abs(project(kc, X) - project(kt, X @ M.T)).max())
    if d > tol:
        raise SystemExit(f"像素不变核对不过：最大差 {d:.3g} px")
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("conv", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("kt", type=Path)
    ap.add_argument("assets", type=Path)
    ap.add_argument("--mode", choices=("F", "Z"), required=True)
    ap.add_argument("--code-dir", type=Path, default=Path("/c/stages/recovery/tier3"),
                    help="另一位用户 stages/recovery/tier3 的部署副本（她的 code 目录只读挂载到 /c，DEPLOYED 在 /c 下）")
    a = ap.parse_args()
    sys.path.insert(0, str(a.code_dir))
    import t3_convert_ep as ce   # 她的 _ray_turn、_pelvis_cv、_move_body

    name = a.conv.name
    if a.out.exists():
        raise SystemExit(f"{a.out} 已存在")
    spec_bytes = a.kt.read_bytes()
    spec = json.loads(spec_bytes)
    oc = dict(np.load(a.conv / "object_cam.npz"))
    kc = np.asarray(oc["intrinsics"], np.float64)
    kt = kmat(float(spec["f"]), float(spec["cx"]), float(spec["cy"]))
    M = mapping(kc, kt, a.mode)

    t = oc["T_cam_glb"].astype(np.float64)
    tn = t.copy()
    tn[:, :3, 3] = t[:, :3, 3] @ M.T
    tn[:, :3, :3] = np.einsum("nij,njk->nik", ce._ray_turn(t[:, :3, 3], tn[:, :3, 3]), t[:, :3, :3])
    d_obj = pixel_check(kc, kt, t[:, :3, 3], M)

    # soma_params.npz 里有对象数组（identity_model_type 等字符串），必须 allow_pickle；文件是另一位用户在本实验室 ml40 上的 CARI4D 转换输出、
    # 只读挂载，她的 t3_convert_ep.py 也这样读同一个文件，不是外部来源
    sp = dict(np.load(a.conv / "soma_params.npz", allow_pickle=True))
    pelvis0 = ce._pelvis_cv(sp, a.assets)
    d_pel = pixel_check(kc, kt, pelvis0, M)
    rot = ce._ray_turn(pelvis0, pelvis0 @ M.T)
    new, err = ce._move_body(sp, rot, lambda p: p @ M.T, a.assets, name, "kfix")

    a.out.mkdir(parents=True)
    np.savez(a.out / "object_cam_focal.npz", T_cam_glb=tn, intrinsics=kt)
    np.savez(a.out / "soma_params_focal.npz", **new)
    shutil.copy2(a.conv / "residual_report.json", a.out / "residual_report.json")
    code_commit = (a.code_dir.parent.parent.parent / "DEPLOYED")
    calib = {"episode": name, "step": "kfix", "mode": a.mode, "focal_spec": a.kt.name,
             "focal_spec_sha256": hashlib.sha256(spec_bytes).hexdigest(), "kt_source": spec.get("source"),
             "fx_used_by_cari4d": float(kc[0, 0]), "fx_new": float(kt[0, 0]), "depth_ratio": float(M[2, 2]),
             "K_c": kc.tolist(), "K_t": kt.tolist(), "M": M.tolist(),
             "pixel_check_max_px": {"object": d_obj, "pelvis": d_pel}, "body_rigid_max_err_mm": round(err * 1000, 4),
             "inputs_sha256": {f: sha256(a.conv / f) for f in ("object_cam.npz", "soma_params.npz", "residual_report.json")},
             "code_deployed": code_commit.read_text().strip() if code_commit.is_file() else None}
    (a.out / "focal_calib.json").write_text(json.dumps(calib, indent=1))
    print(json.dumps({k: calib[k] for k in ("episode", "mode", "fx_new", "depth_ratio", "pixel_check_max_px", "body_rigid_max_err_mm")}))
    print(f"KFIX_EXPRESS_OK {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
