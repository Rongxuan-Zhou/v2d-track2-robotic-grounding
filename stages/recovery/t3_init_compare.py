"""Tier 3 初始化试点的核对 A 与 C：work_fp 与 work_init 两版 CARI4D 原始输出的物体位姿逐帧比（ml40 的 v2d_cari4d 镜像里跑，只用 CPU；
只读另一位用户的 CARI4D 输出，不读任何真值，不读 Tier 1、Tier 2）。判读规则见 agent-doc/experiments/2026-10-16-t3-init-pilot-check.md。

物体位姿取法同另一位用户 t3_convert_ep.py 的 decode：T_cam<-glb = refined.pth 的 pr["pose_abs"] @ wild_export.json 的
source_object_mesh_to_aligned_transform。被替换的帧取 work_init 里 foundationpose/fp_init.json（t3_fp_init.py 写的）。
逐集报：两版各自逐帧转角大于 45°、90° 的帧数（被替换帧含前后各 1 帧 / 其余帧 / 全部），两版逐帧位置差与朝向差的中位、90% 分位（被替换帧 / 其余帧）。
用法（容器内）：python /k/t3_init_compare.py <tier3 根> <集名>... > 输出；最后一行 T3_INIT_COMPARE_DONE。
"""
import json
import sys
from pathlib import Path

import numpy as np


def poses(c: Path) -> np.ndarray:
    import torch
    # weights_only=False：refined.pth 里不只有张量（同 t3_convert_ep.py 的读法）；文件是另一位用户在同一台服务器上跑出的 CARI4D 输出，
    # 容器只读挂载、不联网
    bundle = torch.load(c / "inference" / "refined.pth", map_location="cpu", weights_only=False)
    (export_dir,) = [d for d in (c / "export").iterdir() if d.is_dir()]
    s2a = np.asarray(json.loads((export_dir / "wild_export.json").read_text())["source_object_mesh_to_aligned_transform"], np.float64)
    return np.asarray(bundle["pr"]["pose_abs"], np.float64) @ s2a


def angle(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    """逐帧 Ra 与 Rb 之间的转角（度）。"""
    tr = np.einsum("nij,nij->n", Ra, Rb)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


def replaced_frames(fp_init: Path, n: int) -> np.ndarray:
    """fp_init.json 的 replaced_runs 是左闭右开的 [起, 止)；段长之和要等于 replaced_frames，帧数要等于 frames。"""
    d = json.loads(fp_init.read_text())
    if int(d["frames"]) != n:
        raise SystemExit(f"{fp_init}：frames {d['frames']} 与位姿帧数 {n} 不同")
    mask = np.zeros(n, bool)
    for a, b in d["replaced_runs"]:
        mask[int(a):int(b)] = True
    if int(mask.sum()) != int(d["replaced_frames"]):
        raise SystemExit(f"{fp_init}：replaced_runs 合起来 {int(mask.sum())} 帧，replaced_frames 写的是 {d['replaced_frames']}")
    return mask


def main() -> int:
    root = Path(sys.argv[1])
    for name in sys.argv[2:]:
        seq = f"episode_{int(name[2:4]):06d}"
        cf, ci = root / "runs/work_fp" / name / "cari4d" / seq, root / "runs/work_init" / name / "cari4d" / seq
        Pf, Pi = poses(cf), poses(ci)
        if Pf.shape != Pi.shape:
            raise SystemExit(f"{name}：两版帧数不同 {Pf.shape} {Pi.shape}")
        n = len(Pf)
        rep = replaced_frames(ci / "foundationpose" / "fp_init.json", n)
        near = rep | np.r_[rep[1:], False] | np.r_[False, rep[:-1]]
        out = {"episode": name, "frames": n, "replaced": int(rep.sum())}
        for tag, P in (("fp", Pf), ("init", Pi)):
            step = np.r_[0.0, angle(P[1:, :3, :3], P[:-1, :3, :3])]
            for thr in (45, 90):
                out[f"{tag}_jump{thr}_replaced"] = int(((step > thr) & near).sum())
                out[f"{tag}_jump{thr}_other"] = int(((step > thr) & ~near).sum())
                out[f"{tag}_jump{thr}_all"] = int((step > thr).sum())
        dR = angle(Pf[:, :3, :3], Pi[:, :3, :3])
        dt = np.linalg.norm(Pf[:, :3, 3] - Pi[:, :3, 3], axis=1)
        for tag, m in (("replaced", rep), ("other", ~rep)):
            if m.any():
                out[f"rot_diff_deg_{tag}_p50_p90"] = [round(float(np.median(dR[m])), 2), round(float(np.percentile(dR[m], 90)), 2)]
                out[f"pos_diff_cm_{tag}_p50_p90"] = [round(100 * float(np.median(dt[m])), 2), round(100 * float(np.percentile(dt[m], 90)), 2)]
        out["pos_diff_cm_all_p50"] = round(100 * float(np.median(dt)), 2)
        print(json.dumps(out, ensure_ascii=False), flush=True)
    print("T3_INIT_COMPARE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
