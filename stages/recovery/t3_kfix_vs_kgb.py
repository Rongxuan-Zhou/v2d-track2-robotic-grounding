"""k-fix 的重新表达与另一位用户 convert_kgb（同一组标定内参）逐帧比（ml40 的 v2d_cari4d 镜像里跑，只用 CPU；只读两边的 focal 输出，
不读任何真值，不读 Tier 1、Tier 2）。两份实现同一个映射 M = (f_new/f_used)·K_new⁻¹K_used（深度随焦距）、朝向随视线转、人体骨盆按同一映射、
其余关节刚体跟随；来源是同一个 runs/convert（k-fix 判据 0 核过）。一致到浮点精度就互相印证。
逐集报：物体位置差（m）与朝向差（度）的中位数与最大值、两边写的内参是否相同；人体 transl 与 poses 的最大差。
--stage world：改比世界系的输出（object_world.npz 的 T_world_glb、soma_params_world.npz），用来核另一位用户 gb3 的世界系（地面校正、配准之后）
与 k-fix 的是否一致（主机位那几集两边的地面拟合相同：倾角 1.222°、尺度 0.984，results/t3kfix/floor_group_residual_ml40_1016.txt）。
用法（容器内）：python /k/t3_kfix_vs_kgb.py [--stage focal|world] <k-fix 的 convert 目录> <另一位用户的对应目录>；
最后一行 KFIX_VS_KGB_DONE <最大位置差> <最大朝向差>。
"""
import sys
from pathlib import Path

import numpy as np


def angles(Ra: np.ndarray, Rb: np.ndarray) -> np.ndarray:
    tr = np.einsum("nij,nij->n", Ra, Rb)
    return np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))


STAGES = {"focal": ("object_cam_focal.npz", "T_cam_glb", "soma_params_focal.npz"),
          "world": ("object_world.npz", "T_world_glb", "soma_params_world.npz")}


def main() -> int:
    args = sys.argv[1:]
    stage = "focal"
    if args[:1] == ["--stage"]:
        stage, args = args[1], args[2:]
    obj, key, hum_f = STAGES[stage]
    ours, theirs = Path(args[0]), Path(args[1])
    worst_t, worst_r, n = 0.0, 0.0, 0
    for d in sorted(ours.glob("ep*")):
        a, b = d / obj, theirs / d.name / obj
        if not (a.exists() and b.exists()):
            continue
        oa, ob = np.load(a), np.load(b)
        Ta, Tb = oa[key].astype(np.float64), ob[key].astype(np.float64)
        if Ta.shape != Tb.shape:
            print(f"{d.name}：帧数不同 {Ta.shape} {Tb.shape}")
            continue
        pos = np.linalg.norm(Ta[:, :3, 3] - Tb[:, :3, 3], axis=1)
        ang = angles(Ta[:, :3, :3], Tb[:, :3, :3])
        dt, dr = float(pos.max()), float(ang.max())
        dk = float(np.abs(np.asarray(oa["intrinsics"], float) - np.asarray(ob["intrinsics"], float)).max())
        sa, sb = d / hum_f, theirs / d.name / hum_f
        hum = "人体：缺文件"
        if sa.exists() and sb.exists():
            # 只取 transl、poses 两个数值数组，不需要 allow_pickle（npz 里另有字符串数组，不去读它）
            pa, pb = np.load(sa), np.load(sb)
            hum = "人体 " + "、".join(f"{k} 最大差 {float(np.abs(np.asarray(pa[k], float) - np.asarray(pb[k], float)).max()):.2e}"
                                      for k in ("transl", "poses") if k in pa.files and k in pb.files)
        print(f"{d.name}：{len(Ta)} 帧，物体位置差中位 {float(np.median(pos)):.2e}、最大 {dt:.2e} m，朝向差中位 {float(np.median(ang)):.2e}、"
              f"最大 {dr:.2e}°，内参差 {dk:.2e}；{hum}")
        worst_t, worst_r, n = max(worst_t, dt), max(worst_r, dr), n + 1
    print(f"{n} 集；物体位置最大差 {worst_t:.2e} m，朝向最大差 {worst_r:.2e}°")
    print(f"KFIX_VS_KGB_DONE {worst_t:.2e} {worst_r:.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
