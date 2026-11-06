"""从 diag_rollout.py 存下的 npz 算“物体全程保持第 0 帧参考位姿”的 ADD-AUC（Day 2 的不动基线）。在 FlashCHORD 容器里跑，只用 CPU。

add_static 是 diag_rollout 用评测器自己的顶点（rollout 的 object_vertices_o）与参考位姿算的逐帧 ADD，与 policy 无关，
所以冒烟 policy 的 rollout 也能给出这条序列在这种回放下的基线。ADD-AUC 的算法同评测器（门槛 1–9 cm，9 个点的梯形积分）。
另打出帧数、policy 路径与它内嵌配置里的参考 parquet，供比较脚本核对（四个格子与两份基线必须出自同一个 parquet）。
用法：python /repo/stages/evaluate/static_bar_from_rollout.py <rollout.npz>
"""
import json
import struct
import sys

import numpy as np

AUC_THRESHOLDS = np.arange(0.01, 0.10, 0.01)  # 同评测器 metrics.ADD_THRESHOLDS_M


def auc(samples: np.ndarray) -> float:
    x = np.asarray(samples, np.float64).ravel()
    return float(np.trapezoid([np.mean(x < t) for t in AUC_THRESHOLDS], x=np.linspace(0, 1, AUC_THRESHOLDS.size)))


def main() -> int:
    d = np.load(sys.argv[1], allow_pickle=False)
    add_static = d["add_static"].astype(np.float64)
    bar = auc(add_static)
    frames = int(add_static.shape[0])
    with open(str(d["policy"]), "rb") as f:   # 参考 parquet 取自 policy 内嵌的配置（同 diag_rollout.py）
        n = struct.unpack("<Q", f.read(8))[0]
        cfg = json.loads(json.loads(f.read(n))["__metadata__"]["user.resolved_config_json"])
    parquet = str(cfg["task"]["parquet"]).strip("'")
    print(f"rollout {sys.argv[1]}：{frames} 帧；policy {d['policy']}")
    print(f"  不动基线 ADD-AUC {bar:.4f}（评测器顶点；trimesh 与评测器顶点一致：{bool(d['trimesh_matches_evaluator'])}）")
    print("  summary_json " + json.dumps({"add_auc": bar, "frames": frames, "policy": str(d["policy"]), "parquet": parquet,
                                          "trimesh_matches_evaluator": bool(d["trimesh_matches_evaluator"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
