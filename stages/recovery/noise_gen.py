"""按实测腐蚀模型生成 Tier 2 风格的物体轨迹噪声（仅用于分析：输入是 Tier 1 真值）。

截断随机游走：D_R 每帧左乘 exp(N(0, σ_r² I))，转角超过 30° 投影回 30°；
D_x 每帧加 N(0, σ_x² I)，超过 30 mm 投影回 30 mm；D(0)=I。
σ 取自真实 Tier 2 未饱和帧（overfit_audit P4）：σ_r = 0.754°/轴/帧，σ_x = 0.767 mm/轴/帧。
已知局限：撞上限比例 0.11 vs 真实 0.36，平均幅度 19.4°/17.7 mm vs 21.4°/18.7 mm。
"""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

import add_metrics as A

SIG_R, SIG_X = np.radians(0.754), 0.000767
CAP_R, CAP_X = np.radians(30), 0.030


def corrupt(p_true, seed):
    g = np.random.default_rng(seed); T = len(p_true)
    D = np.eye(3); x = np.zeros(3); DR = np.empty((T, 3, 3)); Dx = np.empty((T, 3))
    for t in range(T):
        if t:
            D = Rot.from_rotvec(g.normal(0, SIG_R, 3)).as_matrix() @ D
            rv = Rot.from_matrix(D).as_rotvec(); a = np.linalg.norm(rv)
            if a > CAP_R: D = Rot.from_rotvec(rv * CAP_R / a).as_matrix()
            x = x + g.normal(0, SIG_X, 3); nx = np.linalg.norm(x)
            if nx > CAP_X: x = x * CAP_X / nx
        DR[t] = D; Dx[t] = x
    q = p_true.copy()
    R = np.einsum("tij,tjk->tik", DR, A.quat_wxyz_to_matrix(p_true[:, 3:7]))
    q[:, 3:7] = Rot.from_matrix(R).as_quat()[:, [3, 0, 1, 2]]
    q[:, :3] = p_true[:, :3] + Dx
    return q
