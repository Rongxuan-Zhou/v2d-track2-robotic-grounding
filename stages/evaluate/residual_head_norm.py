"""checkpoint 里输出头（最后一维等于动作维数的张量）有多大（训练机上经 scripts/dev/in-flash-chord 在容器里跑，只用 CPU；只读，只报不判）。
计划 agent-doc/plan-t12replay-2026-10-20.md 第 2 版 R0b（复核 MINOR 4）：zero_dense 残差头初始为 0，第 409600 步的策略“基本就是参考回放”原是推断，
这里直接量：每个 checkpoint 里形状最后一维为 --act（默认 43：29 个身体关节 + 14 个手指）的张量，打印名字、形状、最大绝对值与均方根。
张量名在 checkpoint 里是匿名的（state.NNNN），不能按顺序认：10/20 量到第 409600 步里 state.0015、state.0018、state.0019 全为 0，就是零初始化的残差均值头
（zero_dense），state.0016、state.0017 是标准差头（`results/t12replay/residual_head_mf4090.txt`）。
用法（容器内）：python /repo/stages/evaluate/residual_head_norm.py [--act 43] <policy.safetensors>...   最后一行 RESIDUAL_HEAD_DONE
"""
import argparse
import sys

import numpy as np
from safetensors import safe_open


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="+")
    ap.add_argument("--act", type=int, default=43)
    a = ap.parse_args()
    for p in a.ckpt:
        print(f"== {p}")
        with safe_open(p, "numpy") as f:
            for k in f.keys():  # noqa: SIM118 -- safe_open 不是 dict
                v = f.get_tensor(k)
                if v.ndim >= 1 and v.shape[-1] == a.act:
                    print(f"  {k}\t{tuple(v.shape)}\tmax|.| {np.abs(v).max():.3e}\trms {np.sqrt((v.astype(np.float64) ** 2).mean()):.3e}")
    print("RESIDUAL_HEAD_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
