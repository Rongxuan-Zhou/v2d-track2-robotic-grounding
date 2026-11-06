"""给一条 motion parquet 算参考回放视频的相机位置（FlashCHORD 容器内跑，要 pyarrow 与 numpy）。

SOP 页面的参考动画由 scripts/debug/replay.py 的 mp4 viewer 渲染，默认相机离场景很远，机器人常在画外。
这里取机器人根节点与物体根节点整条轨迹的水平包围盒：相机看向包围盒中心（高 0.8 m），从包围盒最长轴的垂直方向、
按包围盒大小后退，抬高 1.8 m，保证走过去的路线和操作的物体都在画面里。按 Hydra 覆盖参数的写法打印一行，
另打印帧数（参考的源帧数，逐帧回放时一帧一个控制步）。
用法（容器内）：python /repo/stages/policy/site_ref_camera.py <data.parquet>
"""
import sys

import numpy as np
import pyarrow.parquet as pq


def main() -> None:
    t = pq.read_table(sys.argv[1], columns=["robot_root_position", "object_root_position"]).to_pylist()[0]
    root = np.asarray(t["robot_root_position"], dtype=float)
    obj = np.asarray(t["object_root_position"], dtype=float)
    pts = np.concatenate([root[:, :2], obj[:, :2]])
    lo, hi = pts.min(0), pts.max(0)
    center = (lo + hi) / 2
    extent = hi - lo
    # 看的方向垂直于包围盒的长边，这样沿长边走的动作是横着过画面
    long_axis = np.array([1.0, 0.0]) if extent[0] >= extent[1] else np.array([0.0, 1.0])
    view = np.array([-long_axis[1], long_axis[0]])
    dist = 1.2 * max(float(np.linalg.norm(extent)), 1.5) + 1.5
    target = (float(center[0]), float(center[1]), 0.8)
    pos = (float(center[0] + dist * view[0]), float(center[1] + dist * view[1]), 1.8)
    fmt = lambda v: "[" + ",".join(f"{x:.3f}" for x in v) + "]"
    print(f"viewer.camera_position={fmt(pos)} viewer.camera_target={fmt(target)} viewer.num_frames={len(root)}")


if __name__ == "__main__":
    main()
