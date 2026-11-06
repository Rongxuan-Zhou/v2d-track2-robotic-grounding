"""用 Newton 的 viser / rerun 查看器看 FlashCHORD policy 的 rollout（包装官方 view_policy.py；在 flash_chord 容器里、/workspace 下运行）。

官方的查看器工厂 flash_chord.visualization.viewer.create_viewer 只会构造默认端口的 ViewerViser()，也没有 rerun 后端。
这里替换这个工厂，再用 runpy 原样运行 /workspace/scripts/view_policy.py（官方源码不改）：
  viewer.backend=viser  → ViewerViser(port=$V2D_VISER_PORT，默认 8080；label=$V2D_VIEW_LABEL)。交互式，一直运行到被停掉。
  viewer.backend=rerun  → ViewerRerun(app_id=$V2D_VIEW_LABEL, record_to_rrd=$V2D_RRD_PATH)，只写 .rrd 文件：
                          ViewerRerun 在不连地址、不开网页时会 rr.spawn() 本地窗口，服务器上没有显示器，这里把 spawn 换成空操作；
                          录满 viewer.num_frames 帧就退出。rerun-sdk 不在镜像里，用 `uv pip install --no-deps --target <目录> rerun-sdk==0.27.3` 另装（依赖用镜像自带的；ml40 走腾讯云 PyPI 镜像），
                          PYTHONPATH 要追加镜像原有的 /workspace/src，不要覆盖。Mac 上用同版本 `uv tool install rerun-sdk==0.27.3` 看 .rrd。
用法（容器内）：
  V2D_VISER_PORT=8081 V2D_VIEW_LABEL=ep27 python /repo/<本文件> evaluation.checkpoint=<policy> viewer.backend=viser logging.mode=disabled
  V2D_RRD_PATH=/repo/runs/viewer/rrd/ep27.rrd PYTHONPATH=<rerun 目录>/rerun_sdk:<rerun 目录>:/workspace/src python /repo/<本文件> evaluation.checkpoint=<policy> \
      viewer.backend=rerun viewer.num_frames=<帧数> logging.mode=disabled
"""
import os
import runpy
import sys

import flash_chord.visualization.viewer as fv

VIEW_POLICY = "/workspace/scripts/view_policy.py"
_original_create_viewer = fv.create_viewer


class BoundedViewer:
    """录够 num_frames 帧就让 app 停下（同官方 MP4Viewer 的做法）。"""

    def __init__(self, viewer, num_frames: int, label: str):
        self._viewer = viewer
        self.num_frames = num_frames
        self.frame_count = 0
        self.label = label

    def __getattr__(self, name: str):
        return getattr(self._viewer, name)

    def is_running(self) -> bool:
        return self.frame_count < self.num_frames and self._viewer.is_running()

    def end_frame(self) -> None:
        self._viewer.end_frame()
        self.frame_count += 1

    def close(self) -> None:
        self._viewer.close()
        print(f"rerun recording: {self.label} ({self.frame_count} frames)", flush=True)


def create_viewer(config, device=None):
    import newton.viewer
    import warp as wp

    if config.backend not in {"viser", "rerun"}:
        return _original_create_viewer(config, device=device)
    wp.config.quiet = config.quiet
    if device is not None:
        wp.set_device(device)
    label = os.environ.get("V2D_VIEW_LABEL", "v2d")
    if config.backend == "viser":
        port = int(os.environ.get("V2D_VISER_PORT", "8080"))
        print(f"viser: port {port}, label {label}", flush=True)
        return newton.viewer.ViewerViser(port=port, label=label)
    import rerun as rr

    path = os.environ["V2D_RRD_PATH"]
    rr.spawn = lambda *args, **kwargs: None  # 服务器上不开本地窗口；数据只进 rr.save 设好的文件
    viewer = newton.viewer.ViewerRerun(app_id=label, serve_web_viewer=False, record_to_rrd=path)
    print(f"rerun: rerun-sdk {rr.__version__}, recording to {path}", flush=True)
    return BoundedViewer(viewer, config.num_frames, path)


fv.create_viewer = create_viewer
sys.argv = [VIEW_POLICY, *sys.argv[1:]]
runpy.run_path(VIEW_POLICY, run_name="__main__")
