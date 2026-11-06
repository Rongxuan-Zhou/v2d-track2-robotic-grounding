"""SOP“任务一览”的 G1 参考动画：把参考第 f 帧直接放进场景（不跑物理），无头渲染成 MP4（FlashCHORD 容器内跑）。

上游 scripts/debug/replay.py 的 Replay 自带 "Show kinematic reference" 模式（step() 里 runner.reset(f)），但只在 Viser 的按钮里能选。
这里继承 Replay、把播放模式固定为 "reference"，不改上游源码。计划 agent-doc/plan-site-media-2026-09-30.md（第 2 版，按 review M1）。
相机每帧跟随：看向机器人根节点与物体根节点的中点（平滑），从机器人在物体附近时的平均朝向那一侧看过去，距离随两者间距伸缩。
每帧记录第 0 个刚体（机器人根）的位姿是否有限与高度、机器人与物体的水平距离，结束时写 <输出>.json（判据 M3）。
用法（容器内，/workspace 为 FlashCHORD）：
  PYGLET_HEADLESS=1 python /repo/stages/policy/site_ref_replay.py --config-name=sonic_replay "task.parquet='<data.parquet>'" \
      viewer.backend=mp4 viewer.output_path=<out.mp4> viewer.num_frames=<T> viewer.video_fps=30 viewer.realtime=false \
      viewer.width=640 viewer.height=480 viewer.headless=true start_paused=false markers.enabled=false [scene.support=false]
"""
import importlib.util
import json
import math
from pathlib import Path

import hydra
import numpy as np
import pyarrow.parquet as pq
from omegaconf import DictConfig

from flash_chord.configuration import instantiate_typed
from flash_chord.visualization.viewer import ViewerConfig

_spec = importlib.util.spec_from_file_location("fc_debug_replay", "/workspace/scripts/debug/replay.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def _quat_forward_xy(wxyz: np.ndarray) -> np.ndarray:
    """根节点朝向的水平前向（机体 x 轴投到地面）。"""
    w, x, y, z = wxyz.T
    fx = 1 - 2 * (y * y + z * z)
    fy = 2 * (x * y + w * z)
    v = np.stack([fx, fy], -1)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)


def _camera_track(parquet: str, n: int):
    t = pq.read_table(parquet, columns=["robot_root_position", "robot_root_wxyz", "object_root_position"]).to_pylist()[0]
    root = np.asarray(t["robot_root_position"], dtype=float)[:n]
    quat = np.asarray(t["robot_root_wxyz"], dtype=float)[:n]
    obj = np.asarray(t["object_root_position"], dtype=float)[:n]
    sep = np.linalg.norm(root[:, :2] - obj[:, :2], axis=-1)
    near = sep < 1.5
    fwd = _quat_forward_xy(quat)
    side = fwd[near].mean(0) if near.any() else fwd.mean(0)
    if not np.isfinite(side).all() or np.linalg.norm(side) < 1e-3:
        side = np.array([1.0, 0.0])
    side = side / np.linalg.norm(side)
    # 从机器人前方略偏一侧看过去，避免正对时机器人挡住物体
    ang = math.radians(35.0)
    view = np.array([side[0] * math.cos(ang) - side[1] * math.sin(ang), side[0] * math.sin(ang) + side[1] * math.cos(ang)])
    mid = 0.5 * (root[:, :2] + obj[:, :2])
    dist = np.clip(1.1 * sep + 2.4, 2.8, 6.5)
    tgt, d = mid[0].copy(), float(dist[0])
    track = []
    for i in range(n):  # 指数平滑，避免镜头抖
        tgt = 0.94 * tgt + 0.06 * mid[i]
        d = 0.97 * d + 0.03 * float(dist[i])
        pos = (float(tgt[0] + d * view[0]), float(tgt[1] + d * view[1]), 1.7)
        track.append((pos, (float(tgt[0]), float(tgt[1]), 0.8)))
    return track, root, sep


class ReferenceReplay(_mod.Replay):
    def __init__(self, viewer, args):
        super().__init__(viewer, args)
        self._playing_mode = "reference"
        parquet = str(args.config.task.parquet).strip("'\"")
        self._track, self._ref_root, self._sep = _camera_track(parquet, self.num_frames)
        self._log = []

    def _apply_camera(self, f: int) -> None:
        pos, target = self._track[min(f, len(self._track) - 1)]
        cam = self.viewer.camera
        dx, dy, dz = (b - a for a, b in zip(pos, target))
        r = math.sqrt(dx * dx + dy * dy + dz * dz)
        ux, uy, uz = dx / r, dy / r, dz / r
        if cam.up_axis == 2:
            pitch, yaw = math.degrees(math.asin(uz)), math.degrees(math.atan2(uy, ux))
        elif cam.up_axis == 1:
            pitch, yaw = math.degrees(math.asin(uy)), math.degrees(math.atan2(uz, ux))
        else:
            pitch, yaw = math.degrees(math.asin(ux)), math.degrees(math.atan2(uz, uy))
        self.viewer.set_camera(pos, pitch, yaw)

    def render(self) -> None:
        f = self._shown_frame
        self._apply_camera(f)
        q = self.state.body_q.numpy()[0] if self.state.body_q is not None else None
        finite = bool(q is not None and np.isfinite(q).all())
        self._log.append({"f": int(f), "finite": finite, "body0_z": float(q[2]) if finite else None})
        super().render()


@hydra.main(version_base="1.3", config_path="/workspace/src/flash_chord/configs", config_name="sonic_replay")
def main(cfg: DictConfig) -> None:
    viewer = instantiate_typed(cfg.viewer, ViewerConfig)
    app_holder = {}
    orig_init = ReferenceReplay.__init__

    def init_and_keep(self, v, a):
        orig_init(self, v, a)
        app_holder["app"] = self

    ReferenceReplay.__init__ = init_and_keep
    ReferenceReplay.launch_configured(cfg, viewer, device=cfg.device)
    app = app_holder.get("app")
    out = Path(str(cfg.viewer.output_path)).with_suffix(".json")
    if app is not None:
        log = app._log
        bad = [r["f"] for r in log if not r["finite"]]
        zs = [r["body0_z"] for r in log if r["body0_z"] is not None]
        summary = {"frames_rendered": len(log), "reference_frames": app.num_frames, "first_nonfinite_frame": bad[0] if bad else None,
                   "min_body0_z": min(zs) if zs else None, "max_robot_object_xy_m": float(np.max(app._sep))}
        out.write_text(json.dumps(summary) + "\n")
        print("REF_SUMMARY", json.dumps(summary))


if __name__ == "__main__":
    main()
