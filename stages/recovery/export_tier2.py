"""Tier 2 物体轨迹降噪并落盘。recovery → retarget 的交接产物。

版本：v5.2 = v51 支撑面 3 自由度 + 布朗桥 + 放宽静止判据（0.734）；
      v5.3 = v5.2 + 不穿地约束（nopen.py，rot_s：绕原点最小旋转 + 时间平滑，0.748，见 nopen_eval.py）。
      v5.4 = v5.3 的算法 + **净室参数**（runs/recovery/cleanroom_params.json，只由 Tier 2 推出，
             见 cleanroom_params.py）。v5.2/v5.3 的门限与静止判据来自 Tier 1，合规有疑问。
用法：python stages/recovery/export_tier2.py v5.2|v5.3|v5.4   必须写明版本，没有默认值（v5.2 / v5.3 的门限与静止判据来自 Tier 1；
      提交用 v5.4）；输出到 runs/recovery/tier2_v52、tier2_v53 或 tier2_v54

只读 Tier 2 档位目录（paths.TIER2）：物体轨迹、ground_plane、support_surface、mesh。
不读 Tier 1 的任何文件——档位隔离。降噪效果的核对在另一个分析脚本里做（check_tier2_export.py）。

输出 runs/recovery/tier2_v5X/episode_XXXXXX.npz：
  object_pose (T,7) float64，[x,y,z,qw,qx,qy,qz]，**OpenCV 世界系**（与挑战 parquet 同一约定），全长 T 帧
  visible (T,)       原始可见性；不可见帧先沿用上一有效位姿再参与降噪
  meta               JSON：denoiser、参数、v51.py md5、源 parquet
"""
import hashlib
import json
import sys
from pathlib import Path

VERSIONS = {"v5.2": "tier2_v52", "v5.3": "tier2_v53", "v5.4": "tier2_v54"}
if len(sys.argv) != 2 or sys.argv[1] not in VERSIONS:   # 先于一切 import 与读数据（计划 agent-doc/plan-tier2-denoise-2026-10-04.md 第 8 节）
    sys.exit(f"用法：python stages/recovery/export_tier2.py {'|'.join(VERSIONS)}（必须写明版本，没有默认值；提交用 v5.4）")
VERSION = sys.argv[1]

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "libs" / "v2d_infra")); sys.path.insert(0, str(HERE))
from tier_guard import forbid_tier1
forbid_tier1()          # 档位隔离的机制守卫：本进程此后任何读取 Tier 1 的动作都会抛错
import numpy as np
import pyarrow.parquet as pq
from paths import RUNS, TIER2
from soma_fk import col, R_CV
from t2core import to_soma                             # 纯函数；denoise_v4 import 时会读 Tier 1 metadata
from v51 import V52, denoise_v51, surface_tops, stable_ups
from nopen import enforce, hull_vertices

OUT = RUNS / "recovery" / VERSIONS[VERSION]
if VERSION == "v5.4":
    _CR = json.loads((RUNS / "recovery" / "cleanroom_params.json").read_text())["params"]
    PARAMS = {k: _CR[k] for k in ("ref_tilt", "ref_h", "fb_tilt", "fb_h", "tilt_cap", "h_cap", "st_win", "st_rot",
                                  "st_tr", "margin", "eq_ang", "eq_h")}
    NOPEN_MAX_ROT = np.radians(_CR["nopen_max_rot_deg"])
else:
    PARAMS, NOPEN_MAX_ROT = V52, None


def fill_invisible(p, vis):
    out = p.copy(); first = int(np.argmax(vis)); out[:first] = p[first]; last = first
    for i in range(first, len(p)):
        if vis[i]: last = i
        else: out[i] = p[last]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    md5 = {f: hashlib.md5((HERE / f).read_bytes()).hexdigest() for f in ("v51.py", "nopen.py")}
    meta = {json.loads(l)["episode_index"]: json.loads(l) for l in open(TIER2 / "meta" / "episodes_metadata.jsonl")}
    for ei, m in sorted(meta.items()):
        f = TIER2 / "data" / "chunk-000" / f"episode_{ei:06d}.parquet"
        t = pq.read_table(f)
        vis = np.asarray(col(t, "observation.object.visible")).astype(bool).ravel()
        p_cv = fill_invisible(col(t, "observation.object.pose").astype(np.float64), vis)
        p = to_soma(p_cv)
        g = json.load(open(TIER2 / m["ground_plane"]))["plane"]
        s_ = np.linalg.norm(g[:3]); n = R_CV @ (np.array(g[:3]) / s_); d = g[3] / s_
        if np.median(p[:, :3] @ n + d) < 0: n, d = -n, -d
        q = denoise_v51(p, n, d, surface_tops(TIER2 / m["support_surface"]), stable_ups(str(TIER2 / m["mesh"])), PARAMS)
        if VERSION in ("v5.3", "v5.4"):
            kw = {} if NOPEN_MAX_ROT is None else {"max_rot": NOPEN_MAX_ROT}
            q = enforce(q, hull_vertices(str(TIER2 / m["mesh"])), n, d, mode="rot_s", **kw)
        q_cv = to_soma(q)                                    # R_CV 是对合，同一变换即回到 OpenCV 系
        np.savez(OUT / f"episode_{ei:06d}.npz", object_pose=q_cv, visible=vis,
                 meta=np.array(json.dumps({"denoiser": VERSION, "params": PARAMS, "md5": md5,
                                           "source": str(f.relative_to(TIER2.parents[2]))})))
        print(f"ep{ei:02d} {m['object'][:20]:<20} T={len(q_cv)} 不可见={int((~vis).sum())}")
    print(f"\n写出 {len(meta)} 条到 {OUT}")


if __name__ == "__main__":
    main()
