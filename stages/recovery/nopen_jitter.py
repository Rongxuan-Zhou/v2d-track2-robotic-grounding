import pathlib
"""nopen_jitter: 事后补测（nopen_eval 的判据④用全体帧中位数，对被修改的少数帧不敏感）。

只看 rot_s 实际改动过的帧（旋转改动 > 0.1°）及其前后 1 帧，比较角加速度 |Δω|（°/帧²）：
v5.2 vs rot_s，中位与 p99。只用 Tier 2 输入，不读 Tier 1。
事后判据：修改帧上 p99 ≤ 1.2 × v5.2 同一批帧的 p99。
"""
import sys,json,glob,warnings; warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "libs" / "v2d_infra")); sys.path.insert(0, str(HERE))
import numpy as np, pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as Rot
from soma_fk import col, R_CV
from paths import TIER2
from denoise_v4 import to_soma
from v51 import V52, denoise_v51, surface_tops, stable_ups
from nopen import enforce, hull_vertices

meta = {json.loads(l)["episode_index"]: json.loads(l) for l in open(TIER2 / "meta" / "episodes_metadata.jsonl")}
def acc(p):
    Rm = Rot.from_quat(p[:, [4, 5, 6, 3]]); w = (Rm[1:] * Rm[:-1].inv()).as_rotvec()
    return np.r_[0, 0, np.degrees(np.linalg.norm(np.diff(w, axis=0), axis=1))]
A52, AS = [], []
for ei, m in sorted(meta.items()):
    t = pq.read_table(TIER2 / "data" / "chunk-000" / f"episode_{ei:06d}.parquet")
    vis = np.asarray(col(t, "observation.object.visible")).astype(bool).ravel()
    p = to_soma(col(t, "observation.object.pose")[vis])
    g = json.load(open(TIER2 / m["ground_plane"]))["plane"]; s_ = np.linalg.norm(g[:3]); n = R_CV @ (np.array(g[:3]) / s_); d = g[3] / s_
    if np.median(p[:, :3] @ n + d) < 0: n, d = -n, -d
    b = denoise_v51(p, n, d, surface_tops(TIER2 / m["support_surface"]), stable_ups(str(TIER2 / m["mesh"])), V52)
    q = enforce(b, hull_vertices(str(TIER2 / m["mesh"])), n, d, mode="rot_s")
    dr = np.degrees(np.linalg.norm((Rot.from_quat(q[:, [4,5,6,3]]) * Rot.from_quat(b[:, [4,5,6,3]]).inv()).as_rotvec(), axis=1))
    mod = dr > 0.1; mod = mod | np.r_[mod[1:], False] | np.r_[False, mod[:-1]]
    if mod.any(): A52.append(acc(b)[mod]); AS.append(acc(q)[mod])
a, s = np.concatenate(A52), np.concatenate(AS)
print(f"被修改帧（含前后 1 帧）共 {len(a)} 帧")
print(f"角加速度 °/帧²   v5.2: 中位 {np.median(a):.3f}  p99 {np.percentile(a,99):.3f}   rot_s: 中位 {np.median(s):.3f}  p99 {np.percentile(s,99):.3f}")
print("事后判据（修改帧 p99 ≤ 1.2×v5.2）：" + ("通过" if np.percentile(s, 99) <= 1.2 * np.percentile(a, 99) else "不通过"))
