"""ADD / ADD-AUC between two object-pose trajectories of the same mesh.

Implements DexMachina's metric as released in dexmachina/eval/compute_add.py:
  - sample N vertices per object part
  - per frame: mean L2 distance between vertices posed by the two trajectories
  - AUC: accuracy(thr) = mean(ADD < thr) for thr in 1..9 cm,
         x normalised to [0,1], trapezoidal integration
ADD, not ADD-S: vertices correspond one-to-one, symmetry is NOT forgiven.
"""
import numpy as np

THRESHOLDS = np.arange(0.01, 0.10, 0.01)   # 1..9 cm, 9 values


def quat_wxyz_to_matrix(q):
    """(...,4) wxyz -> (...,3,3). Normalises input."""
    q = np.asarray(q, dtype=np.float64)
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    m[..., 0, 0] = 1 - 2 * (y * y + z * z)
    m[..., 0, 1] = 2 * (x * y - w * z)
    m[..., 0, 2] = 2 * (x * z + w * y)
    m[..., 1, 0] = 2 * (x * y + w * z)
    m[..., 1, 1] = 1 - 2 * (x * x + z * z)
    m[..., 1, 2] = 2 * (y * z - w * x)
    m[..., 2, 0] = 2 * (x * z - w * y)
    m[..., 2, 1] = 2 * (y * z + w * x)
    m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def transform_vertices(verts, pose):
    """verts (V,3), pose (T,7) [x,y,z,qw,qx,qy,qz] -> (T,V,3)."""
    R = quat_wxyz_to_matrix(pose[:, 3:7])             # (T,3,3)
    return np.einsum("tij,vj->tvi", R, verts) + pose[:, None, :3]


def add_per_frame(verts, pose_ref, pose_est):
    """Mean per-vertex L2 distance, per frame. Returns (T,) in metres."""
    a = transform_vertices(verts, pose_ref)
    b = transform_vertices(verts, pose_est)
    return np.linalg.norm(a - b, axis=-1).mean(axis=-1)


def auc_from_add(add, thresholds=THRESHOLDS):
    """DexMachina compute_auc: accuracy sweep, x on [0,1], trapezoid."""
    acc = [float(np.mean(add < t)) for t in thresholds]
    x = np.linspace(0, 1, len(thresholds))
    return float(np.trapezoid(acc, x=x)), acc


def pose_errors(pose_ref, pose_est):
    """Translation (m) and rotation (rad) error per frame."""
    dt = np.linalg.norm(pose_ref[:, :3] - pose_est[:, :3], axis=1)
    q1 = pose_ref[:, 3:7] / (np.linalg.norm(pose_ref[:, 3:7], axis=1, keepdims=True) + 1e-12)
    q2 = pose_est[:, 3:7] / (np.linalg.norm(pose_est[:, 3:7], axis=1, keepdims=True) + 1e-12)
    dq = 2 * np.arccos(np.clip(np.abs((q1 * q2).sum(1)), 0, 1))
    return dt, dq


def sample_mesh_vertices(mesh_path, n=500, seed=0):
    """Load a .glb/.obj and return n vertices (V,3) in object-local metres."""
    import trimesh
    m = trimesh.load(mesh_path, force="scene")
    if isinstance(m, trimesh.Scene):
        m = m.dump(concatenate=True)
    v = np.asarray(m.vertices, dtype=np.float64)
    if len(v) <= n:
        return v
    rng = np.random.default_rng(seed)
    return v[rng.choice(len(v), n, replace=False)]
