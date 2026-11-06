"""生成 Tier 3 打包用的 --frame-references（训练机上、仓库根执行，要 numpy、pyarrow；ml40 宿主机没有 numpy，经 scripts/dev/in-flash-chord 在容器里跑，
容器用 --network host，hostname 同宿主）。计划 agent-doc/plan-r3o-2026-10-15.md 2.4.1。

kit 的 tracks/track_2/README.md：用自己的重建时，打包要加 --frame-references，把策略输入表达在发布的 Track 2 世界系里；每条 episode 一个
episode_XXXXXX.npz，含 episode_index、coordinate_frame=released_track2_world、frame_index [T]、pose_xyzw [T,1,7]（米，xyzw），覆盖所有计分帧。
我们 Tier 3 的策略输入：
  - run 目录 run_record.jsonl 第一条 start 记录的 parquet 路径 /repo/runs/motion/<档位目录>/whole_body/soma/sequence_id=<序列>/robot_name=g1/data.parquet
    给出档位目录（tier3 或 tier3_<名>）与序列（同 fc_export.sh 的读法）；
  - runs/motion_src/<档位目录>/<序列>/poses.npy [T,4,4]：challenge_to_soma 把 Tier 3 root 的 parquet 里 observation.object.pose 原样转成矩阵
    （OpenCV 世界系，不可见帧按前一帧补），重定向读的就是它。这个世界系是重建流程对发布世界系的估计（主机位相机当世界相机，加地面校正与机位配准），
    组委会没确认。打包器那道刚体换算对 Tier 3 只说明帧参考与评测器参考一致（评测器参考本来就是从同一份 poses.npy 刚体变换来的），说明不了世界系对；
  - frame_index 取 0..T−1，前提是来源 parquet 的 frame_index 正好是 0..T−1（下面核）。
核对（按 Fable 5.1 的审查补了来源 parquet、主机与读回）：
  - run 记录 start 的 host 等于本机 hostname（清单的机器列与执行的机器对得上）；
  - v2d_meta.json：tier 3、episode 与清单一致、sequence_id 与 run 记录一致、object_pose_source.kind 是 tier3_reconstruction；
  - 来源 parquet：runs/tier3/root[_<名>]/data/chunk-000/episode_XXXXXX.parquet（档位目录 tier3[_<名>] 对应 root[_<名>]，见 libs/v2d_infra/paths.py），
    sha256 等于 v2d_meta 记的 parquet_sha256，行数等于 poses.npy 的帧数，frame_index 正好是 0..T−1，episode_index 全等于这一集；
    可见帧上 poses.npy 的平移与 observation.object.pose 相同（≤ 1e-9 m），旋转相同（四元数 1 − |点积| ≤ 1e-9）；
  - poses.npy 形状 [T,4,4]、有限、旋转正交（|RᵀR − I| ≤ 1e-6、det > 0）、最后一行 [0,0,0,1]；四元数（w ≥ 0）转回矩阵最大差 ≤ 1e-9；
  - npz 读回再核一遍，不过就删掉这个文件。
进程开头 forbid_tier1()：此后读 Tier 1 的任何文件都会抛错。
用法（仓库根）：python3 stages/evaluate/make_frame_refs_t3.py <选择清单> <机器> --out runs/submission/frame_refs_t3_<名字> [--from-export <导出结果目录>]
  --from-export：策略输入取导出 JSON 的 parquet（先核 JSON 的 checkpoint 等于清单这一行的 run 与步数），不取 run 记录；k-fix 用现有 checkpoint
  对新参考重新导出时用（计划 agent-doc/plan-t3-kfix-2026-10-16.md 方案 4）。
  选择清单每行 tab 分隔：档位 episode run 步数 机器（# 开头为注释，同 pull_submission.sh select）；只做机器一列等于 <机器> 的行，这些行的档位必须是 3。
输出 <out>/episode_XXXXXX.npz 与 <out>/manifest_<机器>.tsv（episode、run、步数、档位目录、序列、帧数、poses.npy 的 sha256、来源 parquet 的 sha256
（这里算的，已核等于 v2d_meta 记的）、文件、sha256）；<out> 已存在就拒绝。拉回 Mac 用 pull_submission.sh frefs，打包前 pack_tier.sh 用
frefs_t3_match.py 逐集核对。最后一行 FRAME_REFS_T3_OK n 或 FRAME_REFS_T3_FAIL n。
测试：stages/evaluate/test_make_frame_refs_t3.py。
"""
import argparse
import hashlib
import json
import re
import socket
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq   # 按属性调用 pq.read_table：forbid_tier1() 包装的是模块上的这个属性

ROOT = Path(__file__).resolve().parents[2]
FRAME = "released_track2_world"
PARQUET = re.compile(r"/repo/runs/motion/(tier3(?:_[a-z0-9]+)?)/whole_body/soma/sequence_id=([^/]+)/robot_name=g1/data\.parquet")
ORTHO_TOL, ROUND_TOL, POS_TOL, QUAT_TOL = 1e-6, 1e-9, 1e-9, 1e-9


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mat_to_xyzw(R: np.ndarray) -> np.ndarray:
    """[T,3,3] 旋转矩阵 → [T,4] xyzw 单位四元数，w ≥ 0。按迹与对角元里最大的一项开方（Shepperd），转角接近 180° 时也稳。"""
    q = np.empty((len(R), 4))
    for i, m in enumerate(R):
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            s = 2.0 * np.sqrt(tr + 1.0)
            q[i] = [(m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s]
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            q[i] = [0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s, (m[2, 1] - m[1, 2]) / s]
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            q[i] = [(m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s, (m[0, 2] - m[2, 0]) / s]
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            q[i] = [(m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s, (m[1, 0] - m[0, 1]) / s]
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    q[q[:, 3] < 0] *= -1.0
    return q


def xyzw_to_mat(q: np.ndarray) -> np.ndarray:
    """[T,4] xyzw → [T,3,3]，公式同 challenge_to_soma.pose7_to_mat。"""
    x, y, z, w = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def start_record(run: str) -> dict:
    for line in (ROOT / "runs/flash_chord" / run / "run_record.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("event") == "start":
                return r
    return {}


def source_parquet(tdir: str, ep: int) -> Path:
    """档位目录 tier3[_<名>] → runs/tier3/root[_<名>] 里这一集的 parquet（libs/v2d_infra/paths.py 的 TIER3 与 tier_dirname）。"""
    return ROOT / "runs/tier3" / ("root" + tdir[len("tier3"):]) / "data/chunk-000" / f"episode_{ep:06d}.parquet"


def check_source(path: Path, ep: int, M: np.ndarray, want_sha) -> tuple:
    """来源 parquet 与 poses.npy 对得上：sha256、行数、frame_index、episode_index、可见帧上的位姿。返回（sha256、不可见帧数）。"""
    got = sha256(path)
    if got != want_sha:
        raise ValueError(f"来源 parquet {path.relative_to(ROOT)} 的 sha256 {got[:16]}… 与 v2d_meta 记的 {str(want_sha)[:16]}… 不同")
    t = pq.read_table(path, columns=["frame_index", "episode_index", "observation.object.pose", "observation.object.visible"])
    T = len(M)
    if t.num_rows != T:
        raise ValueError(f"来源 parquet {t.num_rows} 行，poses.npy {T} 帧")
    fi = np.asarray(t.column("frame_index").to_numpy(), dtype=np.int64)
    epi = np.asarray(t.column("episode_index").to_numpy(), dtype=np.int64)
    if not np.array_equal(fi, np.arange(T)):
        raise ValueError("来源 parquet 的 frame_index 不是 0..T−1（重建的行与视频帧没有一一对应）")
    if not np.all(epi == ep):
        raise ValueError("来源 parquet 的 episode_index 不全等于这一集")
    p7 = np.asarray(t.column("observation.object.pose").combine_chunks().flatten().to_numpy(), dtype=np.float64).reshape(-1, 7)
    vis = np.asarray(t.column("observation.object.visible").to_numpy(zero_copy_only=False), dtype=bool).reshape(-1)
    if not vis.any():
        raise ValueError("来源 parquet 没有可见帧")
    dpos = float(np.abs(p7[vis, :3] - M[vis, :3, 3]).max())
    qn = np.linalg.norm(p7[vis, 3:7], axis=1)
    if (qn < 1e-8).any():
        raise ValueError("来源 parquet 有可见帧的四元数为零")
    q = p7[vis][:, [4, 5, 6, 3]] / qn[:, None]   # wxyz → xyzw
    dq = float((1.0 - np.abs(np.sum(q * mat_to_xyzw(M[vis, :3, :3]), axis=1))).max())
    if dpos > POS_TOL or dq > QUAT_TOL:
        raise ValueError(f"可见帧上 poses.npy 与来源 parquet 的物体位姿不同：平移最大差 {dpos:.2e} m，四元数 1 − |点积| 最大 {dq:.2e}")
    return got, int((~vis).sum())


def motion_path(rec: dict, ep: int, run: str, step: str, export: Path | None) -> tuple:
    """策略输入的 motion parquet 路径与它的出处。默认取 run 记录第一条 start（训练时的参考）；给了 export 时取导出 JSON 的 parquet
    （k-fix 用现有 checkpoint 对新参考重新导出，计划 agent-doc/plan-t3-kfix-2026-10-16.md 方案 4），先核 JSON 的 checkpoint 正是清单这一行的 run 与步数。"""
    if export is None:
        return str(rec.get("parquet", "")), "run 记录第一条 start"
    j = json.loads((export / f"episode_{ep:06d}.json").read_text())
    want = f"/repo/runs/flash_chord/{run}/policy_{step}.safetensors"
    if j.get("checkpoint") != want:
        raise ValueError(f"导出 JSON 的 checkpoint 是 {j.get('checkpoint')}，清单这一行是 {want}")
    return str(j.get("parquet", "")), f"导出 JSON {export.name}/episode_{ep:06d}.json"


def one(ep: int, run: str, step: str, out: Path, host: str, export: Path | None = None) -> tuple:
    rec = start_record(run)
    if rec.get("host") != host:
        raise ValueError(f"run 记录 start 的主机是 {rec.get('host')}，本机是 {host}：清单里这一行的机器列与执行的机器对不上")
    p, where = motion_path(rec, ep, run, step, export)
    m = PARQUET.fullmatch(p)
    if not m:
        raise ValueError(f"{where}的 parquet 不是 Tier 3 的 motion parquet：{p or '没有'}")
    tdir, seq = m.groups()
    return (tdir, seq) + build(ep, tdir, seq, out)


def build(ep: int, tdir: str, seq: str, out: Path) -> tuple:
    """runs/motion_src/<档位目录>/<序列>/poses.npy → <out>/episode_XXXXXX.npz（不经 run；frefs_from_motion_t3.py 也用，计划 agent-doc/plan-t3b2-2026-10-28.md 第 1 步）。
    返回（帧数、poses.npy 的 sha256、来源 parquet 的 sha256、文件名、文件 sha256、|RᵀR − I| 最大、转回最大差、不可见帧数）。"""
    src = ROOT / "runs/motion_src" / tdir / seq
    meta = json.loads((src / "v2d_meta.json").read_text())
    ops = meta.get("object_pose_source") or {}
    if not (str(meta.get("tier")) == "3" and meta.get("episode") == ep and meta.get("sequence_id") == seq and ops.get("kind") == "tier3_reconstruction"):
        raise ValueError(f"v2d_meta.json 不符：tier {meta.get('tier')}、episode {meta.get('episode')}（清单 {ep}）、"
                         f"序列 {meta.get('sequence_id')}（run 记录 {seq}）、来源 {ops.get('kind')}")
    M = np.load(src / "poses.npy", allow_pickle=False)
    if M.ndim != 3 or M.shape[1:] != (4, 4) or len(M) == 0:
        raise ValueError(f"poses.npy 形状 {M.shape}")
    M = M.astype(np.float64)
    if not np.isfinite(M).all():
        raise ValueError("poses.npy 有非有限值")
    R = M[:, :3, :3]
    ortho = float(np.abs(np.einsum("tji,tjk->tik", R, R) - np.eye(3)).max())
    if ortho > ORTHO_TOL or (np.linalg.det(R) <= 0).any() or not np.array_equal(M[:, 3], np.tile([0.0, 0.0, 0.0, 1.0], (len(M), 1))):
        raise ValueError(f"poses.npy 不是刚体变换：|RᵀR − I| 最大 {ortho:.2e}（要 ≤ {ORTHO_TOL:g}），或 det ≤ 0，或最后一行不是 [0,0,0,1]")
    q = mat_to_xyzw(R)
    back = float(np.abs(xyzw_to_mat(q) - R).max())
    if back > ROUND_TOL:
        raise ValueError(f"四元数转回矩阵与原矩阵最大差 {back:.2e}（要 ≤ {ROUND_TOL:g}）")
    psha, nfill = check_source(source_parquet(tdir, ep), ep, M, (ops.get("provenance") or {}).get("parquet_sha256"))
    T = len(M)
    fi = np.arange(T, dtype=np.int64)
    pose = np.concatenate([M[:, :3, 3], q], axis=1)
    path = out / f"episode_{ep:06d}.npz"
    if path.exists():
        raise ValueError(f"{path.name} 已经写过（清单里这台机器上 episode {ep} 出现了两次）")
    np.savez(path, episode_index=np.int64(ep), coordinate_frame=np.array(FRAME), frame_index=fi, pose_xyzw=pose[:, None, :])
    try:   # 照 kit 的 input_frame_reference 读回来核一遍；不过就删掉，免得留下没进 manifest 的文件
        with np.load(path, allow_pickle=False) as z:
            if not (sorted(z.files) == ["coordinate_frame", "episode_index", "frame_index", "pose_xyzw"] and str(z["coordinate_frame"].item()) == FRAME
                    and int(z["episode_index"].item()) == ep and z["pose_xyzw"].shape == (T, 1, 7)
                    and np.array_equal(z["frame_index"], fi) and np.array_equal(z["pose_xyzw"][:, 0], pose)):
                raise ValueError(f"{path.name} 读回来与写进去的不同")
    except Exception:
        path.unlink(missing_ok=True)
        raise
    return T, sha256(src / "poses.npy"), psha, path.name, sha256(path), ortho, back, nfill


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("select", type=Path, help="选择清单：档位 episode run 步数 机器")
    ap.add_argument("machine", choices=("mf4090", "ml40"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--from-export", type=Path, default=None,
                    help="导出结果目录（episode_XXXXXX.json）：参考取导出 JSON 的 parquet，不取 run 记录（k-fix 重新导出用）")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "libs/v2d_infra"))
    from tier_guard import forbid_tier1
    forbid_tier1()   # 档位隔离：此后读 Tier 1 的任何文件都会抛错
    rows, other = [], 0
    for line in a.select.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) < 5:
            sys.exit(f"选择清单的行不足 5 列：{line!r}")
        if f[4] != a.machine:
            other += 1
            continue
        if f[0] != "3":
            sys.exit(f"选择清单里 {a.machine} 的行档位不是 3：{line!r}")
        rows.append((int(f[1]), f[2], f[3]))
    if not rows:
        sys.exit(f"选择清单 {a.select} 里没有 {a.machine} 的行")
    if a.out.exists():
        sys.exit(f"{a.out} 已存在，拒绝覆盖")
    a.out.mkdir(parents=True)
    host = socket.gethostname()
    print(f"== Tier 3 帧参考：{a.machine}（{host}）上 {len(rows)} 集（清单另有 {other} 集在别的机器）→ {a.out}"
          f"（{FRAME}，xyzw，[T,1,7]，frame_index 0..T−1）")
    lines, bad = [], 0
    for ep, run, step in rows:
        try:
            tdir, seq, T, psha, srcsha, f, sha, ortho, back, nfill = one(ep, run, step, a.out, host, a.from_export)
        except Exception as e:   # noqa: BLE001  逐条报
            print(f"  FAIL ep{ep} {run}：{type(e).__name__}: {e}")
            bad += 1
            continue
        lines.append(f"{ep}\t{run}\t{step}\t{tdir}\t{seq}\t{T}\t{psha}\t{srcsha}\t{f}\t{sha}\n")
        print(f"  ok   ep{ep:<2} {run}：{tdir}/{seq}，{T} 帧；来源 parquet 的 sha256、frame_index、episode_index 与可见帧位姿都对得上"
              f"（不可见 {nfill} 帧按前一帧补）；|RᵀR − I| 最大 {ortho:.1e}，四元数转回矩阵最大差 {back:.1e}")
    (a.out / f"manifest_{a.machine}.tsv").write_text(
        "episode\trun\tstep\ttier_dir\tsequence\tframes\tposes_sha256\tsource_parquet_sha256\tfile\tsha256\n" + "".join(lines), encoding="utf-8")
    print(f"FRAME_REFS_T3_OK {len(lines)}" if bad == 0 else f"FRAME_REFS_T3_FAIL {bad}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
