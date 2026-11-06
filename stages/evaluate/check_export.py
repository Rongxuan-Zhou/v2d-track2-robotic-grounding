"""核对一份 8c778f9 导出（stages/evaluate/fc_export.sh 的输出）：结构、来历摘要与 metrics JSON（训练机宿主机上跑：numpy、pyarrow）。

计划 agent-doc/plan-box6-and-eval-runtime-2026-10-01.md 第 3 节的判据 P2、P3。逐项打 ok / FAIL；全过最后一行 EXPORT_OK，否则 EXPORT_FAIL n、退出码 1。
  P2 结构：轨迹 parquet 的元数据是 flash_chord_object_trajectories_v1、米、xyzw，世界数对、物体刚体数等于 motion parquet 的刚体数、起始帧 0；
     步数 = motion parquet 的帧数；行数 = 步数 × 世界数 × 刚体数；每一步恰好“世界数 × 刚体数”行；reference_frame = step；environment_valid 全真；位姿全有限。
  P3 来历：轨迹元数据的 checkpoint_sha256 = 这份 policy 文件的 sha256；eval_code_sha256、eval_script_sha256 等于
     third_party/patches/flash_chord-8c778f9.eval.json（tools/make_fc_patch.sh 按上游 8c778f9 的源码算）；reference_parquet = metrics JSON 的 parquet。
     metrics JSON：world_count、step_count、checkpoint_sha256 对得上，non_finite_worlds 为空。
另报（不作判据）：评测器记下的参考位姿与 motion parquet 的物体位姿逐帧最大差（位置 mm、朝向度），以及 metrics JSON 的主要指标。
用法（仓库根）：python3 stages/evaluate/check_export.py <run> <步数> <世界数> [--no-traj]
  在 runs/flash_chord/<run>/export_8c778f9/s<步数>_w<世界数>/ 里找唯一的 episode_*.json 与同名 .parquet；--no-traj 只核 JSON。
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
DIGESTS = ROOT / "third_party/patches/flash_chord-8c778f9.eval.json"
SCHEMA = "flash_chord_object_trajectories_v1"
ACH = [f"achieved_{k}" for k in ("position_x", "position_y", "position_z", "quaternion_x", "quaternion_y", "quaternion_z", "quaternion_w")]
REF = [c.replace("achieved_", "reference_") for c in ACH]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def host(path: str) -> Path:
    """容器里的 /repo/… 换成仓库里的路径。"""
    return ROOT / path[len("/repo/"):] if path.startswith("/repo/") else Path(path)


class Gate:
    def __init__(self) -> None:
        self.fails = 0

    def __call__(self, ok: bool, msg: str) -> bool:
        print(f"  {'ok  ' if ok else 'FAIL'} {msg}")
        self.fails += not ok
        return ok


def motion_reference(parquet: Path) -> tuple:
    """motion_v1 parquet（一行存整条序列）里的物体刚体数，以及第一个刚体的逐帧位置与 xyzw 四元数。"""
    t = pq.read_table(parquet, columns=["object_body_position", "object_body_wxyz"])
    pos = np.asarray(t.column("object_body_position")[0].as_py(), dtype=np.float64)
    wxyz = np.asarray(t.column("object_body_wxyz")[0].as_py(), dtype=np.float64)
    pos, wxyz = pos.reshape(len(pos), -1, 3), wxyz.reshape(len(wxyz), -1, 4)
    return pos.shape[1], pos[:, 0], np.concatenate([wxyz[:, 0, 1:], wxyz[:, 0, :1]], axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("step", type=int)
    ap.add_argument("worlds", type=int)
    ap.add_argument("--no-traj", action="store_true")
    a = ap.parse_args()
    g = Gate()
    d = ROOT / "runs/flash_chord" / a.run / "export_8c778f9" / f"s{a.step}_w{a.worlds}"
    js = sorted(d.glob("episode_*.json"))
    print(f"== {a.run} 第 {a.step:,d} 步，{a.worlds} 世界：{d.relative_to(ROOT)}")
    if not g(len(js) == 1, f"恰好一份 episode_*.json（找到 {len(js)} 份）"):
        print(f"EXPORT_FAIL {g.fails}")
        return 1
    jpath = js[0]
    rep = json.loads(jpath.read_text())
    policy = ROOT / "runs/flash_chord" / a.run / f"policy_{a.step}.safetensors"
    ck = sha256(policy)
    ref_parquet = host(rep.get("parquet", ""))
    nbody, pos_ref, quat_ref = motion_reference(ref_parquet)
    frames = len(pos_ref)
    digests = json.loads(DIGESTS.read_text())
    print(f"  info policy sha256 {ck[:16]}…；motion parquet {rep.get('parquet')}（{frames} 帧）")

    g(rep.get("world_count") == a.worlds, f"JSON world_count {rep.get('world_count')} = {a.worlds}")
    g(rep.get("step_count") == frames, f"JSON step_count {rep.get('step_count')} = motion 帧数 {frames}")
    g(rep.get("checkpoint_sha256") == ck, "JSON checkpoint_sha256 = policy 文件的 sha256")
    g(rep.get("non_finite_worlds") == [], f"JSON non_finite_worlds 为空（{rep.get('non_finite_worlds')}）")
    m = rep.get("metrics", {})
    keys = ("add_auc", "chord_sr", "maniptrans_sr", "spider_sr_uncentered", "mppe_cm", "mean_per_point_error_cm", "relative_position_error_cm")
    print("  info " + "，".join(f"{k} {m[k]:.4f}" for k in keys if isinstance(m.get(k), (int, float))))

    if not a.no_traj:
        qpath = jpath.with_suffix(".parquet")
        if g(qpath.is_file(), f"有 {qpath.name}"):
            pf = pq.ParquetFile(qpath)
            md = {k.decode(): v.decode() for k, v in (pf.schema_arrow.metadata or {}).items()}
            t = pf.read()
            g(md.get("flash_chord.schema") == SCHEMA, f"schema {md.get('flash_chord.schema')}")
            g(md.get("flash_chord.position_unit") == "meter" and md.get("flash_chord.quaternion_order") == "xyzw",
              f"单位 {md.get('flash_chord.position_unit')}、四元数 {md.get('flash_chord.quaternion_order')}")
            g(md.get("flash_chord.world_count") == str(a.worlds) and md.get("flash_chord.object_body_count") == str(nbody)
              and md.get("flash_chord.start_frame") == "0",
              f"元数据 world_count {md.get('flash_chord.world_count')}、object_body_count {md.get('flash_chord.object_body_count')}、"
              f"start_frame {md.get('flash_chord.start_frame')}")
            g(md.get("flash_chord.step_count") == str(frames), f"元数据 step_count {md.get('flash_chord.step_count')} = {frames}")
            per = a.worlds * nbody
            g(t.num_rows == frames * per, f"行数 {t.num_rows} = {frames} × {a.worlds} 世界 × {nbody} 个刚体")
            step = t.column("step").to_numpy()
            counts = np.bincount(step, minlength=frames) if len(step) else np.zeros(frames, int)
            g(len(counts) == frames and bool((counts == per).all()), f"step 0..{frames - 1} 每步 {per} 行")
            g(bool((t.column("reference_frame").to_numpy() == step).all()), "reference_frame = step（起始帧 0、连续）")
            g(bool(t.column("environment_valid").to_numpy(zero_copy_only=False).all()), "environment_valid 全真")
            ach = np.stack([t.column(c).to_numpy() for c in ACH], axis=1)
            ref = np.stack([t.column(c).to_numpy() for c in REF], axis=1)
            g(bool(np.isfinite(ach).all() and np.isfinite(ref).all()), "achieved / reference 位姿全有限")
            g(md.get("flash_chord.checkpoint_sha256") == ck, "元数据 checkpoint_sha256 = policy 文件的 sha256")
            g(md.get("flash_chord.eval_code_sha256") == digests["eval_code_sha256"],
              f"元数据 eval_code_sha256 {str(md.get('flash_chord.eval_code_sha256'))[:16]}… = 8c778f9 的 {digests['eval_code_sha256'][:16]}…")
            g(md.get("flash_chord.eval_script_sha256") == digests["eval_script_sha256"],
              f"元数据 eval_script_sha256 {str(md.get('flash_chord.eval_script_sha256'))[:16]}… = 8c778f9 的 {digests['eval_script_sha256'][:16]}…")
            g(md.get("flash_chord.reference_parquet") == rep.get("parquet"), "元数据 reference_parquet = JSON 的 parquet")
            print(f"  info 元数据 episode_index {md.get('flash_chord.episode_index', '（无，打包器按文件名 ' + qpath.name + '）')}")
            if t.num_rows == frames * per:
                r0 = ref.reshape(frames, a.worlds, nbody, 7)[:, 0, 0]
                dp = np.linalg.norm(r0[:, :3] - pos_ref, axis=1).max() * 1000
                dot = np.abs(np.sum(r0[:, 3:] / np.linalg.norm(r0[:, 3:], axis=1, keepdims=True) * quat_ref, axis=1))
                da = np.degrees(2 * np.arccos(np.clip(dot, 0, 1))).max()
                print(f"  info 评测器记下的参考位姿 vs motion parquet 的物体位姿：位置最大差 {dp:.4f} mm，朝向最大差 {da:.4f}°（只报，不作判据）")
    print("EXPORT_OK" if g.fails == 0 else f"EXPORT_FAIL {g.fails}")
    return 0 if g.fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
