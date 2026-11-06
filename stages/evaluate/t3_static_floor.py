"""Tier 3：物体一直停在第 0 帧时，对自己重建参考的 MPPE / SP-SR / MP-SR / ADD-AUC（训练机上经 scripts/dev/in-flash-chord 在容器里跑；挂 tier_guard.forbid_tier1）。
计划 agent-doc/plan-t3static-2026-10-21.md 第 1 节第 2 步（第 2 版）。读评测器用的那份参考：motion parquet 的 object_body_position / object_body_wxyz（wxyz，换成 xyzw），
逐帧回放下一帧源数据对一个控制步，所以参考帧就是 parquet 的全部帧。MPPE、SP-SR、MP-SR 用工具包的官方公式（v2dlb/object_pose.py，与 t2_static_floor.py 同一套）；
ADD-AUC 用网格表面 500 个采样点（seed 0），与评测器的顶点不完全相同，只并列报、不进选集规则。另报物体位姿列的 sha256（核对与原 run 的参考是同一份）。
参考路径从原 run 的 runs/flash_chord/<run>/hydra/s1/.hydra/overrides.yaml 里的 task.parquet 读（t3m1022a 混用几种参考，计划复核 MAJOR 3），同时打出这条 run 的全部覆盖，供站立 run 照抄。
用法（仓库根，容器内）：python stages/evaluate/t3_static_floor.py <工具包根> ep=<run 名> …   最后一行 T3_STATIC_FLOOR_DONE
"""

import hashlib
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "libs" / "v2d_infra"))
from tier_guard import forbid_tier1  # noqa: E402

forbid_tier1()

import numpy as np  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import trimesh  # noqa: E402


REPO = Path("/repo") if Path("/repo/runs").is_dir() else Path(".")


def load_kit(kit: Path):
    spec = importlib.util.spec_from_file_location("kit_object_pose", kit / "v2dlb" / "object_pose.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    kit = Path(sys.argv[1])
    op = load_kit(kit)
    print("集\t帧\t不动 MPPE（cm）\t不动 SP-SR\t不动 MP-SR\t不动 ADD-AUC（近似）\t物体列 sha256\tparquet")
    for arg in sys.argv[2:]:
        ep, run = arg.split("=", 1)
        ov = REPO / "runs" / "flash_chord" / run / "hydra" / "s1" / ".hydra" / "overrides.yaml"
        lines = [x.strip().removeprefix("- ") for x in ov.read_text().splitlines() if x.strip()]
        tp = [x for x in lines if x.startswith("task.parquet=")]
        if len(tp) != 1:
            raise SystemExit(f"{ov}：task.parquet 应有 1 条，读到 {len(tp)} 条")
        path = tp[0].split("=", 1)[1].strip("'\"")
        print(f"# {ep} {run} 覆盖：{' '.join(lines)}")
        if "tier3" not in path:
            raise SystemExit(f"{path} 不是 Tier 3 的参考")
        row = pq.read_table(path, columns=["object_body_position", "object_body_wxyz", "object_mesh_paths"]).to_pylist()[0]
        p = np.asarray(row["object_body_position"], dtype=np.float64)          # (T, B, 3)
        q = np.asarray(row["object_body_wxyz"], dtype=np.float64)              # (T, B, 4) wxyz
        sha = hashlib.sha256(p.tobytes() + q.tobytes()).hexdigest()[:16]
        ref = np.concatenate([p, q[..., 1:], q[..., :1]], axis=-1)[:, None]    # (T, 1, B, 7) xyzw
        static = np.repeat(ref[:1], len(ref), axis=0)
        mesh = trimesh.load(row["object_mesh_paths"][0], force="mesh")
        verts = np.asarray(trimesh.sample.sample_surface(mesh, 500, seed=0)[0], dtype=np.float64)[None]
        mppe = op._mean_per_point_error_cm(static, ref)
        sp = float(np.mean(op._spider_success(static, ref)))
        mp = float(np.mean(op._maniptrans_success(static, ref, np.arange(ref.shape[2]))))
        auc = op._add_auc(op._object_add(static, ref, verts))
        print(f"{ep}\t{len(ref)}\t{mppe:.2f}\t{sp:.3f}\t{mp:.3f}\t{auc:.3f}\t{sha}\t{path}")
    print("T3_STATIC_FLOOR_DONE")


if __name__ == "__main__":
    main()
