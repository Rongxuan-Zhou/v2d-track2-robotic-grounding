"""按官方评测路径跑一次，同时把被打分的那个 rollout 存成 npz（在 FlashCHORD 容器里跑，占 GPU）。

做法：在 flash_chord.evaluation.harness 模块里包一层 record_object_pose_rollout 与 score_reference_tracking（harness 在模块顶层
导入它们、在 evaluate_checkpoint 里按名字调用），然后用 runpy 运行**未改动的** scripts/evaluate_policy.py，参数照抄
stages/policy/fc_eval.sh。评测 JSON 由上游脚本自己写；存下的逐帧数据就是它打分用的那个 rollout。
两个函数必须各恰好被调用 1 次，物体刚体必须只有 1 个，数值必须全部有限，否则报错退出。
另存（都用评测器自己的函数与顶点）：逐帧逐世界 ADD、“物体保持第 0 帧参考位姿”的逐帧 ADD、参考手（palm 坐标帧）位置与它到
参考物体网格顶点的最近距离、参考里各手的接触标记、上游四个文件的 md5。ADD 存 float64，其余浮点数组存 float32。
--palms（计划 agent-doc/plan-diag-mpsr0-2026-10-02.md 第 3.1 节）：另经 stages/evaluate/diag_palms.py 在每一步之后抄下机器人两只手掌的世界坐标
（robot_palm_pos，(T,W,H,3)，H 按 robot_palm_sides），同时抄下物体位置与评测器记的 achieved_pose_w 逐位比（palm_object_sync_max_abs_m，
不含非有限的世界）；参考位姿改存 float64，供 diag_phase.py 逐位重算评测器的成功判据。不加 --palms 时与原来完全相同。

--actions N [--action-worlds 256]（计划 agent-doc/plan-t12sat-2026-10-20.md）：另经 diag_palms.ActionRecorder 抄前 N 步、前若干个世界的策略动作（policy_action，(N,W',A)，
还没乘 residual_scale；剔除非有限世界后的编号在 policy_action_worlds）。不加时与原来完全相同。
--bodies（计划 agent-doc/plan-t12fall-2026-10-20.md）：另经 diag_palms.BodyRecorder 抄骨盆、躯干、两只脚踝的逐帧位姿（robot_body_pose，(T,W,4,7)，位置 + 四元数 xyzw，
与 robot_body_names）。不加 --bodies 时与原来完全相同。
--fingers（agent-doc/experiments/2026-10-30-t1hard-diag.md）：另经 diag_palms.FingerRecorder 抄 14 个 Dex3 手指关节的逐帧实际角度与 PD 目标角
（robot_finger_q、robot_finger_target，(T,W,14)，与 robot_finger_names）。不加 --fingers 时与原来完全相同。
--parquet <容器内路径>（计划 agent-doc/plan-t12ref-2026-10-20.md）：把 checkpoint 配置里的参考 parquet 换成这一份（包一层 harness._resolve_parquet），用零残差回放试改过的参考；
评测 JSON 里的 parquet 不是它就作废；网格按它解析；npz 记 reference_parquet（不加时记 checkpoint 配置里那份）。
用法：python /repo/stages/evaluate/diag_rollout.py <policy.safetensors> <输出前缀> --worlds 4096 [--palms] [--bodies] [--fingers] [--parquet <p>]   （容器内路径）
输出：<前缀>.npz 与 <前缀>.json（上游 evaluate_policy.py 写的评测报告，格式同 fc_eval）
"""
import argparse
import dataclasses
import hashlib
import json
import runpy
import struct
import sys
from pathlib import Path

import numpy as np

EVAL_SCRIPT = "/workspace/scripts/evaluate_policy.py"


def md5(path: str) -> str:
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def attr(obj, name):
    v = getattr(obj, name)
    return v() if callable(v) else v


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("policy")
    ap.add_argument("out_prefix")
    ap.add_argument("--worlds", type=int, default=4096)
    ap.add_argument("--palms", action="store_true", help="另抄机器人两只手掌的逐帧位置，见文件头")
    ap.add_argument("--parquet", help="换掉 checkpoint 配置里的参考 parquet（容器内绝对路径；计划 agent-doc/plan-t12ref-2026-10-20.md：用零残差回放试改过的参考）")
    ap.add_argument("--actions", type=int, default=0, help="另抄前 N 步策略动作（diag_palms.ActionRecorder，计划 agent-doc/plan-t12sat-2026-10-20.md）")
    ap.add_argument("--action-worlds", type=int, default=256)
    ap.add_argument("--fingers", action="store_true", help="另抄 14 个手指关节的逐帧角度与目标角（diag_palms.FingerRecorder）")
    ap.add_argument("--bodies", action="store_true", help="另抄骨盆、躯干、两只脚踝的逐帧位姿（diag_palms.BodyRecorder，计划 agent-doc/plan-t12fall-2026-10-20.md）")
    a = ap.parse_args()
    out = a.out_prefix
    for suffix in (".npz", ".json"):
        Path(out + suffix).unlink(missing_ok=True)  # 旧输出先删，失败时不会误读上一次的

    import trimesh
    from flash_chord.assets.registry import resolve_asset_path
    from flash_chord.evaluation import harness, metrics, rollout

    calls = {"record": 0, "score": 0}
    cap = {}
    record, score = harness.record_object_pose_rollout, harness.score_reference_tracking

    def record_wrapper(*args, **kwargs):
        calls["record"] += 1
        if a.palms:   # 换掉传进来的 jax_env：它的 step 返回前抄一次手掌与物体位置
            import diag_palms
            args = list(args)
            env = args[0] if args else kwargs["env"]
            cap["palms"] = diag_palms.PalmRecorder(env, kwargs["step_count"])
            if len(args) > 1:
                args[1] = cap["palms"].wrap(args[1])
            else:
                kwargs["jax_env"] = cap["palms"].wrap(kwargs["jax_env"])
        if a.bodies:   # 同一时机再抄一组刚体位姿；套在 --palms 的包装外面
            import diag_palms
            args = list(args)
            env = args[0] if args else kwargs["env"]
            cap["bodies"] = diag_palms.BodyRecorder(env, kwargs["step_count"])
            if len(args) > 1:
                args[1] = cap["bodies"].wrap(args[1])
            else:
                kwargs["jax_env"] = cap["bodies"].wrap(kwargs["jax_env"])
        if a.fingers:   # 同一时机抄手指关节角与目标角
            import diag_palms
            args = list(args)
            env = args[0] if args else kwargs["env"]
            cap["fingers"] = diag_palms.FingerRecorder(env, kwargs["step_count"])
            if len(args) > 1:
                args[1] = cap["fingers"].wrap(args[1])
            else:
                kwargs["jax_env"] = cap["fingers"].wrap(kwargs["jax_env"])
        if a.actions > 0:   # 最外层：抄的是策略交给 step 的动作
            import diag_palms
            args = list(args)
            cap["actions"] = diag_palms.ActionRecorder(a.actions, a.action_worlds)
            if len(args) > 1:
                args[1] = cap["actions"].wrap(args[1])
            else:
                kwargs["jax_env"] = cap["actions"].wrap(kwargs["jax_env"])
        cap["rollout"] = record(*args, **kwargs)
        cap["reference"] = kwargs.get("reference")
        return cap["rollout"]

    def score_wrapper(*args, **kwargs):
        calls["score"] += 1
        cap["criteria"] = args[1] if len(args) > 1 else kwargs["criteria"]
        cap["tracking"] = score(*args, **kwargs)
        return cap["tracking"]

    if a.parquet:   # 评测器按 checkpoint 自带配置解析参考路径（harness._resolve_parquet），这里换成给定的那份；官方源码不改
        if not Path(a.parquet).is_file():
            print(f"没有 {a.parquet}", file=sys.stderr)
            return 2
        harness._resolve_parquet = lambda config, source_root: str(Path(a.parquet).resolve())
    harness.record_object_pose_rollout = record_wrapper
    harness.score_reference_tracking = score_wrapper
    sys.argv = [EVAL_SCRIPT, f"evaluation.checkpoint={a.policy}", f"evaluation.world_count={a.worlds}",
                f"evaluation.metrics_output={out}.json", f"hydra.run.dir={out}_hydra"]
    sys.path.insert(0, str(Path(EVAL_SCRIPT).parent))  # 与直接运行该脚本时的 sys.path 一致
    try:
        runpy.run_path(EVAL_SCRIPT, run_name="__main__")
    except SystemExit as e:  # 上游若以 sys.exit(...) 结束：成功就继续存 npz，失败原样抛出
        if e.code not in (0, None):
            raise
    if a.parquet:   # 换参考时，评测器写进 JSON 的 parquet 必须正是新参考，否则作废（同 eval_playback_override.py）
        report = json.loads(Path(out + ".json").read_text()) if Path(out + ".json").exists() else {}
        if str(report.get("parquet", "")).strip("'") != str(Path(a.parquet).resolve()):
            print(f"评测 JSON 的 parquet 是 {report.get('parquet')}，要换成的是 {a.parquet}：换参考没有生效，结果作废", file=sys.stderr)
            return 2

    if calls != {"record": 1, "score": 1} or cap.get("reference") is None:
        print(f"包装调用次数不对或拿不到参考：{calls}；上游可能改了 evaluate_checkpoint 的调用方式", file=sys.stderr)
        return 2
    r, c, ref = cap["rollout"], cap["criteria"], cap["reference"]
    ach, refp = np.asarray(r.achieved_pose_w, np.float64), np.asarray(r.reference_pose_w, np.float64)
    verts = np.asarray(r.object_vertices_o, np.float64)
    T, W, B = ach.shape[:3]
    if B != 1:
        print(f"物体刚体有 {B} 个，本诊断只支持 1 个", file=sys.stderr)
        return 2
    add = metrics.object_add(ach, refp, verts)[:, :, 0]                                            # (T,W) m
    add_static = metrics.object_add(np.broadcast_to(refp[:1], refp.shape)[:, None], refp, verts)[:, 0, 0]  # (T,) m

    # 参考手：palm 坐标帧位置（与评测器 palm 误差用的是同一组参考手腕），到参考物体网格（全部顶点）的最近距离
    s0 = int(r.start_frame)
    names = list(attr(ref, "robot_frame_names"))
    fpos = np.asarray(attr(ref, "robot_frame_pos_w"), np.float64)[s0:s0 + T]                     # (T,F,3)
    palm_idx = [i for i, n in enumerate(names) if "palm" in n]
    if a.parquet:   # 换过参考：网格按新参考解析
        parquet = str(Path(a.parquet).resolve())
    else:
        with open(a.policy, "rb") as f:  # 参考 parquet 取自 policy 内嵌的配置（与评测器重建环境用的是同一份）
            n = struct.unpack("<Q", f.read(8))[0]
            parquet = json.loads(json.loads(f.read(n))["__metadata__"]["user.resolved_config_json"])["task"]["parquet"].strip("'")
    mesh_paths = list(attr(ref, "object_mesh_paths"))
    mesh = np.asarray(trimesh.load(resolve_asset_path(str(mesh_paths[0]), reference_path=parquet), force="mesh").vertices,
                      np.float64)
    # trimesh 读的网格与评测器（仿真模型里已缩放、已变换的网格）是否同一坐标系：按评测器的抽样规则从 trimesh 顶点重抽，
    # 与 rollout 的 object_vertices_o 比。一致才用全部顶点算距离，否则退回评测器的 500 个顶点
    pick = np.random.default_rng(metrics.ADD_VERTEX_SEED).choice(len(mesh), size=metrics.ADD_VERTEX_COUNT,
                                                                  replace=len(mesh) < metrics.ADD_VERTEX_COUNT)
    trimesh_matches = bool(np.allclose(mesh[pick], verts[0], atol=1e-5))
    dist_verts = mesh if trimesh_matches else verts[0]
    q = refp[:, 0, 3:7]
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    rot = np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                    np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                    np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)
    palm_dist = np.empty((T, len(palm_idx)))
    for t in range(T):
        world_verts = dist_verts @ rot[t].T + refp[t, 0, :3]
        for k, i in enumerate(palm_idx):
            palm_dist[t, k] = np.sqrt(((world_verts - fpos[t, i]) ** 2).sum(1).min())
    sides = list(attr(ref, "sides"))
    contact = np.stack([np.asarray(ref.contact_active(s), bool)[s0:s0 + T] for s in sides], -1) if sides else np.zeros((T, 0), bool)

    arrays = {"achieved_pose_w": ach, "reference_pose_w": refp, "tracking_error": np.asarray(r.tracking_error, np.float64),
              "add": add, "add_static": add_static, "ref_palm_pos": fpos[:, palm_idx], "ref_palm_mesh_dist": palm_dist}
    extra, keep64 = {}, {"add", "add_static"}
    if a.palms:
        palms, objects = cap["palms"].arrays()
        ok = np.ones(W, bool); ok[list(r.non_finite_worlds)] = False
        sync = float(np.abs(objects[:, ok].astype(np.float64) - ach[:, ok, :, :3]).max()) if ok.any() else float("nan")
        palms[:, ~ok] = 0.0   # 同评测器对非有限世界的做法：清零后照存（npz 留着排查），diag_phase.py 的 V0 见到非有限世界就不分类
        arrays["robot_palm_pos"] = palms.astype(np.float64)
        extra = {"robot_palm_sides": np.array(cap["palms"].sides), "palm_object_sync_max_abs_m": np.float64(sync)}
        keep64.add("reference_pose_w")   # 评测器用 float64 的参考位姿算成功判据；逐位重算要同一精度
    if a.bodies:
        bodies = cap["bodies"].arrays()
        okb = np.ones(W, bool); okb[list(r.non_finite_worlds)] = False
        bodies[:, ~okb] = 0.0
        arrays["robot_body_pose"] = bodies.astype(np.float64)
        extra = {**extra, "robot_body_names": np.array(cap["bodies"].names)}
    if a.fingers:
        fq, ft = cap["fingers"].arrays()
        okf = np.ones(W, bool); okf[list(r.non_finite_worlds)] = False
        fq[:, ~okf] = 0.0; ft[:, ~okf] = 0.0
        arrays["robot_finger_q"], arrays["robot_finger_target"] = fq, ft
        extra = {**extra, "robot_finger_names": np.array(cap["fingers"].names)}
    if a.actions > 0:
        acts = cap["actions"].arrays()
        okw = [w for w in range(min(acts.shape[1], W)) if w not in set(r.non_finite_worlds)]
        arrays["policy_action"] = acts[:, okw]
        extra = {**extra, "policy_action_worlds": np.array(okw, dtype=np.int64)}
    bad = [k for k, v in arrays.items() if not np.isfinite(v).all()]
    if bad:
        print(f"非有限值：{bad}", file=sys.stderr)
        return 2
    criteria = dataclasses.asdict(c) if dataclasses.is_dataclass(c) else dict(vars(c))
    np.savez_compressed(
        out + ".npz",
        **{k: (v if k in keep64 else v.astype(np.float32)) for k, v in arrays.items()},  # ADD 留 float64：重算全程 ADD-AUC 要与评测 JSON 对到 1e-6
        **extra,
        object_vertices_o=verts.astype(np.float32),
        tracking_error_names=np.array(r.tracking_error_names),
        ref_palm_names=np.array([names[i] for i in palm_idx]),
        contact_sides=np.array(sides), contact_active=contact,
        completion_truncated=np.asarray(r.completion.truncated, bool),
        completion_progress=np.asarray(r.completion.reference_progress, np.float64),
        non_finite_worlds=np.array(r.non_finite_worlds, dtype=np.int64),
        criteria_json=np.array(json.dumps(criteria)),
        reported_success_rate=np.float64(cap["tracking"].success_rate),
        control_fps=np.float64(r.control_fps), start_frame=np.int64(s0), step_count=np.int64(T), world_count=np.int64(W),
        policy=np.array(str(a.policy)), reference_parquet=np.array(parquet), wrapper_calls=np.array(json.dumps(calls)),
        upstream_md5=np.array(json.dumps({Path(p).name: md5(p) for p in
                                          (harness.__file__, rollout.__file__, metrics.__file__, EVAL_SCRIPT)})),
        mesh_vertex_count=np.int64(len(mesh)),
        trimesh_matches_evaluator=np.bool_(trimesh_matches),
        palm_dist_vertex_source=np.array("trimesh_all_vertices" if trimesh_matches else "evaluator_object_vertices_o"),
    )
    sync_note = f"；手掌 {extra['robot_palm_sides'].tolist()}，物体位置同步最大差 {float(extra['palm_object_sync_max_abs_m']):.3g} m" if a.palms else ""
    print(f"rollout: {out}.npz（T={T} W={W}，包装调用 {calls}；trimesh 顶点与评测器一致 {trimesh_matches}{sync_note}）  report: {out}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
