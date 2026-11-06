"""发散取证用的 rollout：按官方评测路径跑一次，逐步抄下物体与两只手掌的世界坐标，原样存下（非有限值不清零）。
在 FlashCHORD 容器里跑，占 GPU；只读仿真状态，不写。计划 agent-doc/plan-ep7-eval-and-guards-2026-10-02.md 的 A 部分。

为什么另写一个：diag_rollout.py --palms 存盘前把非有限世界的手掌清零，评测器也把这些世界的物体位姿清零（rollout.py:279），
崩之前那几帧的轨迹就看不到了；评分阶段若因有世界被判终止而报错（harness.py:525–526），diag_rollout.py 不存盘。本脚本只取证、不评分：
包一层 harness.record_object_pose_rollout（把传进来的 jax_env 换成 diag_palms.PalmRecorder 的代理，做法同 diag_rollout.py）与
harness.score_reference_tracking（只记下它是否报错），用 runpy 运行未改动的 scripts/evaluate_policy.py。

用法：python /repo/stages/evaluate/blowup_rollout.py <policy.safetensors> <输出前缀> --worlds 4096   （容器内路径）
输出 <前缀>.npz：
  object_pos (T,W,B,3)、palm_pos (T,W,H,3)：float32 原始值；palm_sides
  first_bad_frame (W,)：这个世界的物体或手掌第一次出现非有限值、或绝对值超过 FAR_M 的帧，没有就是 -1
  evaluator_non_finite_worlds、terminated (W,)、reference_pose_w (T,B,7)、tracking_error 与名字（评测器自己记的，原样）
  score_error：评分阶段的报错文字（没有报错为空串）
  upstream_md5：上游 harness.py、rollout.py、metrics.py 与 evaluate_policy.py 的 md5（同 diag_rollout.py 的做法）
最后打印一行 BLOWUP_ROLLOUT_OK 与汇总；评分报 “must disable task failure termination” 不算失败（取证不靠评分），其他异常原样抛出。
输出的 npz 或 json 已存在就拒绝（退出码 2）：不覆盖上一次的证据。<前缀>.json 是上游评测脚本自己写的报告，评分报错时没有；
它的分数不用于候选条件与质量的结论（那两项以 8c778f9 副本的 fc_eval 为准），这里的世界号也不与那次评测一一对应。
V2D_BLOWUP_EVAL_SCRIPT：只给测试用，换掉要运行的评测脚本路径（stages/evaluate/test_blowup_rollout.py）。
"""
import argparse
import hashlib
import json
import os
import runpy
import sys
from pathlib import Path

import numpy as np

EVAL_SCRIPT = os.environ.get("V2D_BLOWUP_EVAL_SCRIPT", "/workspace/scripts/evaluate_policy.py")
FAR_M = 100.0  # 位置绝对值超过它也算崩了：数值很大但还有限的状态同样会毁掉训练
TERMINATION_ERROR = "must disable task failure termination"


def first_bad_frame(*arrays: np.ndarray) -> np.ndarray:
    """各数组形状 (T,W,…)。返回 (W,)：第一次出现非有限或超远值的帧，没有为 -1。"""
    bad = None
    for a in arrays:
        flat = a.reshape(a.shape[0], a.shape[1], -1)
        with np.errstate(invalid="ignore"):
            b = (~np.isfinite(flat)).any(-1) | (np.abs(flat) > FAR_M).any(-1)
        bad = b if bad is None else (bad | b)
    first = bad.argmax(0)
    first[~bad.any(0)] = -1
    return first.astype(np.int64)


def upstream_md5(harness) -> dict:
    """上游评测代码四个文件的 md5；取不到的（例如测试里的假模块）记空串。"""
    out = {}
    names = {"harness.py": getattr(harness, "__file__", None), "evaluate_policy.py": EVAL_SCRIPT}
    for mod in ("rollout", "metrics"):
        try:
            m = __import__(f"flash_chord.evaluation.{mod}", fromlist=[mod])
            names[f"{mod}.py"] = getattr(m, "__file__", None)
        except Exception:  # noqa: BLE001  测试里没有这两个模块
            names[f"{mod}.py"] = None
    for name, path in names.items():
        try:
            out[name] = hashlib.md5(Path(path).read_bytes()).hexdigest()
        except (TypeError, OSError):
            out[name] = ""
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("policy")
    ap.add_argument("out_prefix")
    ap.add_argument("--worlds", type=int, default=4096)
    a = ap.parse_args()
    out = a.out_prefix
    for suffix in (".npz", ".json"):
        if Path(out + suffix).exists():
            print(f"已存在 {out + suffix}：不覆盖上一次的证据，确认后移走再跑", file=sys.stderr)
            return 2

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from flash_chord.evaluation import harness

    calls = {"record": 0, "score": 0}
    cap = {"score_error": ""}
    record, score = harness.record_object_pose_rollout, harness.score_reference_tracking

    def record_wrapper(*args, **kwargs):
        calls["record"] += 1
        import diag_palms  # 环境建好之后才导入，同 diag_rollout.py
        args = list(args)
        env = args[0] if args else kwargs["env"]
        cap["rec"] = diag_palms.PalmRecorder(env, kwargs["step_count"])
        if len(args) > 1:
            args[1] = cap["rec"].wrap(args[1])
        else:
            kwargs["jax_env"] = cap["rec"].wrap(kwargs["jax_env"])
        cap["rollout"] = record(*args, **kwargs)
        return cap["rollout"]

    def score_wrapper(*args, **kwargs):
        calls["score"] += 1
        try:
            return score(*args, **kwargs)
        except ValueError as e:  # 只认“有世界被判终止”这一种：记下原文再照常抛出，由下面决定放不放过
            if TERMINATION_ERROR in str(e):
                cap["score_error"] = str(e)
            raise

    harness.record_object_pose_rollout = record_wrapper
    harness.score_reference_tracking = score_wrapper
    sys.argv = [EVAL_SCRIPT, f"evaluation.checkpoint={a.policy}", f"evaluation.world_count={a.worlds}",
                f"evaluation.metrics_output={out}.json", f"hydra.run.dir={out}_hydra"]
    sys.path.insert(0, str(Path(EVAL_SCRIPT).parent))  # 与直接运行该脚本时的 sys.path 一致
    try:
        runpy.run_path(EVAL_SCRIPT, run_name="__main__")
    except SystemExit as e:  # hydra 会把任务里的异常打印出来再 sys.exit(1)；只有评分阶段那一种报错才放过
        if e.code not in (0, None) and not cap["score_error"]:
            raise
    except ValueError:
        if not cap["score_error"]:
            raise

    if calls["record"] != 1 or "rollout" not in cap:
        print(f"包装调用次数不对：{calls}；上游可能改了 evaluate_checkpoint 的调用方式", file=sys.stderr)
        return 2
    r = cap["rollout"]
    palms, objects = cap["rec"].arrays()
    first = first_bad_frame(objects, palms)
    bad_worlds = np.flatnonzero(first >= 0)
    terminated = np.asarray(r.completion.terminated, bool)
    evaluator_nf = np.array(r.non_finite_worlds, dtype=np.int64)
    np.savez_compressed(
        out + ".npz",
        object_pos=objects.astype(np.float32), palm_pos=palms.astype(np.float32), palm_sides=np.array(cap["rec"].sides),
        first_bad_frame=first, evaluator_non_finite_worlds=evaluator_nf, terminated=terminated,
        reference_pose_w=np.asarray(r.reference_pose_w, np.float64),
        tracking_error=np.asarray(r.tracking_error, np.float32), tracking_error_names=np.array(r.tracking_error_names),
        score_error=np.array(cap["score_error"]), far_m=np.float64(FAR_M),
        start_frame=np.int64(int(r.start_frame)), step_count=np.int64(objects.shape[0]), world_count=np.int64(objects.shape[1]),
        policy=np.array(str(a.policy)), wrapper_calls=np.array(json.dumps(calls)),
        upstream_md5=np.array(json.dumps(upstream_md5(harness))),
    )
    T, W = objects.shape[:2]
    print(f"rollout: {out}.npz（T={T} W={W}，包装调用 {calls}）")
    print(f"  崩掉的世界（物体或手掌非有限、或超过 {FAR_M:g} m）{len(bad_worlds)} 个；评测器记的非有限世界 {len(evaluator_nf)} 个；"
          f"被判终止的世界 {int(terminated.sum())} 个；评分报错：{cap['score_error'] or '无'}")
    if len(bad_worlds):
        frames = first[bad_worlds]
        values, counts = np.unique(frames, return_counts=True)
        order = np.argsort(-counts)[:8]
        top = "、".join(f"第 {int(values[i])} 帧 {int(counts[i])} 个" for i in order)
        print(f"  第一次出问题的帧：最早 {int(frames.min())}，中位 {int(np.median(frames))}，最晚 {int(frames.max())}；最集中的：{top}")
        print(f"  世界号（前 20 个）：{bad_worlds[:20].tolist()}")
    print(f"BLOWUP_ROLLOUT_OK bad_worlds={len(bad_worlds)} evaluator_non_finite={len(evaluator_nf)} terminated={int(terminated.sum())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
