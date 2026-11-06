"""在 8c778f9 评测器里原样跑 scripts/evaluate_policy.py，同时把逐步、逐世界的跟踪误差存成 npz（容器里执行）。

官方评测在内存里记录 rollout.tracking_error（步, 世界, 6：骨盆位置、骨盆朝向、手掌位置、手掌朝向、物体位置、物体朝向），只用来事后判失败原因、不落盘。
这里替换 flash_chord.evaluation.harness.score_reference_tracking：先把 rollout 里要的数组存下来，再调原函数，评测本身一处不改。
计划 agent-doc/plan-t1-fall-order-2026-10-07.md 第 3 节。

用法（容器里，/workspace 是 8c778f9 副本）：
  V2D_DUMP_OUT=/repo/runs/.../errors.npz python /repo/stages/evaluate/eval_dump_errors.py evaluation.checkpoint=... evaluation.world_count=4096 ...
其余参数原样交给 evaluate_policy.py（与 stages/policy/fc_eval.sh 相同）。
"""

import hashlib
import os
import runpy
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path("/workspace/scripts/evaluate_policy.py")


CODE = ("/workspace/scripts/evaluate_policy.py", "/workspace/src/flash_chord/evaluation/harness.py",
        "/workspace/src/flash_chord/evaluation/rollout.py", "/workspace/src/flash_chord/evaluation/metrics.py")


def dump(path, rollout, criteria):
    from flash_chord.evaluation import metrics

    success, pos, ori_deg = metrics.maniptrans_success(rollout.achieved_pose_w, rollout.reference_pose_w, rollout.body_object_ids)
    sp_ok, sp_pos, sp_ori = metrics.spider_success(rollout.achieved_pose_w, rollout.reference_pose_w)
    np.savez_compressed(
        path,
        tracking_error=np.asarray(rollout.tracking_error, dtype=np.float16),
        error_names=np.asarray(rollout.tracking_error_names),
        criteria_error_names=np.asarray(criteria.error_names),
        thresholds=np.asarray(criteria.thresholds, dtype=np.float64),
        active_after_step=np.asarray(criteria.active_after_step, dtype=np.int64),
        completion_step=rollout.completion.completion_step,
        terminated=rollout.completion.terminated,
        truncated=rollout.completion.truncated,
        reference_progress=rollout.completion.reference_progress,
        non_finite_worlds=np.asarray(rollout.non_finite_worlds, dtype=np.int64),
        mp_success=success,
        mp_position_m=pos,
        mp_orientation_deg=ori_deg,
        sp_success=sp_ok,
        sp_position_m=sp_pos,
        sp_orientation_rad=sp_ori,
        control_fps=np.float64(rollout.control_fps),
        code_md5=np.asarray([f"{hashlib.md5(Path(f).read_bytes()).hexdigest()}  {f}" for f in CODE]),
    )
    print(f"eval_dump_errors: wrote {path} tracking_error {tuple(np.shape(rollout.tracking_error))}", flush=True)


def main():
    out = os.environ.get("V2D_DUMP_OUT")
    if not out:
        sys.exit("eval_dump_errors: V2D_DUMP_OUT 未设")
    # 与官方调用一致：sys.path[0] 是脚本目录，去掉本包装所在的目录
    here = str(Path(__file__).resolve().parent)
    sys.path[:] = [p for p in sys.path if p != here]
    sys.path.insert(0, str(SCRIPT.parent))
    from flash_chord.evaluation import harness

    original = harness.score_reference_tracking
    calls = []

    def patched(rollout, criteria):
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError("score_reference_tracking 被调用了不止一次，npz 会被覆盖")
        dump(out, rollout, criteria)
        return original(rollout, criteria)

    harness.score_reference_tracking = patched
    sys.argv = [str(SCRIPT), *sys.argv[1:]]
    runpy.run_path(str(SCRIPT), run_name="__main__")
    if len(calls) != 1:
        sys.exit(f"eval_dump_errors: 钩子被调用 {len(calls)} 次（应为 1）")


if __name__ == "__main__":
    main()
