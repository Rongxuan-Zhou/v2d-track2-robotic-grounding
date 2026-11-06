"""按指定的回放方式评测一个 checkpoint（在 FlashCHORD 容器里跑，占 GPU）；Day 2 的交叉评测用。

官方评测器（harness.evaluate_checkpoint）从 checkpoint 元数据里的配置重建参考，回放方式跟训练时一样。这里照 diag_rollout.py 的做法，
在 flash_chord.evaluation.harness 里替换 checkpoint_config（harness 在模块顶层导入它、在 evaluate_checkpoint 里按名字调用），
只改 task.source_frame_playback；改成真实时长时补上 G1 的两个 palm 校验容差键（逐帧训练的 checkpoint 里没有）；
reference.num_frames 按 parquet 自己的帧数与 fps 重算（评测路径不校验它，这里用它核对：评测报告的 step_count 必须等于它）。
其余一律不动，然后用 runpy 运行未改动的 scripts/evaluate_policy.py，评测 JSON 由上游脚本自己写。
替换函数必须恰好被调用 1 次；另写 <输出>.override.json：改了哪些键、改前改后的值、调用次数、step_count 核对结果。

用法（容器内路径）：python /repo/stages/evaluate/eval_playback_override.py <policy.safetensors> <输出 JSON>
        --playback {frame,real,same} [--worlds 4096] [--palm-position-tol 0.002] [--palm-orientation-tol 0.005]
        [--parquet /repo/runs/motion/…/data.parquet] [--extra evaluation.reset_mode=explicit …]
  --parquet：换参考（k-fix，计划 agent-doc/plan-t3-kfix-2026-10-16.md 方案 4），task.parquet 与 reference.parquet 一起换，num_frames 按新 parquet 重算；
    评测 JSON 里的 parquet 必须正是它，否则作废。--extra：透传 Hydra 覆盖（只许 evaluation.* 与 logging.*），导出轨迹时用。
  --playback same：恒等覆盖（一个键都不改），用来核对包装本身不改变结果。
  输出 JSON 不要叫 eval_<步数>_w<世界数>.json（fc_eval.sh 见到同名文件会当成匹配回放的结果直接打印）。
"""
import argparse
import copy
import json
import runpy
import sys
from pathlib import Path

EVAL_SCRIPT = "/workspace/scripts/evaluate_policy.py"
TOL_KEYS = ("frame_position_tolerance", "frame_orientation_tolerance")


def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flat(v, key + "."))
        else:
            out[key] = v
    return out


def override_config(cfg: dict, playback: str, pos_tol: float, ori_tol: float, src_frames: int, src_fps: float, parquet: str = None):
    """返回（改过的配置副本，{键: [改前, 改后]}）。纯函数，不依赖 flash_chord，Mac 上可测。
    parquet 给了时（k-fix 用现有 checkpoint 对新参考评测与导出，计划 agent-doc/plan-t3-kfix-2026-10-16.md 方案 4）：task.parquet 与
    reference.parquet 一起换成它（task.parquet 保留原来的引号写法），reference.num_frames 按换后的回放方式与新 parquet 的帧数（src_frames、src_fps
    由调用方从新 parquet 读）重算。新路径必须是容器里的 /repo/runs/motion/… 绝对路径。"""
    if playback not in ("frame", "real", "same"):
        raise ValueError(f"playback 只能是 frame / real / same，收到 {playback!r}")
    out = copy.deepcopy(cfg)
    if parquet is not None:
        if not (str(parquet).startswith("/repo/runs/motion/") and str(parquet).endswith("/data.parquet")):
            raise ValueError(f"新参考要是 /repo/runs/motion/…/data.parquet，收到 {parquet!r}")
        old_task = str(out["task"]["parquet"])
        q = old_task[0] if old_task[:1] in ("'", '"') and old_task[-1:] == old_task[:1] else ""
        out["task"]["parquet"] = f"{q}{parquet}{q}"
        out["reference"]["parquet"] = parquet
    if playback != "same":
        if float(out["task"].get("motion_speed", 1.0)) != 1.0:
            raise ValueError(f"motion_speed 是 {out['task']['motion_speed']}，本工具只处理 1.0（逐帧回放要求 1.0）")
        playback_fps = float(out["reference"]["playback_fps"])
        if playback == "frame":
            out["task"]["source_frame_playback"] = True
            out["reference"]["num_frames"] = int(src_frames)
        else:
            out["task"]["source_frame_playback"] = False
            out["reference"]["num_frames"] = int(round((src_frames - 1) / src_fps * playback_fps)) + 1
            emb = out["embodiment"]["config"]
            for key, tol in zip(TOL_KEYS, (pos_tol, ori_tol)):
                if float(emb.get(key, 0.0)) < tol:     # 已有更宽的（例如本来就是真实时长训的）就不动
                    emb[key] = tol
    if parquet is not None:   # 换了参考：帧数按换后的回放方式与新 parquet 重算（同一集通常不变，变了也要跟上）
        if out["task"].get("source_frame_playback", True):
            out["reference"]["num_frames"] = int(src_frames)
        else:
            out["reference"]["num_frames"] = int(round((src_frames - 1) / src_fps * float(out["reference"]["playback_fps"]))) + 1
    before, after = flat(cfg), flat(out)
    changes = {k: [before.get(k), after.get(k)] for k in sorted(set(before) | set(after)) if before.get(k) != after.get(k)}
    return out, changes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("policy")
    ap.add_argument("out_json")
    ap.add_argument("--playback", required=True, choices=("frame", "real", "same"))
    ap.add_argument("--worlds", type=int, default=4096)
    ap.add_argument("--palm-position-tol", type=float, default=0.002)
    ap.add_argument("--palm-orientation-tol", type=float, default=0.005)
    ap.add_argument("--parquet", default=None, help="换参考：容器里的 /repo/runs/motion/…/data.parquet（k-fix 用现有 checkpoint 对新参考评测与导出）")
    ap.add_argument("--extra", action="append", default=[],
                    help="透传给 evaluate_policy.py 的 Hydra 覆盖，只许 evaluation.* 与 logging.*（导出要 reset_mode=explicit 与轨迹输出）")
    a = ap.parse_args()
    for x in a.extra:
        if not (x.startswith("evaluation.") or x.startswith("logging.")) or "=" not in x:
            ap.error(f"--extra 只许 evaluation.*=… 或 logging.*=…，收到 {x!r}")
    out = Path(a.out_json)
    side = Path(str(out) + ".override.json")
    for p in (out, side):
        p.unlink(missing_ok=True)  # 旧输出先删，失败时不会误读上一次的

    import pyarrow.parquet as pq
    from flash_chord.evaluation import harness

    orig = harness.checkpoint_config
    state = {"calls": 0}

    def patched(metadata):
        state["calls"] += 1
        cfg = orig(metadata)
        parquet = a.parquet or str(cfg["task"]["parquet"]).strip("'")
        row = pq.read_table(parquet, columns=["fps", "robot_joint_positions"]).to_pylist()[0]
        new, changes = override_config(cfg, a.playback, a.palm_position_tol, a.palm_orientation_tol,
                                       len(row["robot_joint_positions"]), float(row["fps"]), parquet=a.parquet)
        state.update(changes=changes, parquet=parquet, src_frames=len(row["robot_joint_positions"]), src_fps=float(row["fps"]),
                     num_frames=new["reference"]["num_frames"])
        return new

    harness.checkpoint_config = patched
    sys.argv = [EVAL_SCRIPT, f"evaluation.checkpoint={a.policy}", f"evaluation.world_count={a.worlds}",
                f"evaluation.metrics_output={out}", f"hydra.run.dir={out}_hydra", *a.extra]
    sys.path.insert(0, str(Path(EVAL_SCRIPT).parent))  # 与直接运行该脚本时的 sys.path 一致
    try:
        runpy.run_path(EVAL_SCRIPT, run_name="__main__")
    except SystemExit as e:  # 上游若以 sys.exit(...) 结束：成功就继续核对，失败原样抛出
        if e.code not in (0, None):
            raise
    report = json.loads(out.read_text()) if out.exists() else {}
    step_ok = report.get("step_count") == state.get("num_frames")
    if a.parquet and report.get("parquet") != a.parquet:   # 换参考时，评测器写进 JSON 的 parquet 必须正是新参考
        print(f"评测 JSON 的 parquet 是 {report.get('parquet')}，要换成的是 {a.parquet}：换参考没有生效，结果作废", file=sys.stderr)
        return 2
    side.write_text(json.dumps({"policy": a.policy, "playback": a.playback, "worlds": a.worlds, "calls": state["calls"],
                                "changes": state.get("changes"), "parquet": state.get("parquet"), "src_frames": state.get("src_frames"),
                                "src_fps": state.get("src_fps"), "expected_step_count": state.get("num_frames"),
                                "step_count": report.get("step_count"), "step_count_ok": step_ok}, ensure_ascii=False, indent=1))
    print(f"eval_playback_override: playback={a.playback}，替换函数调用 {state['calls']} 次，改动 {state.get('changes')}，"
          f"step_count {report.get('step_count')}（应为 {state.get('num_frames')}）")
    if state["calls"] != 1 or not step_ok:
        print("调用次数不是 1 或 step_count 对不上：上游可能改了 evaluate_checkpoint 的调用方式，结果作废", file=sys.stderr)
        return 2
    m = report["metrics"]
    print(f"  add_auc {m['add_auc']:.4f}  chord_sr {m['chord_sr']:.4f}  maniptrans_sr {m['maniptrans_sr']:.4f}  "
          f"spider_sr_uncentered {m['spider_sr_uncentered']:.4f}  mppe_cm {m['mppe_cm']:.3f}")
    print("  summary_json " + json.dumps({"add_auc": m["add_auc"], "chord_sr": m["chord_sr"], "maniptrans_sr": m["maniptrans_sr"],
                                          "spider_sr_uncentered": m["spider_sr_uncentered"], "mppe_cm": m["mppe_cm"],
                                          "world_count": report.get("world_count"), "step_count": report.get("step_count"),
                                          "checkpoint_sha256": report.get("checkpoint_sha256"), "playback": a.playback,
                                          "parquet": state.get("parquet")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
