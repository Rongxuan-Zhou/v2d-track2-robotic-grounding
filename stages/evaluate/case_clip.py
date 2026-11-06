"""SOP“成功与失败案例”的策略 rollout 视频：在同一次评测 rollout 里录下全部世界的刚体位姿、按登记规则挑世界，再把录下的位姿渲染成视频（容器里执行）。

计划 agent-doc/plan-site-cases-video-2026-10-08.md 第 2 版。评测时所有世界初始状态相同，差别来自 GPU 计算的不确定性，某个编号的世界重跑不一定重现同样的结局，
所以要录下当次的状态再渲染。两种模式都原样跑官方 /workspace/scripts/evaluate_policy.py（runpy），只替换 harness 里的函数，官方代码不改：

  record：替换 harness.record_object_pose_rollout，在 rollout 的每一步之后把全部世界的 body_q 抄进内存（与 stages/evaluate/diag_palms.py 同一做法）；
          替换 harness.score_reference_tracking，拿逐世界误差按第 4 节的规则分类、挑世界，过三道门（物体位置与评测器记录逐步一致、无非有限、
          骨盆无单步跳变或在跳变前截断），只把选中世界的逐帧位姿与元数据存成 npz（V2D_CASE_OUT），然后照常打分。类别由 V2D_CASE_KINDS 给。
  render：world_count=1。替换 harness.record_object_pose_rollout：先核刚体数、刚体名与重置位姿，再渲右栏（参考，env.reset(frame_id=f)，每条 run 一次）
          与每个类别的左栏（把录下的位姿写进 state_0.body_q），同一条相机轨迹，按顺序录进同一段 panels.mp4（一个进程只能用一个 MP4 viewer），
          帧号范围写进 render.json；渲完直接结束进程（hydra 会接住异常）。V2D_CASE_IN、V2D_CASE_PICKS（逗号分隔）、
          V2D_CASE_DIR、V2D_CASE_PARQUET（容器内路径，算相机）。之后由 case_compose.py 拼成两栏成片。
Tier 2 的 run（V2D_CASE_RUN 以 t2- 开头）在两种模式里都先调 forbid_tier1()。
"""

import json
import math
import os
import runpy
import sys
from pathlib import Path

import numpy as np

SCRIPT = Path("/workspace/scripts/evaluate_policy.py")
HERE = Path(__file__).resolve().parent
K_MAIN = 25
SYNC_TOL_M = 1e-6
RESET_TOL_M = 1e-3
JUMP_M = 0.3


def pick_worlds(success, t_obj, t_pel, pos_m, kinds, k=K_MAIN):
    """按登记规则挑世界（逐世界数组；t_* 为 −1 表示没超）。返回 {类别: (世界, 说明, 这一类的中位数)}；空类返回 (None, 原因, nan)。平手取编号最小的。"""
    failed = ~success
    fallen = failed & (t_pel >= 0)
    obj_first = fallen & (t_obj >= 0) & (t_obj <= t_pel - k)
    out = {}
    for kind in kinds:
        if kind == "success":
            mask, key = success, pos_m
        elif kind == "fall":
            mask, key = obj_first, (t_pel - t_obj).astype(float)
        elif kind == "nearmiss":
            mask, key = failed & (t_pel < 0), pos_m
        else:
            raise ValueError(f"未知类别 {kind}")
        idx = np.flatnonzero(mask)
        if not len(idx):
            out[kind] = (None, f"这次 rollout 里没有 {kind} 类的世界", float("nan"))
            continue
        med = float(np.median(key[idx]))
        w = int(idx[np.argmin(np.abs(key[idx] - med))])
        out[kind] = (w, f"{len(idx)} 个 {kind} 世界里取最接近中位数 {med:.4g} 的", med)
    return out


def _guard_tier():
    if os.environ.get("V2D_CASE_RUN", "").startswith("t2-"):
        sys.path.insert(0, "/repo/libs")
        from v2d_infra.tier_guard import forbid_tier1
        forbid_tier1()
        print("case_clip: Tier 2 run，已调 forbid_tier1()", flush=True)


def _prepare_runpy():
    here = str(HERE)
    sys.path[:] = [p for p in sys.path if p != here]
    sys.path.insert(0, str(SCRIPT.parent))


def _world_index(model, world_count):
    bw = model.body_world.numpy() if getattr(model, "body_world", None) is not None else None
    n = model.body_count
    if bw is None:
        if n % world_count:
            raise RuntimeError(f"没有 body_world，刚体数 {n} 不能被世界数 {world_count} 整除")
        per = n // world_count
        return [np.arange(w * per, (w + 1) * per) for w in range(world_count)]
    return [np.flatnonzero(bw == w) for w in range(world_count)]


def _short(name):
    return str(name).rsplit("/", 1)[-1]


def record():
    out = os.environ["V2D_CASE_OUT"]
    kinds = [k for k in os.environ.get("V2D_CASE_KINDS", "success,fall,nearmiss").split(",") if k]
    _guard_tier()
    _prepare_runpy()
    sys.path.insert(1, str(HERE))
    from fall_order import crossings
    from flash_chord.evaluation import harness, metrics

    box = {"frames": [], "init": None, "env": None}
    orig_record = harness.record_object_pose_rollout
    orig_score = harness.score_reference_tracking

    def rec(env, jax_env, *a, **k):
        box["env"] = env
        box["init"] = env.state_0.body_q.numpy().copy()
        box["ids"] = np.asarray(env.command.body_ids_w.numpy())
        cls = type(jax_env)
        orig_step = cls.step

        def step(self, action):
            o = orig_step(self, action)
            box["frames"].append(env.state_0.body_q.numpy().copy())
            return o

        cls.step = step
        try:
            return orig_record(env, jax_env, *a, **k)
        finally:
            cls.step = orig_step

    def score(rollout, criteria):
        env = box["env"]
        W = env.world_count
        frames = np.stack(box["frames"])  # (T, 全部刚体, 7)
        T = rollout.tracking_error.shape[0]
        if frames.shape[0] != T:
            raise RuntimeError(f"录到 {frames.shape[0]} 步，rollout 是 {T} 步")
        success, pos, ori = metrics.maniptrans_success(rollout.achieved_pose_w, rollout.reference_pose_w, rollout.body_object_ids)
        z = {"tracking_error": rollout.tracking_error, "thresholds": np.asarray(criteria.thresholds), "active_after_step": np.asarray(criteria.active_after_step)}
        _, t_obj, t_pel, _, _ = crossings(z)
        valid = np.array([w not in set(rollout.non_finite_worlds) for w in range(W)])
        picks = pick_worlds(success & valid, np.where(valid, t_obj, -1), np.where(valid, t_pel, -1), pos, kinds)
        idx = _world_index(env.scene.model, W)
        names = list(getattr(env.scene.model, "body_key", []) or [])
        ids = box["ids"].reshape(W, -1)
        save = {"world_count": np.int64(W), "step_count": np.int64(T), "mp_sr": np.float64(success[valid].mean()),
                "run": np.str_(os.environ.get("V2D_CASE_RUN", "")), "kinds": np.asarray(kinds),
                "thresholds_mp": np.asarray([0.03, 30.0])}
        for kind, (w, note, med) in picks.items():
            save[f"{kind}_note"] = np.str_(note)
            save[f"{kind}_median"] = np.float64(med)
            if w is None:
                print(f"case_clip: {kind} → {note}", flush=True)
                continue
            fw = frames[:, idx[w]]
            # 门 1：从录下的 body_q 取出的物体位置，与评测器自己记录的 achieved_pose_w 逐步一致（证明世界切片与步对齐）
            g = ids[w] if ids[w].max() >= len(idx[w]) else idx[w][ids[w]]
            sync = float(np.max(np.abs(frames[:, g, :3].astype(np.float64) - rollout.achieved_pose_w[:, w, :, :3])))
            if not sync <= SYNC_TOL_M:
                raise RuntimeError(f"{kind} 世界 {w}：物体位置与评测器记录最大差 {sync:.3g} m > {SYNC_TOL_M}")
            if not np.isfinite(fw).all():
                raise RuntimeError(f"{kind} 世界 {w}：录下的位姿含非有限值")
            # 门 3：骨盆（世界内第 0 个刚体）单步位移超过 0.3 m 视为跳变，在跳变前截断
            disp = np.linalg.norm(np.diff(fw[:, 0, :3], axis=0), axis=-1)
            jumps = np.flatnonzero(disp > JUMP_M)
            cut = int(jumps[0]) + 1 if len(jumps) else T
            save[f"{kind}_world"] = np.int64(w)
            save[f"{kind}_init"] = box["init"][idx[w]].astype(np.float32)
            save[f"{kind}_frames"] = fw[:cut].astype(np.float32)
            save[f"{kind}_names"] = np.asarray([_short(names[i]) for i in idx[w]]) if names else np.asarray([])
            save[f"{kind}_stats"] = np.asarray([float(pos[w]), float(ori[w]), float(success[w]), float(t_obj[w]), float(t_pel[w]),
                                                sync, float(disp.max()), float(cut)])
            print(f"case_clip: {kind} → 世界 {w}（{note}）；MP 位置误差 {pos[w] * 100:.2f} cm、朝向 {ori[w]:.1f}°、成功 {bool(success[w])}、"
                  f"t_obj {t_obj[w]}、t_pel {t_pel[w]}；同步差 {sync:.2g} m；骨盆最大单步位移 {disp.max():.3f} m；截到 {cut}/{T} 步", flush=True)
        np.savez_compressed(out, **save)
        print(f"case_clip: rollout MP-SR {success[valid].mean():.4f}（{W} 世界）→ {out}", flush=True)
        return orig_score(rollout, criteria)

    harness.record_object_pose_rollout = rec
    harness.score_reference_tracking = score
    sys.argv = [str(SCRIPT), *sys.argv[1:]]
    runpy.run_path(str(SCRIPT), run_name="__main__")


def _camera(viewer, pos, target):
    """与 stages/policy/site_ref_replay.py 的 _apply_camera 相同：由相机位置与目标点算 pitch、yaw。"""
    import warp as wp
    dx, dy, dz = (b - a for a, b in zip(pos, target))
    r = math.sqrt(dx * dx + dy * dy + dz * dz)
    ux, uy, uz = dx / r, dy / r, dz / r
    up = viewer.camera.up_axis
    if up == 2:
        pitch, yaw = math.degrees(math.asin(uz)), math.degrees(math.atan2(uy, ux))
    elif up == 1:
        pitch, yaw = math.degrees(math.asin(uy)), math.degrees(math.atan2(uz, ux))
    else:
        pitch, yaw = math.degrees(math.asin(ux)), math.degrees(math.atan2(uz, uy))
    viewer.set_camera(wp.vec3(*pos), pitch, yaw)


def render():
    src = np.load(os.environ["V2D_CASE_IN"], allow_pickle=False)
    picks = [k for k in os.environ["V2D_CASE_PICKS"].split(",") if k]
    outdir = Path(os.environ["V2D_CASE_DIR"])
    parquet = os.environ["V2D_CASE_PARQUET"]
    for kind in picks:
        if f"{kind}_frames" not in src:
            raise SystemExit(f"case_clip: npz 里没有 {kind}（{src[f'{kind}_note']}）")
    _guard_tier()
    _prepare_runpy()
    sys.path.insert(1, "/repo/stages/policy")
    from site_ref_replay import _camera_track
    from flash_chord.evaluation import harness
    from flash_chord.visualization.viewer import ViewerConfig, create_viewer

    report = {"picks": picks, "gates": {}}

    def hook(env, jax_env, *a, **k):
        model, state = env.scene.model, env.state_0
        idx = _world_index(model, 1)[0]
        names = [_short(n) for n in (getattr(model, "body_key", []) or [])]
        q0 = state.body_q.numpy()
        for kind in picks:
            fr, init, rec_names = src[f"{kind}_frames"], src[f"{kind}_init"], [str(n) for n in src[f"{kind}_names"]]
            # 门 2：刚体数、刚体名逐个相同；1 世界的重置位姿与录下的重置位姿相差 ≤ 1 mm
            if len(idx) != fr.shape[1]:
                raise RuntimeError(f"{kind}：1 世界刚体数 {len(idx)} 与录下的 {fr.shape[1]} 不同")
            if rec_names and names and [names[i] for i in idx] != rec_names:
                raise RuntimeError(f"{kind}：刚体名对不上")
            gap = float(np.max(np.linalg.norm(q0[idx, :3] - init[:, :3], axis=-1)))
            if not gap <= RESET_TOL_M:
                raise RuntimeError(f"{kind}：重置位姿最大差 {gap:.4f} m > {RESET_TOL_M}")
            report["gates"][kind] = {"bodies": int(len(idx)), "names_checked": bool(rec_names and names), "reset_gap_m": gap}
        T = max(int(src[f"{k}_frames"].shape[0]) for k in picks)
        track, _, _ = _camera_track(parquet, int(src["step_count"]))
        # 一个进程只能用一个 MP4 viewer（官方 MP4Viewer 在底层 GL viewer 关掉后不再写帧）：各左栏与参考栏按顺序录进同一段视频，记下帧号范围
        segs = [(kind, int(src[f"{kind}_frames"].shape[0])) for kind in picks] + [("right", T)]
        total = sum(n for _, n in segs)
        cfg = ViewerConfig(backend="mp4", output_path=str(outdir / "panels.mp4"), num_frames=total, headless=True,
                           realtime=False, width=640, height=480, video_fps=30.0)
        viewer = create_viewer(cfg)
        viewer.set_model(model)

        def rollout_state(fr, f):
            q = q0.copy()
            q[idx] = fr[f]
            state.body_q.assign(q)
            return state

        def reference_state(f):
            env.reset(frame_id=f)
            return env.state_0

        start = 0
        for name, n in segs:
            fr = src[f"{name}_frames"] if name != "right" else None
            finite = 0
            for f in range(n):
                st = reference_state(f) if fr is None else rollout_state(fr, f)
                finite += int(np.isfinite(st.body_q.numpy()).all())
                _camera(viewer, *track[min(f, len(track) - 1)])
                viewer.begin_frame((start + f) / 30.0)
                viewer.log_state(st)
                viewer.end_frame()
            report.setdefault("segments", {})[name] = [start, start + n]
            report[f"{name}_finite"] = finite
            start += n
        viewer.close()
        report["written_frames"] = int(viewer.frame_count)
        report["total_frames"] = total
        (outdir / "render.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
        print(f"case_clip: 渲染完成 {json.dumps(report, ensure_ascii=False)}", flush=True)
        sys.stdout.flush()
        os._exit(0 if viewer.frame_count == total else 3)  # hydra 会接住异常并以 1 退出，所以这里直接结束进程

    harness.record_object_pose_rollout = hook
    sys.argv = [str(SCRIPT), *sys.argv[1:]]
    runpy.run_path(str(SCRIPT), run_name="__main__")
    raise SystemExit("case_clip: 评测没有走到 rollout，什么也没渲染")


if __name__ == "__main__":
    mode = sys.argv.pop(1)
    {"record": record, "render": render}[mode]()
