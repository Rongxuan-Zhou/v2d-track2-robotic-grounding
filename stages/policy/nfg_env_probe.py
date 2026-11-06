"""非有限防护的 GPU 闸 T1、T1b、T2a–T2d、T2n 与 K1（守 docs/contracts/policy.md 的 POLICY-I001、POLICY-I002、POLICY-I003）。
计划 agent-doc/plan-ep7-eval-and-guards-2026-10-02.md 第 5 版的 B.4。训练机上、仓库根执行，一个闸一个进程：
  V2D_GPU=1 V2D_FC_SRC=<副本> scripts/dev/record ep7/nfg_probe_<闸>_<机器> scripts/dev/in-flash-chord "python /repo/stages/policy/nfg_env_probe.py <闸>"
t1、t1b 挂未打防护的 fc_src/v040_p185，其余挂打了防护的 fc_src/v040_p185_nfg；脚本按挂上的 termination.py 里有没有防护核对，挂错就退出。

照训练的做法建 ep7 的环境（Tier 2 参考；hydra 组合 train_flash_sac + experiment=g1_recon_body_flash_sac，64 个世界，reset.seed=42），
RLEnv → reset → capture_step → reset，之后只走录好的图（训练也是这样）。动作全给 0。先跑 100 步，第 100 步之后注入，再跑 120 步，
逐步记下每个世界的刚体位姿、关节量、观测、评论家上下文、终止观测、qacc_warmstart（与 act，若有）是否有限，done、truncation、9 列终止诊断、
episode_step、timestep，存到 /repo/runs/nfg_probe/<闸>.npz。注入都经 wp.launch 写（与仿真同一个流），位姿类注入写 joint_q，不写 body_q
（下一个子步只从 joint_q、joint_qd 推 MuJoCo 的 qpos、qvel）。

  t1   未打防护：世界 7 的关节速度全写 NaN。判据：世界 7 第一次被重置之后的下一步，仿真之后仍然非有限（看那一步的终止观测；说明重置救不回来）。
  t1b  未打防护：只把世界 9 的 qacc_warmstart 整行写 NaN，不动位置、速度。判据：下一步它的刚体位姿全部非有限。
  t2a  打了防护：同 t1 的注入；另做 K1：注入之前的 100 步，每步把 body_q、timestep、episode_step 与物体参考位姿拷一份，
       上游 v0.4.0 与打了防护的两版终止 kernel 各算一次，12 个输出逐位相同。
  t2b  打了防护：世界 7 的物体平移写 1e6（有限，超出 fp16）。
  t2c  打了防护：世界 7 的物体四元数写 NaN。
  t2d  打了防护：同一步坏 3 个世界（关节速度 NaN）：世界 3 在第 90 步后显式重置到第 0 帧（注入时在冻结期），世界 5 在第 96 步后显式重置到
       倒数第 5 帧（注入后的第一步正好是截断步），世界 7 照常。
  t2n  打了防护、但在本进程里关掉重置时清 warmstart（只在这个探针进程里替换 envs.base._solver_rows_to_clear，不是训练的开关）：同 t1 的注入。
       判据：世界 7 被判终止、重置后的下一步仿真之后仍然非有限（同 t1，看终止观测：防护会在同一步把它重置，步末状态总是有限的；与 t1 一起说明清 warmstart 是必要的）。
  t2a–t2d 的判据：每个被注入的世界在注入后的第一步 done 为 1，9 列诊断恰好是（骨盆 1、手掌 0、物体 0、骨盆位置误差 -1、其余 0），
  这一步之后它的 qacc_warmstart（与 act）有限；注入后 120 步里它的刚体位姿、关节量、观测、上下文全程有限（第一步之后的终止观测也有限）；
  其余世界 220 步全程有限。
最后一行 NFG_PROBE_DONE <闸> PASS|FAIL；判据不过时退出码 1，没跑完时报错退出。
"""
import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import warp as wp

CONFIG_DIR = "/workspace/src/flash_chord/configs"
UPSTREAM_TERMINATION = "/repo/third_party/video_to_data/robotic_grounding/flash_chord/src/flash_chord/lifecycle/termination.py"
PARQUET = ("/repo/runs/motion/tier2/whole_body/soma/sequence_id=2026-05-29_15-20-43_iron_round_table_02/"
           "robot_name=g1/data.parquet")
WORLDS = 64
PRE_STEPS = 100
POST_STEPS = 120
FAR_M = 1.0e3
GATES = ("t1", "t1b", "t2a", "t2b", "t2c", "t2d", "t2n")
PLAIN_GATES = ("t1", "t1b")
SENTINEL_ROW = [1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0]


@wp.kernel
def write_span(values: wp.array(dtype=wp.float32), start: int, value: float):
    i = wp.tid()
    values[start + i] = value


@wp.kernel
def write_row(values: wp.array2d(dtype=wp.float32), row: int, value: float):
    j = wp.tid()
    values[row, j] = value


def build_env():
    from flash_chord.configuration import instantiate_typed
    from flash_chord.embodiments.base import Embodiment
    from flash_chord.envs.rl import RLEnv, RLEnvConfig
    from flash_chord.scene.collision import CollisionPolicy
    from flash_chord.scene.setup import setup_scene
    from flash_chord.training.flash_sac.config import TrainingConfig
    from hydra import compose, initialize_config_dir

    overrides = ["experiment=g1_recon_body_flash_sac", f"task.parquet='{PARQUET}'", f"scene.world_count={WORLDS}", "reset.seed=42"]
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base="1.3"):
        cfg = compose(config_name="train_flash_sac", overrides=overrides)
    env_config = instantiate_typed(cfg.env, RLEnvConfig)
    embodiment = instantiate_typed(cfg.embodiment, Embodiment)
    collision = instantiate_typed(cfg.collision, CollisionPolicy)
    training = instantiate_typed(cfg.training, TrainingConfig)
    device = wp.get_device(training.device)
    with wp.ScopedDevice(device):
        setup = setup_scene(
            parquet=cfg.task.parquet,
            control_fps=env_config.sim.fps,
            motion_speed=cfg.task.motion_speed,
            embodiment=embodiment,
            collision=collision,
            world_count=cfg.scene.world_count,
            include_support=cfg.scene.support,
            decompose_objects=cfg.scene.decompose_objects,
            object_scale_min=cfg.scene.object_scale_min,
            object_scale_max=cfg.scene.object_scale_max,
            object_scale_seed=cfg.scene.object_scale_seed,
            contact_friction=cfg.scene.get("contact_friction", 1.0),
            object_free_joint_damping=cfg.scene.get("object_free_joint_damping", 0.0),
            source_frame_playback=bool(cfg.task.source_frame_playback),
            motion_start_frame=int(cfg.task.motion_start_frame),
            motion_end_frame=int(cfg.task.motion_end_frame),
        )
        env = RLEnv(setup.scene, setup.reference, config=env_config)
        env.reset()
        env.capture_step()
        env.reset()
    print(f"环境：{setup.parquet_path}；{WORLDS} 个世界，参考 {env.reference.num_frames} 帧，设备 {device}；"
          f"终止阈值 骨盆 {env.termination.config.pelvis_position_threshold} m / {env.termination.config.pelvis_orientation_threshold} rad，"
          f"冻结 {env.termination.config.reset_freeze_steps} 步；VOC {float(env.voc_scale.numpy()[0])}", flush=True)
    return env, device


def solver_rows(env):
    data = getattr(env.solver, "mjw_data", None)
    if data is None:
        raise RuntimeError("solver 没有 mjw_data，探针只认 MuJoCo-Warp 求解器")
    act = getattr(data, "act", None)
    if act is not None and (act.ndim != 2 or act.shape[1] == 0):
        act = None
    return data.qacc_warmstart, act


class Recorder:
    """逐步记下每个世界的有限性与终止输出。"""

    def __init__(self, env):
        self.env = env
        self.W = env.world_count
        self.B = env.model.body_count // self.W
        self.nq = env.model.joint_coord_count // self.W
        self.nv = env.model.joint_dof_count // self.W
        self.obs_dim = env.observation_dim
        self.ctx_dim = len(env.critic_context_names)
        self.warm, self.act = solver_rows(env)
        self.rows = {k: [] for k in ("body", "far", "joint_q", "joint_qd", "obs", "ctx", "term_obs", "warm", "act",
                                     "done", "trunc", "diag", "episode_step", "timestep")}

    def world_state(self):
        e = self.env
        bq = e.state_0.body_q.numpy().reshape(self.W, self.B, 7)
        body = np.isfinite(bq).all(axis=(1, 2))
        far = (np.abs(np.nan_to_num(bq[..., :3], nan=0.0, posinf=np.inf, neginf=-np.inf)) > FAR_M).any(axis=(1, 2))
        jq = np.isfinite(e.state_0.joint_q.numpy().reshape(self.W, self.nq)).all(axis=1)
        jqd = np.isfinite(e.state_0.joint_qd.numpy().reshape(self.W, self.nv)).all(axis=1)
        return body, far, jq, jqd

    def record(self):
        e = self.env
        body, far, jq, jqd = self.world_state()
        r = self.rows
        r["body"].append(body)
        r["far"].append(far)
        r["joint_q"].append(jq)
        r["joint_qd"].append(jqd)
        r["obs"].append(np.isfinite(e.observation.numpy().reshape(self.W, self.obs_dim)).all(axis=1))
        r["ctx"].append(np.isfinite(e.critic_context.numpy().reshape(self.W, self.ctx_dim)).all(axis=1))
        r["term_obs"].append(np.isfinite(e.terminal_observation.numpy().reshape(self.W, self.obs_dim)).all(axis=1))
        r["warm"].append(np.isfinite(self.warm.numpy()).all(axis=1))
        r["act"].append(np.isfinite(self.act.numpy()).all(axis=1) if self.act is not None else np.ones(self.W, bool))
        r["done"].append(e.done.numpy().copy())
        r["trunc"].append(e.truncation.numpy().copy())
        r["diag"].append(e.termination.packed_diagnostics.numpy().reshape(self.W, 9).copy())
        r["episode_step"].append(e.episode_step.numpy().copy())
        r["timestep"].append(e.timestep.numpy().copy())

    def arrays(self):
        return {k: np.asarray(v) for k, v in self.rows.items()}


def healthy(a, t, w):
    """第 t 步结束时世界 w 的状态与观测都有限（终止观测另算）。"""
    return bool(a["body"][t, w] and not a["far"][t, w] and a["joint_q"][t, w] and a["joint_qd"][t, w]
                and a["obs"][t, w] and a["ctx"][t, w])


def load_upstream_termination():
    spec = importlib.util.spec_from_file_location("v2d_termination_upstream", UPSTREAM_TERMINATION)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class K1:
    """注入之前每一步：拷一份终止判定的输入，上游与打了防护的两版 kernel 各算一次，比 12 个输出。"""

    def __init__(self, env):
        from flash_chord.lifecycle import termination as patched

        self.env = env
        self.term = env.termination
        self.patched = patched.evaluate_recon_body_termination
        self.upstream = load_upstream_termination().evaluate_recon_body_termination
        self.steps = 0
        self.mismatch = []
        self.evaluations = 0
        self.terminated = 0
        self.frozen = 0
        self.unfrozen_terminated = 0

    def outputs(self):
        W = self.term.world_count
        ints = [wp.zeros(W, dtype=wp.int32) for _ in range(5)]
        floats = [wp.zeros(W, dtype=wp.float32) for _ in range(6)]
        return [*ints, *floats, wp.zeros(W * 9, dtype=wp.float32)]

    def launch(self, kernel, body_q, timestep, episode_step, obj_pos, obj_quat, outs):
        t = self.term
        c = t.config
        wp.launch(
            kernel,
            dim=t.world_count,
            inputs=[
                body_q, timestep, episode_step,
                t.reference.joint_q, t.reference.wrist_pos_w, t.reference.wrist_quat_w,
                t.wrist_frames.body_ids, t.wrist_frames.body_to_frame_pos, t.wrist_frames.body_to_frame_quat,
                t.object_body_ids_w, obj_pos, obj_quat,
                t.pelvis_body_id, t.reference.num_frames, t.bodies_per_world, 2, t.num_objects, c.reset_freeze_steps,
                c.pelvis_position_threshold, c.pelvis_orientation_threshold, c.palm_position_threshold,
                c.palm_orientation_threshold, c.object_position_threshold, c.object_orientation_threshold,
                int(c.truncate_at_reference_end),
            ],
            outputs=outs,
        )

    def check(self):
        e = self.env
        body_q = wp.clone(e.state_0.body_q)
        timestep = wp.clone(e.timestep)
        episode_step = wp.clone(e.episode_step)
        obj_pos = wp.clone(self.term.object_ref_pos_w)
        obj_quat = wp.clone(self.term.object_ref_quat_w)
        assert np.isfinite(body_q.numpy()).all(), "K1 只用全部有限的 body_q"
        old, new = self.outputs(), self.outputs()
        self.launch(self.upstream, body_q, timestep, episode_step, obj_pos, obj_quat, old)
        self.launch(self.patched, body_q, timestep, episode_step, obj_pos, obj_quat, new)
        self.steps += 1
        for i, (a, b) in enumerate(zip(old, new)):
            a, b = a.numpy(), b.numpy()
            if a.dtype != b.dtype or a.tobytes() != b.tobytes():
                self.mismatch.append((self.steps, i))
        terminated = old[0].numpy()
        frozen = episode_step.numpy() <= self.term.config.reset_freeze_steps
        self.evaluations += terminated.size
        self.terminated += int(terminated.sum())
        self.frozen += int(frozen.sum())
        self.unfrozen_terminated += int((terminated.astype(bool) & ~frozen).sum())


def inject(env, gate, worlds):
    W = env.world_count
    nq = env.model.joint_coord_count // W
    nv = env.model.joint_dof_count // W
    q_root = int(env.reset_table.object_root_q_ids.numpy()[0])
    for w in worlds:
        if gate in ("t1", "t2a", "t2d", "t2n"):
            wp.launch(write_span, dim=nv, inputs=[env.state_0.joint_qd, w * nv, float("nan")])
        elif gate == "t2b":
            wp.launch(write_span, dim=3, inputs=[env.state_0.joint_q, w * nq + q_root, 1.0e6])
        elif gate == "t2c":
            wp.launch(write_span, dim=4, inputs=[env.state_0.joint_q, w * nq + q_root + 3, float("nan")])
        elif gate == "t1b":
            warm, _ = solver_rows(env)
            wp.launch(write_row, dim=warm.shape[1], inputs=[warm, w, float("nan")])
        else:
            raise ValueError(gate)
    wp.synchronize_device(env.device)


def explicit_reset(env, world, frame):
    W = env.world_count
    mask = np.zeros(W, np.int32)
    mask[world] = 1
    frames = np.zeros(W, np.int32)
    frames[world] = frame
    env.reset(reset_frame=wp.array(frames, dtype=wp.int32, device=env.device),
              reset_mask=wp.array(mask, dtype=wp.int32, device=env.device))


def fmt_diag(d):
    return "[" + ", ".join("nan" if np.isnan(x) else f"{x:.4g}" for x in d) + "]"


def print_world(a, w, start, stop):
    for t in range(start, stop):
        print(f"  第 {t + 1:3d} 步 世界 {w}：done {a['done'][t, w]} trunc {a['trunc'][t, w]} 诊断 {fmt_diag(a['diag'][t, w])} "
              f"刚体有限 {int(a['body'][t, w])} 超远 {int(a['far'][t, w])} q {int(a['joint_q'][t, w])} qd {int(a['joint_qd'][t, w])} "
              f"观测 {int(a['obs'][t, w])} 上下文 {int(a['ctx'][t, w])} 终止观测 {int(a['term_obs'][t, w])} "
              f"warmstart {int(a['warm'][t, w])} act {int(a['act'][t, w])} episode_step {a['episode_step'][t, w]} timestep {a['timestep'][t, w]}")


def verdict_t1(a, w, first):
    """第一次被重置（done 或 trunc）之后的下一步，这个世界在那一步的仿真里是否又变成非有限。

    看那一步的终止观测（仿真之后、自动重置之前算的）。打了防护的副本会在同一步把坏世界判终止、重置，步末状态因此总是
    重置后的有限状态，只看步末状态会漏掉（t2n 第一次跑就这样误判成“恢复了”）；未打防护的副本冻结期内不判终止，
    步末状态与终止观测都是非有限。步末状态非有限的步数另报。
    """
    T = a["done"].shape[0]
    completed = [t for t in range(first, T) if a["done"][t, w] or a["trunc"][t, w]]
    if not completed:
        print(f"世界 {w} 在注入后 {POST_STEPS} 步里没有被重置过")
        return False, completed
    t_reset = completed[0]
    if t_reset + 1 >= T:
        print("第一次重置发生在最后一步，看不到重置之后")
        return False, completed
    after = not a["term_obs"][t_reset + 1, w]
    term_bad = sum(not a["term_obs"][t, w] for t in range(t_reset + 1, T))
    end_bad = sum(not healthy(a, t, w) for t in range(t_reset + 1, T))
    warm_bad = sum(not a["warm"][t, w] for t in range(t_reset + 1, T))
    print(f"世界 {w}：注入后第一次被重置在第 {t_reset + 1} 步（原因列 {fmt_diag(a['diag'][t_reset, w][:3])}）；"
          f"重置后的下一步（第 {t_reset + 2} 步）仿真之后{'仍然非有限' if after else '是有限的'}；"
          f"重置之后共 {T - t_reset - 1} 步里，终止观测非有限 {term_bad} 步、步末状态或观测非有限 {end_bad} 步、warmstart 非有限 {warm_bad} 步；"
          f"注入后被重置的步：{[t + 1 for t in completed][:12]}{'…' if len(completed) > 12 else ''}（共 {len(completed)} 次）")
    return after, completed


def verdict_t2(a, injected, first):
    ok = True
    T = a["done"].shape[0]
    for w in injected:
        row = a["diag"][first, w].tolist()
        flagged = bool(a["done"][first, w] == 1 and row == SENTINEL_ROW)
        warm_ok = bool(a["warm"][first, w] and a["act"][first, w])
        bad_steps = [t + 1 for t in range(first, T) if not healthy(a, t, w)]
        bad_term = [t + 1 for t in range(first + 1, T) if not a["term_obs"][t, w]]
        warm_bad = [t + 1 for t in range(first, T) if not (a["warm"][t, w] and a["act"][t, w])]
        print(f"世界 {w}：注入后第一步 done {a['done'][first, w]} trunc {a['trunc'][first, w]} 诊断 {fmt_diag(row)} → "
              f"{'按骨盆判坏、哨兵 -1' if flagged else '没有按判据判坏'}；这一步之后 warmstart/act {'有限' if warm_ok else '非有限'}；"
              f"之后 {T - first} 步里状态或观测非有限的步 {bad_steps[:10]}，终止观测非有限的步（第一步之后）{bad_term[:10]}，"
              f"warmstart/act 非有限的步 {warm_bad[:10]}")
        ok &= flagged and warm_ok and not bad_steps and not bad_term
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("gate", choices=GATES)
    ap.add_argument("--out", default=None)
    a_ = ap.parse_args()
    gate = a_.gate
    out = Path(a_.out or f"/repo/runs/nfg_probe/{gate}.npz")
    if out.exists():
        print(f"{out} 已存在，不覆盖", flush=True)
        return 2

    wp.init()
    from flash_chord.envs import base
    from flash_chord.lifecycle import termination

    guarded = "nonfinite" in Path(termination.__file__).read_text()
    if guarded == (gate in PLAIN_GATES):
        print(f"挂错了副本：闸 {gate} 要{'未' if gate in PLAIN_GATES else ''}打防护的副本，挂上的 {termination.__file__} "
              f"{'有' if guarded else '没有'}防护", flush=True)
        return 2
    if gate == "t2n":
        base._solver_rows_to_clear = lambda solver, world_count: ()
        print("t2n：本进程里关掉了重置时清 warmstart（envs.base._solver_rows_to_clear 换成返回空）", flush=True)

    env, device = build_env()
    rec = Recorder(env)
    print(f"每个世界：刚体 {rec.B}、joint_q {rec.nq}、joint_qd {rec.nv}、观测 {rec.obs_dim}、上下文 {rec.ctx_dim}；"
          f"qacc_warmstart {tuple(rec.warm.shape)}，act {'无（na = 0）' if rec.act is None else tuple(rec.act.shape)}", flush=True)
    injected = {"t1b": [9], "t2d": [3, 5, 7]}.get(gate, [7])
    k1 = K1(env) if gate == "t2a" else None
    nf = env.reference.num_frames
    with wp.ScopedDevice(device):
        for step in range(1, PRE_STEPS + 1):
            env.step()
            rec.record()
            if k1 is not None:
                k1.check()
            if gate == "t2d" and step == 90:
                explicit_reset(env, 3, 0)
            if gate == "t2d" and step == 96:
                explicit_reset(env, 5, nf - 5)
        pre = rec.arrays()
        pre_bad = [(t + 1, w) for t in range(PRE_STEPS) for w in range(env.world_count) if not healthy(pre, t, w)]
        print(f"注入之前 {PRE_STEPS} 步：状态或观测非有限的（步，世界）{pre_bad[:10]}；"
              f"done 合计 {int(pre['done'].sum())}，trunc 合计 {int(pre['trunc'].sum())}", flush=True)
        print(f"注入：闸 {gate}，世界 {injected}；此刻 episode_step {[int(pre['episode_step'][-1, w]) for w in injected]}，"
              f"timestep {[int(pre['timestep'][-1, w]) for w in injected]}（参考最后一帧 {nf - 1}）", flush=True)
        inject(env, gate, injected)
        for _ in range(POST_STEPS):
            env.step()
            rec.record()
    a = rec.arrays()
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, gate=gate, injected=np.asarray(injected), pre_steps=PRE_STEPS, num_frames=nf, **a)
    print(f"已存 {out}", flush=True)

    first = PRE_STEPS  # 注入后第一步在数组里的下标
    for w in injected:
        print(f"== 世界 {w}，注入前 3 步到注入后 12 步：")
        print_world(a, w, first - 3, min(first + 12, a["done"].shape[0]))
    others = [w for w in range(env.world_count) if w not in injected]
    other_bad = [(t + 1, w) for t in range(a["done"].shape[0]) for w in others
                 if not healthy(a, t, w) or not a["term_obs"][t, w]]
    print(f"其余 {len(others)} 个世界 {a['done'].shape[0]} 步里状态、观测或终止观测非有限的（步，世界）：{other_bad[:10]}")

    if gate in ("t1", "t2n"):
        stays, _ = verdict_t1(a, injected[0], first)
        if gate == "t2n":
            row = a["diag"][first, injected[0]].tolist()
            print(f"t2n：注入后第一步诊断 {fmt_diag(row)}（{'按骨盆判坏、哨兵 -1' if row == SENTINEL_ROW else '不是哨兵行'}）")
        ok = stays and not other_bad
        print(f"{gate.upper()} 判据（重置后的下一步仿真之后仍然非有限，其余世界有限）：{'过' if ok else '不过'}")
    elif gate == "t1b":
        w = injected[0]
        bq_bad_all = not a["body"][first, w]
        print(f"T1B：只写 warmstart 之后的下一步，世界 {w} 刚体位姿非有限：{bq_bad_all}；关节量 q {int(a['joint_q'][first, w])} qd {int(a['joint_qd'][first, w])}")
        ok = bq_bad_all and not other_bad
        print(f"T1B 判据（下一步变成非有限，其余世界有限）：{'过' if ok else '不过'}")
    else:
        ok = verdict_t2(a, injected, first) and not other_bad
        if gate == "t2d":
            print(f"t2d：注入后第一步 trunc：世界 3 {a['trunc'][first, 3]}、世界 5 {a['trunc'][first, 5]}、世界 7 {a['trunc'][first, 7]}；"
                  f"注入时 episode_step {[int(a['episode_step'][first - 1, w]) for w in injected]}")
            ok &= bool(a["trunc"][first, 5] == 1 and a["episode_step"][first - 1, 3] < env.termination.config.reset_freeze_steps)
        if k1 is not None:
            print(f"K1：注入之前 {k1.steps} 步、{k1.evaluations} 次世界评估（其中冻结期 {k1.frozen}、上游判终止 {k1.terminated}、"
                  f"冻结期外判终止 {k1.unfrozen_terminated}）；12 个输出不逐位相同的（步，输出号）{k1.mismatch[:10]}")
            print(f"K1 判据：{'过' if not k1.mismatch else '不过'}")
            ok &= not k1.mismatch
        print(f"{gate.upper()} 判据：{'过' if ok else '不过'}")
    print(f"NFG_PROBE_DONE {gate} {'PASS' if ok else 'FAIL'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
