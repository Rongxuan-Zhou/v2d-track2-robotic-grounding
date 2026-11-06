"""诊断 rollout 用：每一步之后抄下机器人两只手掌与物体刚体的世界坐标（FlashCHORD 容器里跑，占 GPU；只读仿真状态，不写）。

计划 agent-doc/plan-diag-mpsr0-2026-10-02.md 第 3.1 节。公式与 ReconBody 终止（flash_chord/lifecycle/termination.py，v0.4.0）
每一步算 palm 误差时相同：palm(w, h) = transform_point(body_q[w × bodies_per_world + wrist_frames.body_ids[h]], wrist_frames.body_to_frame_pos[h])，
h 的顺序是 termination.config.sides。物体：body_q[env.command.body_ids_w[w × B + b]] 的平移，与评测器 record_object_pose 抄的是同一个刚体。
抄的时机：包一层传给 record_object_pose_rollout 的 jax_env，在它的 step 返回之前抄，与评测器在 step 之后抄物体位姿读的是同一个 env.state_0。
diag_rollout.py 用抄下的物体位置与评测器自己记的 achieved_pose_w 逐位比，作为同步闸。
用法（diag_rollout.py --palms 内部）：rec = PalmRecorder(env, step_count)；把 jax_env 换成 rec.wrap(jax_env)；跑完 palms, objects = rec.arrays()
"""
import numpy as np
import warp as wp


@wp.kernel
def _gather_palms(body_q: wp.array(dtype=wp.transform), body_ids: wp.array(dtype=wp.int32), local_pos: wp.array(dtype=wp.vec3),
                  bodies_per_world: int, hands: int, step: int, slots: int, out: wp.array(dtype=wp.vec3)):
    w, h = wp.tid()
    out[step * slots + w * hands + h] = wp.transform_point(body_q[w * bodies_per_world + body_ids[h]], local_pos[h])


@wp.kernel
def _gather_objects(body_q: wp.array(dtype=wp.transform), object_ids_w: wp.array(dtype=wp.int32), step: int, slots: int,
                    out: wp.array(dtype=wp.vec3)):
    i = wp.tid()
    out[step * slots + i] = wp.transform_get_translation(body_q[object_ids_w[i]])


class _StepProxy:
    """只截 step：其余属性原样转给被包的 jax_env。"""

    def __init__(self, inner, after):
        self._inner = inner
        self._after = after

    def step(self, action):
        result = self._inner.step(action)
        self._after()
        return result

    def __getattr__(self, name):
        return getattr(self._inner, name)


class PalmRecorder:
    def __init__(self, env, step_count: int):
        term = env.termination
        missing = [a for a in ("wrist_frames", "bodies_per_world", "world_count", "config") if not hasattr(term, a)]
        if missing:
            raise RuntimeError(f"env.termination（{type(term).__name__}）缺 {missing}：不是 v0.4.0 的 ReconBodyTermination，--palms 用不了")
        frames = term.wrist_frames
        self.env, self.device = env, env.device
        self.sides = tuple(str(s) for s in term.config.sides)
        self.hands = int(frames.count)
        if self.hands != len(self.sides) or self.hands < 1:
            raise RuntimeError(f"手掌坐标帧 {self.hands} 个，与 sides {self.sides} 对不上")
        self.world_count = int(term.world_count)
        self.bodies_per_world = int(term.bodies_per_world)
        self.body_ids, self.local_pos = frames.body_ids, frames.body_to_frame_pos
        self.object_ids_w = env.command.body_ids_w
        self.bodies = int(env.command.layout.num_bodies)
        self.steps = int(step_count)
        self.palm_slots = self.world_count * self.hands
        self.object_slots = self.world_count * self.bodies
        self.palms = wp.zeros(self.steps * self.palm_slots, dtype=wp.vec3, device=self.device)
        self.objects = wp.zeros(self.steps * self.object_slots, dtype=wp.vec3, device=self.device)
        self.step = 0

    def _after(self):
        if self.step >= self.steps:
            raise RuntimeError(f"jax_env.step 被调用超过 step_count（{self.steps}）次")
        body_q = self.env.state_0.body_q
        # 显式给 device：评测器的注释说，launch 的 device 与数组不一致时会被静默跳过
        wp.launch(_gather_palms, dim=(self.world_count, self.hands), device=self.device,
                  inputs=[body_q, self.body_ids, self.local_pos, self.bodies_per_world, self.hands, self.step, self.palm_slots],
                  outputs=[self.palms])
        wp.launch(_gather_objects, dim=self.object_slots, device=self.device,
                  inputs=[body_q, self.object_ids_w, self.step, self.object_slots], outputs=[self.objects])
        self.step += 1

    def wrap(self, jax_env):
        return _StepProxy(jax_env, self._after)

    def arrays(self):
        if self.step != self.steps:
            raise RuntimeError(f"jax_env.step 被调用 {self.step} 次，应为 {self.steps} 次")
        palms = self.palms.numpy().reshape(self.steps, self.world_count, self.hands, 3)
        objects = self.objects.numpy().reshape(self.steps, self.world_count, self.bodies, 3)
        return palms, objects


@wp.kernel
def _gather_bodies(body_q: wp.array(dtype=wp.transform), body_ids: wp.array(dtype=wp.int32), bodies_per_world: int, n: int,
                   step: int, slots: int, out: wp.array(dtype=wp.transform)):
    w, k = wp.tid()
    out[step * slots + w * n + k] = body_q[w * bodies_per_world + body_ids[k]]


class BodyRecorder:
    """--bodies（计划 agent-doc/plan-t12fall-2026-10-20.md）：每一步之后抄下指定刚体（默认骨盆、躯干、两只脚踝）的完整位姿（位置 + 四元数 xyzw），
    时机同 PalmRecorder。刚体编号按 env.model.body_label（没有就用 body_key）在第 0 个世界里的末段名字找（标签可能带前缀，取最后一个 “/” 之后的部分）；
    每个名字必须恰好找到一个，否则报错并打印前 80 个标签。"""

    NAMES = ("pelvis", "torso_link", "left_ankle_roll_link", "right_ankle_roll_link")

    def __init__(self, env, step_count: int, names=NAMES):
        term = env.termination
        self.env, self.device = env, env.device
        self.world_count = int(term.world_count)
        self.bodies_per_world = int(term.bodies_per_world)
        raw = getattr(env.model, "body_label", None) or getattr(env.model, "body_key", None)
        if not raw:
            raise RuntimeError("env.model 既没有 body_label 也没有 body_key（或为空），找不到刚体名")
        labels = [str(x) for x in raw]
        first = [lab.rsplit("/", 1)[-1] for lab in labels[: self.bodies_per_world]]
        ids = []
        for n in names:
            hit = [i for i, lab in enumerate(first) if lab == n]
            if len(hit) != 1:
                raise RuntimeError(f"刚体 {n} 在第 0 个世界里找到 {len(hit)} 个；前 80 个标签：{labels[:80]}")
            ids.append(hit[0])
        self.names = tuple(names)
        self.n = len(ids)
        self.body_ids = wp.array(ids, dtype=wp.int32, device=self.device)
        self.steps = int(step_count)
        self.slots = self.world_count * self.n
        self.poses = wp.zeros(self.steps * self.slots, dtype=wp.transform, device=self.device)
        self.step = 0

    def _after(self):
        if self.step >= self.steps:
            raise RuntimeError(f"jax_env.step 被调用超过 step_count（{self.steps}）次")
        wp.launch(_gather_bodies, dim=(self.world_count, self.n), device=self.device,
                  inputs=[self.env.state_0.body_q, self.body_ids, self.bodies_per_world, self.n, self.step, self.slots], outputs=[self.poses])
        self.step += 1

    def wrap(self, jax_env):
        return _StepProxy(jax_env, self._after)

    def arrays(self):
        if self.step != self.steps:
            raise RuntimeError(f"jax_env.step 被调用 {self.step} 次，应为 {self.steps} 次")
        return self.poses.numpy().reshape(self.steps, self.world_count, self.n, 7)


class _ActionProxy:
    """只截 step：先把传进来的动作交给 record，再原样转给被包的 jax_env；其余属性原样转给被包的 jax_env。"""

    def __init__(self, inner, record):
        self._inner = inner
        self._record = record

    def step(self, action):
        self._record(action)
        return self._inner.step(action)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class ActionRecorder:
    """--actions N（计划 agent-doc/plan-t12sat-2026-10-20.md）：抄策略交给 jax_env.step 的动作（FlashSAC 的确定性动作 = action_scale · tanh(均值)，
    还没乘 residual_scale），只抄前 N 步、前 --action-worlds 个世界，存 float32 (N, W', A)。步数不足 N 时只存实际跑到的步数。"""

    def __init__(self, max_steps: int, worlds: int):
        self.max_steps, self.worlds = int(max_steps), int(worlds)
        self.rows = []
        self.step = 0

    def _record(self, action):
        if self.step < self.max_steps:
            self.rows.append(np.asarray(action[: self.worlds], dtype=np.float32))
        self.step += 1

    def wrap(self, jax_env):
        return _ActionProxy(jax_env, self._record)

    def arrays(self):
        if not self.rows:
            raise RuntimeError("一步动作也没抄到")
        return np.stack(self.rows)


@wp.kernel
def _gather_fingers(joint_q: wp.array(dtype=wp.float32), joint_target: wp.array(dtype=wp.float32), q_ids: wp.array(dtype=wp.int32),
                    dof_ids: wp.array(dtype=wp.int32), q_per_world: int, dof_per_world: int, n: int, step: int, slots: int,
                    out_q: wp.array(dtype=wp.float32), out_target: wp.array(dtype=wp.float32)):
    w, k = wp.tid()
    out_q[step * slots + w * n + k] = joint_q[w * q_per_world + q_ids[k]]
    out_target[step * slots + w * n + k] = joint_target[w * dof_per_world + dof_ids[k]]


class FingerRecorder:
    """--fingers（agent-doc/experiments/2026-10-30-t1hard-diag.md 的第 2 件事）：每一步之后抄下 14 个 Dex3 手指关节的实际角度（state_0.joint_q）
    与这一步交给 PD 的目标角（env.action.joint_target），时机同 PalmRecorder。关节编号用动作项自己的 finger_q_ids、finger_dof_ids
    （runtime/actions/sonic.py：目标 = 参考手指角 + 每世界偏移 + finger_residual_scale × 残差），每世界的坐标数与自由度数按 model 的总数除以世界数。"""

    def __init__(self, env, step_count: int):
        act = env.action
        missing = [x for x in ("finger_q_ids", "finger_dof_ids", "finger_joint_names", "joint_target") if not hasattr(act, x)]
        if missing:
            raise RuntimeError(f"env.action（{type(act).__name__}）缺 {missing}：不是 SONIC 动作项，--fingers 用不了")
        self.env, self.device = env, env.device
        self.world_count = int(env.termination.world_count)
        self.q_per_world = int(env.model.joint_coord_count) // self.world_count
        self.dof_per_world = int(env.model.joint_dof_count) // self.world_count
        self.names = tuple(str(x) for x in act.finger_joint_names)
        self.q_ids, self.dof_ids = act.finger_q_ids, act.finger_dof_ids
        self.n = len(self.names)
        if self.n != int(self.q_ids.shape[0]) or self.n != int(self.dof_ids.shape[0]):
            raise RuntimeError(f"手指名 {self.n} 个，与 q_ids {self.q_ids.shape}、dof_ids {self.dof_ids.shape} 对不上")
        self.steps = int(step_count)
        self.slots = self.world_count * self.n
        self.q = wp.zeros(self.steps * self.slots, dtype=wp.float32, device=self.device)
        self.target = wp.zeros(self.steps * self.slots, dtype=wp.float32, device=self.device)
        self.step = 0

    def _after(self):
        if self.step >= self.steps:
            raise RuntimeError(f"jax_env.step 被调用超过 step_count（{self.steps}）次")
        wp.launch(_gather_fingers, dim=(self.world_count, self.n), device=self.device,
                  inputs=[self.env.state_0.joint_q, self.env.action.joint_target, self.q_ids, self.dof_ids, self.q_per_world,
                          self.dof_per_world, self.n, self.step, self.slots], outputs=[self.q, self.target])
        self.step += 1

    def wrap(self, jax_env):
        return _StepProxy(jax_env, self._after)

    def arrays(self):
        if self.step != self.steps:
            raise RuntimeError(f"jax_env.step 被调用 {self.step} 次，应为 {self.steps} 次")
        shape = (self.steps, self.world_count, self.n)
        return self.q.numpy().reshape(shape), self.target.numpy().reshape(shape)
