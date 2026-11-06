"""非有限防护补丁的学习器侧组件测试（K3；守 docs/contracts/policy.md 的 POLICY-I004、POLICY-I005、POLICY-I006）。在 FlashCHORD 容器里、CPU 上跑 JAX，挂打了防护补丁的副本：
  V2D_FC_SRC=<fc_src/v040_p185_nfg> scripts/dev/in-flash-chord "JAX_PLATFORMS=cpu python /repo/stages/policy/nfg_test_learner.py"
计划 agent-doc/plan-ep7-eval-and-guards-2026-10-02.md 的 B.4。学习器按上游测试的小配置建（2 个世界、观测 3 维、动作 2 维、分项 2 个）。

  1. 干净的 transition：清洗函数的每个输出数组与输入逐位相同（含 -0.0），计数都是 0。
  2. 坏的输入：分项 NaN、下一步观测 inf、重置后观测 1e5（fp16 存不下）、重置后上下文 NaN、动作 NaN（这几种分开测）：
     输出全部有限；坏世界的分项是“终止 1、其余 0”、terminated 为 1、骨盆原因位为 1；被换掉的观测、上下文等于学习器当前的观测、上下文；另一个世界逐位不变。
     相位取终止时的上下文（next_critic_context）。
  3. 环境判坏（骨盆原因 1、骨盆位置误差 -1）：记在 env_flagged，不记 learner_only；哨兵值计数后清成 0。
     环境判坏且已重置的世界：下一步观测与上下文是 NaN 时换成学习器当前的，重置后的观测与上下文有限，逐位原样返回。
  4. 接到真的 consume_transition 上连喂 8 步（n-step 3、回放按 fp16 存），第 2、5 步注入坏世界：回放的 storage 与 pending、奖励归一化器、学习器的观测与指标全部有限。
     反向对照：同样的输入不经清洗直接喂，奖励归一化器变成 NaN（说明这个测试测得出问题）。
  5. 窗口报告与停止条件：连续 3 步坏 → 停止理由；learner_only 累计超过 100 → 停止理由；reset_window 只清窗口量；
     事件行以 NONFINITE_GUARD_EVENT 开头、停训行以 NONFINITE_GUARD_STOP 开头，键值对可按名字取。
最后一行 NFG_LEARNER_OK 或报错退出。
"""
import jax
import jax.numpy as jnp
import numpy as np
from flash_chord.lifecycle.reset import RESET_CONTEXT_NAMES
from flash_chord.training.flash_sac import nonfinite_guard as ng
from flash_chord.training.flash_sac.config import (
    ExplorationConfig,
    NetworkConfig,
    OptimizationConfig,
    ReplayConfig,
    RewardNormalizationConfig,
    TrainingConfig,
)
from flash_chord.training.flash_sac.learner import Transition, create_learner

CONTEXT_NAMES = tuple(RESET_CONTEXT_NAMES) + ("normalized_reference_phase",)


def config(storage="float16"):
    return TrainingConfig(
        world_count=2,
        total_environment_steps=40,
        updates_per_collection=2.0,
        mixed_precision=False,
        compute_dtype="float32",
        replay=ReplayConfig(capacity=16, minimum_size=2, batch_size=2, n_step=3, observation_storage_dtype=storage),
        network=NetworkConfig(actor_blocks=1, actor_hidden_dim=8, critic_blocks=1, critic_hidden_dim=8, expansion=2, atom_count=11),
        optimization=OptimizationConfig(),
        exploration=ExplorationConfig(),
        reward_normalization=RewardNormalizationConfig(enabled=True),
    )


def initial():
    observation = jnp.arange(6, dtype=jnp.float32).reshape(2, 3) / 10.0
    context = jnp.asarray([[1.0, 0.5, 0.0, 0.25], [0.75, 0.25, 1.0, 0.5]], dtype=jnp.float32)
    return observation, context


def layout(storage="float16"):
    return ng.guard_layout(
        storage_dtype=storage,
        objective_term_names=("termination", "tracking"),
        termination_cause_names=("pelvis",),
        tracking_error_names=("pelvis_position_m",),
        critic_context_names=CONTEXT_NAMES,
    )


def transition(step=0):
    observation, context = initial()
    terminated = jnp.asarray([False, step % 3 == 2], dtype=jnp.int32)
    truncated = jnp.asarray([step % 2 == 1, False], dtype=jnp.int32)
    return Transition(
        action=jnp.asarray([[0.1, -0.0], [0.3, -0.2]], dtype=jnp.float32),
        objective_terms=jnp.asarray([[0.0, 0.5 + step], [float(step % 3 == 2), 0.75]], dtype=jnp.float32),
        terminated=terminated,
        truncated=truncated,
        episode_reference_progress=jnp.asarray([0.5 * (step % 2), 0.25], dtype=jnp.float32),
        termination_diagnostics=jnp.asarray([[0.0, 0.1 + step], [float(step % 3 == 2), 0.8]], dtype=jnp.float32),
        next_observation=observation + 1.0 + step,
        next_critic_context=context + 0.25,
        post_reset_observation=observation + 2.0 + step,
        post_reset_critic_context=context + 0.5,
    )


def same_bits(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


FIELDS = ("action", "objective_terms", "terminated", "truncated", "episode_reference_progress", "termination_diagnostics",
          "next_observation", "next_critic_context", "post_reset_observation", "post_reset_critic_context")


def test_clean(sanitize) -> None:
    observation, context = initial()
    t = transition(0)
    out, state = sanitize(t, observation, context, ng.create_guard_state(2))
    for name in FIELDS:
        assert same_bits(getattr(out, name), getattr(t, name)), f"干净输入的 {name} 变了"
    assert int(state.env_flagged) == 0 and int(state.learner_only) == 0 and int(state.max_consecutive) == 0
    print("1 干净输入：10 个字段逐位相同（动作里的 -0.0 也保留），计数为 0")


def test_bad(sanitize) -> None:
    observation, context = initial()
    cases = {
        "分项 NaN": lambda t: t.replace(objective_terms=t.objective_terms.at[0, 1].set(jnp.nan)),
        "下一步观测 inf": lambda t: t.replace(next_observation=t.next_observation.at[0, 2].set(jnp.inf)),
        "重置后观测 1e5": lambda t: t.replace(post_reset_observation=t.post_reset_observation.at[0, 0].set(1.0e5)),
        "重置后上下文 NaN": lambda t: t.replace(post_reset_critic_context=t.post_reset_critic_context.at[0, 1].set(jnp.nan)),
        "动作 NaN": lambda t: t.replace(action=t.action.at[0, 0].set(jnp.nan)),
    }
    for label, corrupt in cases.items():
        t = corrupt(transition(0))
        out, state = sanitize(t, observation, context, ng.create_guard_state(2))
        for name in FIELDS:
            assert np.isfinite(np.asarray(getattr(out, name))).all(), (label, name)
        assert np.asarray(out.objective_terms)[0].tolist() == [1.0, 0.0], label
        assert int(out.terminated[0]) == 1 and float(out.termination_diagnostics[0, 0]) == 1.0, label
        for name in FIELDS:
            assert same_bits(np.asarray(getattr(out, name))[1], np.asarray(getattr(t, name))[1]), (label, "另一个世界变了", name)
        if label == "下一步观测 inf":
            assert same_bits(np.asarray(out.next_observation)[0], np.asarray(observation)[0])
        if label == "重置后观测 1e5":
            assert same_bits(np.asarray(out.post_reset_observation)[0], np.asarray(observation)[0])
        if label == "重置后上下文 NaN":
            assert same_bits(np.asarray(out.post_reset_critic_context)[0], np.asarray(context)[0])
        if label == "动作 NaN":
            assert float(out.action[0, 0]) == 0.0 and float(out.action[0, 1]) == float(t.action[0, 1])
        assert int(state.learner_only) == 1 and int(state.env_flagged) == 0, label
        assert float(state.phase_min) == float(t.next_critic_context[0, 3]) == float(state.phase_max), label
        assert int(state.first_world) == 0, label
    print(f"2 坏输入：{len(cases)} 种情形都清洗成有限值，坏世界成了“终止 1、其余 0”的失败样本，另一个世界逐位不变")


def test_env_flag(sanitize) -> None:
    observation, context = initial()
    t = transition(0)
    t = t.replace(
        terminated=t.terminated.at[1].set(1),
        termination_diagnostics=t.termination_diagnostics.at[1].set(jnp.asarray([1.0, ng.ENV_FLAG_PELVIS_ERROR])),
    )
    out, state = sanitize(t, observation, context, ng.create_guard_state(2))
    assert int(state.env_flagged) == 1 and int(state.learner_only) == 0
    assert np.asarray(out.objective_terms)[1].tolist() == [1.0, 0.0]
    assert float(out.termination_diagnostics[1, 0]) == 1.0 and float(out.termination_diagnostics[1, 1]) == 0.0
    for name in FIELDS:
        assert same_bits(np.asarray(getattr(out, name))[0], np.asarray(getattr(t, name))[0]), ("另一个世界变了", name)

    # 环境已经重置了这个世界：终止时的观测与上下文是 NaN，重置后的是新回合的第一帧观测（有限），后者要原样留下
    t2 = t.replace(
        next_observation=t.next_observation.at[1].set(jnp.nan),
        next_critic_context=t.next_critic_context.at[1].set(jnp.nan),
    )
    out2, state2 = sanitize(t2, observation, context, ng.create_guard_state(2))
    assert int(state2.env_flagged) == 1 and int(state2.learner_only) == 0
    assert same_bits(np.asarray(out2.next_observation)[1], np.asarray(observation)[1])
    assert same_bits(np.asarray(out2.next_critic_context)[1], np.asarray(context)[1])
    assert same_bits(out2.post_reset_observation, t2.post_reset_observation), "重置后的观测有限时应原样返回"
    assert same_bits(out2.post_reset_critic_context, t2.post_reset_critic_context), "重置后的上下文有限时应原样返回"
    assert float(state2.phase_min) == float(context[1, 3]), "终止上下文的相位是 NaN 时退回学习器当前上下文的相位"
    print("3 环境判坏：记在 env_flagged，不记 learner_only；哨兵值 -1 计数后清成 0；终止观测 NaN 时只换它，"
          "有限的重置后观测与上下文逐位原样返回")


def consume_run(cfg, sanitize, *, use_guard: bool):
    observation, context = initial()
    state, executables = create_learner(
        cfg, observation, context, action_dim=2, objective_term_count=2, termination_cause_count=1,
        tracking_error_count=1, critic_context_names=CONTEXT_NAMES,
    )
    action, exploration, key = executables.initial_action(
        state.actor.params, state.actor.batch_stats, state.replay.size, state.observation, state.exploration, state.key)
    state = state.replace(exploration=exploration, key=key)
    guard = ng.create_guard_state(2)
    metrics = None
    for step in range(8):
        t = transition(step).replace(action=action)
        if step == 2:
            t = t.replace(objective_terms=t.objective_terms.at[0, 1].set(jnp.nan),
                          next_observation=t.next_observation.at[0, 0].set(jnp.nan))
        if step == 5:
            t = t.replace(post_reset_observation=t.post_reset_observation.at[1, 2].set(1.0e5),
                          post_reset_critic_context=t.post_reset_critic_context.at[1, 0].set(jnp.nan))
        if use_guard:
            t, guard = sanitize(t, state.observation, state.critic_context, guard)
        state, action, metrics = executables.consume_transition(
            state, t, jnp.asarray([1.0, 0.5], dtype=jnp.float32), jnp.asarray(0.05, dtype=jnp.float32),
            jnp.asarray(0.25, dtype=jnp.float32), jnp.asarray(False), jnp.asarray(False))
    return state, metrics, guard


def all_finite(tree) -> bool:
    return all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(tree)
               if np.issubdtype(np.asarray(x).dtype, np.floating))


def test_consume(sanitize) -> None:
    cfg = config("float16")
    state, metrics, guard = consume_run(cfg, sanitize, use_guard=True)
    assert all_finite(state.replay), "回放（storage 与 pending）里有非有限值"
    assert all_finite(state.reward_normalizer), "奖励归一化器非有限"
    assert all_finite(state.observation) and all_finite(state.critic_context)
    assert all_finite(state.actor) and all_finite(state.critic), "网络参数或 BatchNorm 统计量非有限"
    assert all_finite(metrics), "学习器指标非有限"
    assert int(guard.learner_only_total) == 2, int(guard.learner_only_total)
    bad_state, _, _ = consume_run(cfg, sanitize, use_guard=False)
    assert not all_finite(bad_state.reward_normalizer), "反向对照：不清洗时奖励归一化器应当变成 NaN"
    print("4 连喂 8 步（第 2、5 步有坏世界）：回放、归一化器、网络与指标全部有限；反向对照不清洗时归一化器变 NaN")


def test_report() -> None:
    state = ng.create_guard_state(2)
    _, reason = ng.window_report(state)
    assert reason is None
    state = state.replace(max_consecutive=jnp.asarray(3, dtype=jnp.int32))
    metrics, reason = ng.window_report(state)
    assert reason and "consecutive" in reason and metrics["guard/event_phase_min"] == -1.0
    state = ng.reset_window(state).replace(learner_only_total=jnp.asarray(101, dtype=jnp.int32))
    _, reason = ng.window_report(state)
    assert reason and "learner-only" in reason
    cleared = ng.reset_window(state.replace(learner_only=jnp.asarray(5, dtype=jnp.int32), first_world=jnp.asarray(1, dtype=jnp.int32)))
    assert int(cleared.learner_only) == 0 and int(cleared.first_world) == -1 and int(cleared.learner_only_total) == 101
    metrics, reason = ng.window_report(state.replace(env_flagged=jnp.asarray(2, dtype=jnp.int32), first_world=jnp.asarray(7, dtype=jnp.int32)))
    event = ng.event_line(614400, metrics)
    fields = dict(kv.split("=", 1) for kv in event.split()[1:])
    assert event.startswith("NONFINITE_GUARD_EVENT "), event
    assert fields["environment_steps"] == "614400" and fields["env_flagged"] == "2" and fields["first_world"] == "7", event
    assert fields["learner_only_total"] == "101" and fields["phase_min"] == "-1.0000", event
    stop = ng.stop_line(614400, metrics, reason)
    assert stop.startswith("NONFINITE_GUARD_STOP environment_steps=614400 learner_only_total=101 "), stop
    assert "learner-only" in stop.split("reason=", 1)[1], stop
    print("5 窗口报告：两种停止条件都报得出；reset_window 只清窗口量，累计量与连续计数保留；事件行、停训行的前缀与键值对可解析")


def main() -> int:
    assert jax.default_backend() == "cpu", jax.default_backend()
    sanitize = ng.make_transition_sanitizer(layout("float16"))
    test_clean(sanitize)
    test_bad(sanitize)
    test_env_flag(sanitize)
    test_consume(sanitize)
    test_report()
    print("NFG_LEARNER_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
