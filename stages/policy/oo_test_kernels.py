"""r3o 物体朝向项的组件测试（守 docs/contracts/policy.md 的 POLICY-I007、I008、I009；计划 agent-doc/plan-r3o-2026-10-15.md 2.1，审查第 7、8 条）。在 FlashCHORD 容器里、CPU 上跑，挂打了 r3o 补丁的副本：
  V2D_FC_SRC=<fc_src/v040_p185_nfg_oo> scripts/dev/in-flash-chord "python /repo/stages/policy/oo_test_kernels.py"
O1：同一组输入分别交给上游 v0.4.0 的 compose_recon_body_objective（从 /repo/third_party/video_to_data 里的源码按文件加载）与打了补丁的版本。
    第 13 项权重 0：前 12 项与分数逐位相同；第 13 项等于 numpy 按同一组四元数算的 exp(-θ²/σ²)（含与目标完全相同、互为相反数、相差 90° 的世界）。
    第 13 项的 σ 用 0.7，与手掌朝向的 0.4 不同：kernel 或实参若把两者弄混，第 13 项就对不上（代码审查第 1 条）。
O2：第 13 项权重 1：分数比上游正好多 frame_dt × 第 13 项。
O3：ReconBodyObjective.set_weights 收课程给的 12 个上游名字，第 13 个补上对象自己的权重；课程写了第 13 个就用写的；少名字照旧报错。
O4：CurriculumStage.weights_for：名单里有 object_orientation 而课程没写时补环境变量给的权重；名单里没有它时（别的目标函数）与上游相同。
O5：真实的 configs/curriculum/recon_body_flash_sac.yaml 实例化后 9 段逐段 weights_for：前 12 个等于这一段的映射，第 13 个等于环境变量。
O6：V2D_OBJ_ORI_WEIGHT 为负或非有限、V2D_OBJ_ORI_STD 不是正的有限数时报错；默认是 0 与 0.4。
O7：按源码核对 ReconBodyObjective.evaluate 里 wp.launch(compose_recon_body_objective, …) 的实参与 kernel 形参一一对应：个数相同，
    object_target_quat_w 对 self.command.body_target_quat_w、object_orientation_std 对 self.object_orientation_std 等（代码审查第 2 条；
    O1、O2 用测试自己的实参表，碰不到类里那一份）。
最后一行 OO_KERNELS_OK，或断言失败退出。
"""
import importlib.util
import math
import os
import sys

import numpy as np
import warp as wp

UPSTREAM = "/repo/third_party/video_to_data/robotic_grounding/flash_chord/src/flash_chord/objectives/recon_body.py"
CURRICULUM_YAML = "/workspace/src/flash_chord/configs/curriculum/recon_body_flash_sac.yaml"
DEVICE = "cpu"
W, B, Q, F = 64, 5, 12, 10   # 世界、每个世界的刚体（骨盆 0、手腕 1 与 2、物体 4）、每个世界的坐标数、参考帧数
OBJ, STDS, DT = 4, (0.3, 0.4, 1.0, 0.05, 0.2, 0.4), 0.02
ORI_STD = 0.7   # 第 13 项的 σ，故意与 STDS 里手掌朝向的 0.4 不同
WEIGHTS12 = (-300.0, -1.0e-4, -1.0e-6, -1.0e-3, 1.0, 1.0, 5.0, 1.0, 1.0, 1.0, 1.0, 5.0)


def load_upstream():
    spec = importlib.util.spec_from_file_location("v2d_recon_body_upstream", UPSTREAM)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def unit(q):
    return (q / np.linalg.norm(q, axis=-1, keepdims=True)).astype(np.float32)


def qmul(a, b):   # xyzw
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)


def make_inputs(rng):
    body = np.concatenate([rng.normal(scale=0.5, size=(W * B, 3)), unit(rng.normal(size=(W * B, 4)))], 1).astype(np.float32)
    target = unit(rng.normal(size=(W, 4)))
    body[0 * B + OBJ, 3:] = target[0]                       # 世界 0：与目标完全相同
    body[1 * B + OBJ, 3:] = -target[1]                      # 世界 1：互为相反数（同一个朝向）
    s = math.sqrt(0.5)
    body[2 * B + OBJ, 3:] = unit(qmul(target[2], np.array([0.0, 0.0, s, s])))   # 世界 2：绕 z 差 90°
    ref = rng.normal(scale=0.5, size=(F, Q)).astype(np.float32)
    ref[:, 3:7] = unit(rng.normal(size=(F, 4)))
    return {
        "body_q": body, "joint_q": rng.normal(scale=0.5, size=W * Q).astype(np.float32),
        "timestep": rng.integers(0, F + 3, size=W).astype(np.int32),
        "episode_start_frame": rng.integers(0, F, size=W).astype(np.int32),
        "terminated": rng.integers(0, 2, size=W).astype(np.int32),
        "action_rate_l2": rng.random(W).astype(np.float32), "action_l2": rng.random(W).astype(np.float32),
        "force_closure": rng.random(W).astype(np.float32),
        "reference_joint_q_offset": rng.normal(scale=0.1, size=W * Q).astype(np.float32), "reference_joint_q": ref,
        "reference_wrist_pos_w": rng.normal(scale=0.5, size=(F * 2, 3)).astype(np.float32),
        "reference_wrist_quat_w": unit(rng.normal(size=(F * 2, 4))),
        "wrist_local_pos": rng.normal(scale=0.05, size=(2, 3)).astype(np.float32),
        "wrist_local_quat": unit(rng.normal(size=(2, 4))),
        "object_target_pos_w": rng.normal(scale=0.5, size=(W, 3)).astype(np.float32), "object_target_quat_w": target,
    }


def launch(module, d, weights, patched, ori_std=0.4):
    def arr(x, dtype):
        return wp.array(x, dtype=dtype, device=DEVICE)

    n = 13 if patched else 12
    terms = wp.zeros(W * n, dtype=wp.float32, device=DEVICE)
    score = wp.zeros(W, dtype=wp.float32, device=DEVICE)
    obj_quat = [arr(d["object_target_quat_w"], wp.quat)] if patched else []
    ori = [ori_std] if patched else []
    wp.launch(
        module.compose_recon_body_objective,
        dim=W,
        inputs=[
            arr(d["body_q"], wp.transform), arr(d["joint_q"], wp.float32), arr(d["timestep"], wp.int32),
            arr(d["episode_start_frame"], wp.int32), arr(d["terminated"], wp.int32), arr(d["action_rate_l2"], wp.float32),
            arr(d["action_l2"], wp.float32), arr(d["force_closure"], wp.float32), arr(d["reference_joint_q_offset"], wp.float32),
            wp.array(d["reference_joint_q"], dtype=wp.float32, device=DEVICE), arr(d["reference_wrist_pos_w"], wp.vec3),
            arr(d["reference_wrist_quat_w"], wp.quat), arr(np.array([1, 2], dtype=np.int32), wp.int32),
            arr(d["wrist_local_pos"], wp.vec3), arr(d["wrist_local_quat"], wp.quat),
            arr(np.arange(7, 12, dtype=np.int32), wp.int32), arr(np.array([7, 8, 9], dtype=np.int32), wp.int32),
            arr(np.full(5, -0.3, dtype=np.float32), wp.float32), arr(np.full(5, 0.3, dtype=np.float32), wp.float32),
            OBJ, arr(d["object_target_pos_w"], wp.vec3), *obj_quat, 0, F, B, Q, 5, 3, *STDS, *ori,
            arr(np.asarray(weights, dtype=np.float32), wp.float32), DT,
        ],
        outputs=[terms, score],
        device=DEVICE,
    )
    return terms.numpy().reshape(W, n), score.numpy()


def expected_term(d, std=0.4):
    a = d["body_q"][np.arange(W) * B + OBJ, 3:].astype(np.float64)
    b = d["object_target_quat_w"].astype(np.float64)
    theta = 2.0 * np.arccos(np.clip(np.abs(np.sum(a * b, axis=1)), 0.0, 1.0))
    return np.exp(-(theta ** 2) / std ** 2), theta


def test_o1_o2(upstream, patched) -> None:
    d = make_inputs(np.random.default_rng(11))
    t_old, s_old = launch(upstream, d, WEIGHTS12, patched=False)
    t_new, s_new = launch(patched, d, WEIGHTS12 + (0.0,), patched=True, ori_std=ORI_STD)
    assert t_old.tobytes() == np.ascontiguousarray(t_new[:, :12]).tobytes(), "O1：前 12 项不逐位相同"
    assert s_old.tobytes() == s_new.tobytes(), "O1：权重 0 时分数不逐位相同"
    want, theta = expected_term(d, ORI_STD)
    err = np.abs(t_new[:, 12] - want).max()
    assert err < 1e-5, f"O1：第 13 项与 numpy 差 {err}"
    assert t_new[0, 12] > 0.99999 and t_new[1, 12] > 0.99999, "O1：与目标相同或互为相反数时第 13 项不是 1"
    assert abs(theta[2] - math.pi / 2) < 1e-4 and abs(t_new[2, 12] - math.exp(-(math.pi / 2) ** 2 / ORI_STD ** 2)) < 1e-5, "O1：差 90° 的世界不对"
    _, s_one = launch(patched, d, WEIGHTS12 + (1.0,), patched=True, ori_std=ORI_STD)
    gap = np.abs((s_one - s_old) - DT * t_new[:, 12]).max()
    assert gap < 1e-5, f"O2：权重 1 时分数多出的部分与 frame_dt × 第 13 项差 {gap}"
    print(f"O1/O2：{W} 个世界，前 12 项与分数逐位相同；第 13 项与 numpy 最大差 {err:.2e}，范围 {t_new[:, 12].min():.3g}–{t_new[:, 12].max():.6f}；"
          f"权重 1 时分数差与 frame_dt × 第 13 项最大差 {gap:.2e}")


def test_o3(rb) -> None:
    obj = object.__new__(rb.ReconBodyObjective)
    obj.weights = wp.zeros(13, dtype=wp.float32, device=DEVICE)
    obj.object_orientation_weight = 0.75
    m12 = {name: float(i + 1) for i, name in enumerate(rb.RECON_BODY_OBJECTIVE_TERM_NAMES)}
    obj.set_weights(m12)
    assert obj.weights.numpy().tolist() == [float(i + 1) for i in range(12)] + [0.75], "O3：12 个名字时第 13 个没补上"
    obj.set_weights(dict(m12, object_orientation=2.0))
    assert obj.weights.numpy()[12] == 2.0, "O3：课程写了第 13 个时没用写的"
    try:
        obj.set_weights({k: v for k, v in m12.items() if k != "progress"})
    except ValueError:
        pass
    else:
        raise AssertionError("O3：少一个上游名字没报错")
    assert obj.term_names == rb.RECON_BODY_OBJECTIVE_TERM_NAMES + ("object_orientation",)
    print("O3：set_weights 补第 13 个、课程写了就用写的、少名字报错")


def test_o4_o6() -> None:
    from flash_chord.lifecycle.curriculum import CurriculumStage
    from flash_chord.objectives import recon_body as rb
    from flash_chord.v2d_extras import object_orientation_params
    names13 = rb.RECON_BODY_OBJECTIVE_TERM_NAMES + ("object_orientation",)
    m12 = {name: float(i + 1) for i, name in enumerate(rb.RECON_BODY_OBJECTIVE_TERM_NAMES)}
    stage = CurriculumStage(voc_scale=1.0, objective_weights=m12)
    os.environ.pop("V2D_OBJ_ORI_WEIGHT", None)
    os.environ.pop("V2D_OBJ_ORI_STD", None)
    assert object_orientation_params() == (0.0, 0.4), "O6：默认不是 0 与 0.4"
    assert stage.weights_for(names13) == tuple(m12.values()) + (0.0,), "O4：环境变量没设时第 13 个不是 0"
    os.environ["V2D_OBJ_ORI_WEIGHT"] = "1.0"
    assert stage.weights_for(names13) == tuple(m12.values()) + (1.0,), "O4：环境变量 1.0 时第 13 个不对"
    other = CurriculumStage(voc_scale=1.0, objective_weights={"a": 1.0, "b": 2.0})
    assert other.weights_for(("a", "b")) == (1.0, 2.0), "O4：别的目标函数受了影响"
    for env, value in (("V2D_OBJ_ORI_WEIGHT", "-1"), ("V2D_OBJ_ORI_WEIGHT", "nan"), ("V2D_OBJ_ORI_STD", "0"), ("V2D_OBJ_ORI_STD", "inf")):
        saved = os.environ.get(env)
        os.environ[env] = value
        try:
            object_orientation_params()
        except ValueError:
            pass
        else:
            raise AssertionError(f"O6：{env}={value} 没报错")
        finally:
            if saved is None:
                os.environ.pop(env, None)
            else:
                os.environ[env] = saved
    print("O4/O6：weights_for 补第 13 个、别的目标函数不变；默认 0 与 0.4；非法值报错")


def test_o5() -> None:
    from omegaconf import OmegaConf
    from flash_chord.configuration import instantiate_typed
    from flash_chord.lifecycle.curriculum import FixedCurriculum
    from flash_chord.objectives import recon_body as rb
    os.environ["V2D_OBJ_ORI_WEIGHT"] = "1.0"
    names13 = rb.RECON_BODY_OBJECTIVE_TERM_NAMES + ("object_orientation",)
    cur = instantiate_typed(OmegaConf.load(CURRICULUM_YAML), FixedCurriculum)   # 同训练与评测的实例化方式（_convert_="all"）
    assert len(cur.stages) == 9, f"O5：课程不是 9 段：{len(cur.stages)}"
    for i, stage in enumerate(cur.stages):
        assert "object_orientation" not in stage.objective_weights, f"O5：第 {i + 1} 段的配置里出现了第 13 个名字"
        got = stage.weights_for(names13)
        want = tuple(float(stage.objective_weights[n]) for n in rb.RECON_BODY_OBJECTIVE_TERM_NAMES) + (1.0,)
        assert got == want, f"O5：第 {i + 1} 段 {got} ≠ {want}"
    print(f"O5：{CURRICULUM_YAML} 的 9 段逐段 weights_for，第 13 个都是 1.0，前 12 个等于这一段的映射")


def test_o7(rb) -> None:
    import ast
    import inspect
    import textwrap
    kernel = rb.compose_recon_body_objective
    func = getattr(kernel, "func", None)
    params = list(inspect.signature(func).parameters) if func else [a.label for a in kernel.adj.args]
    tree = ast.parse(textwrap.dedent(inspect.getsource(rb.ReconBodyObjective.evaluate)))
    calls = [c for c in ast.walk(tree) if isinstance(c, ast.Call) and getattr(c.func, "attr", "") == "launch"
             and c.args and ast.unparse(c.args[0]) == "compose_recon_body_objective"]
    assert len(calls) == 1, f"O7：evaluate 里 launch compose_recon_body_objective 的地方有 {len(calls)} 处"
    kw = {k.arg: k.value for k in calls[0].keywords}
    actual = [ast.unparse(e) for e in kw["inputs"].elts + kw["outputs"].elts]
    assert len(actual) == len(params), f"O7：实参 {len(actual)} 个、形参 {len(params)} 个"
    m = dict(zip(params, actual))
    want = {"body_q": "state.body_q", "object_body_id": "self.object_body_id", "object_target_pos_w": "self.command.body_target_pos_w",
            "object_target_quat_w": "self.command.body_target_quat_w", "pelvis_body_id": "self.pelvis_body_id",
            "object_position_std": "self.config.object_position_std", "palm_position_std": "self.config.palm_position_std",
            "palm_orientation_std": "self.config.palm_orientation_std", "object_orientation_std": "self.object_orientation_std",
            "weights": "self.weights", "frame_dt": "self.frame_dt", "terms": "self.terms", "score": "self.score"}
    bad = {k: (m.get(k), v) for k, v in want.items() if m.get(k) != v}
    assert not bad, f"O7：类里 wp.launch 的实参与 kernel 形参对不上（实际、应为）：{bad}"
    print(f"O7：类里 launch 的 {len(actual)} 个实参与 kernel 形参一一对应（核了 {len(want)} 个关键位置）")


def main() -> None:
    wp.init()
    from flash_chord.objectives import recon_body as patched
    test_o1_o2(load_upstream(), patched)
    test_o3(patched)
    test_o7(patched)
    test_o4_o6()
    test_o5()
    print("OO_KERNELS_OK")


if __name__ == "__main__":
    main()
