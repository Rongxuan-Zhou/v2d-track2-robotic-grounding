"""非有限防护补丁的环境侧组件测试（K1 的 CPU 版与 K2；守 docs/contracts/policy.md 的 POLICY-I001、POLICY-I002、POLICY-I003）。在 FlashCHORD 容器里、CPU 上跑，挂打了防护补丁的副本：
  V2D_FC_SRC=<fc_src/v040_p185_nfg> scripts/dev/in-flash-chord "python /repo/stages/policy/nfg_test_kernels.py"
计划 agent-doc/plan-ep7-eval-and-guards-2026-10-02.md 的 B.4。

K1（CPU 版，合成数据）：同一组输入分别交给上游 v0.4.0 的 evaluate_recon_body_termination（从 /repo/third_party/video_to_data 里的源码按文件加载，
  #185 没有改这个文件）与打了补丁的版本。全部有限、离原点不远的世界：12 个输出逐位相同（覆盖冻结期内外、有失败有不失败）。
  任一刚体位姿含 NaN 或 inf、平移超过 1e3 m 的世界：被判骨盆失败（terminated 1、pelvis 1、palm 0、object 0），骨盆位置误差 -1，其余误差 0；
  平移恰好 999 m 的世界不算。GPU 上用真实 body_q 的那一半由 nfg_env_probe.py 做。
K2：clear_nonfinite_solver_rows 只清“在重置掩码里、且这一行有非有限值”的世界，整行清零；有限的行（含很大的有限值）与不在掩码里的世界一位不动。
  _solver_rows_to_clear：没有 mjw_data 返回空；act 宽度为 0 跳过；世界数对不上报错。
最后一行 NFG_KERNELS_OK 或报错退出。
"""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import warp as wp

UPSTREAM = "/repo/third_party/video_to_data/robotic_grounding/flash_chord/src/flash_chord/lifecycle/termination.py"
DEVICE = "cpu"


def load_upstream():
    spec = importlib.util.spec_from_file_location("v2d_termination_upstream", UPSTREAM)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def random_quats(rng, n):
    q = rng.normal(size=(n, 4)).astype(np.float32)
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def make_inputs(rng, worlds, bodies, frames=30):
    """合成一组终止判定的输入；物体刚体放在每个世界块的最后一个。"""
    body_pos = rng.normal(scale=0.5, size=(worlds * bodies, 3)).astype(np.float32)
    body_rot = random_quats(rng, worlds * bodies)
    ref = rng.normal(scale=0.5, size=(frames, 7)).astype(np.float32)
    ref[:, 3:7] = random_quats(rng, frames)
    hands = 2
    return {
        "body_q": np.concatenate([body_pos, body_rot], axis=1),
        "timestep": rng.integers(0, frames + 3, size=worlds).astype(np.int32),
        "episode_step": rng.integers(0, 120, size=worlds).astype(np.int32),
        "reference_joint_q": ref,
        "reference_wrist_pos_w": rng.normal(scale=0.5, size=(frames * hands, 3)).astype(np.float32),
        "reference_wrist_quat_w": random_quats(rng, frames * hands),
        "wrist_body_ids": np.array([3, 5], dtype=np.int32),
        "wrist_local_pos": rng.normal(scale=0.05, size=(hands, 3)).astype(np.float32),
        "wrist_local_quat": random_quats(rng, hands),
        "object_body_ids_w": np.array([w * bodies + bodies - 1 for w in range(worlds)], dtype=np.int32),
        "object_ref_pos_w": rng.normal(scale=0.5, size=(worlds, 3)).astype(np.float32),
        "object_ref_quat_w": random_quats(rng, worlds),
        "frames": frames,
        "hands": hands,
    }


def run_kernel(kernel, d, worlds, bodies):
    def arr(x, dtype):
        return wp.array(x, dtype=dtype, device=DEVICE)

    outs_i = [wp.zeros(worlds, dtype=wp.int32, device=DEVICE) for _ in range(5)]
    outs_f = [wp.zeros(worlds, dtype=wp.float32, device=DEVICE) for _ in range(6)]
    packed = wp.zeros(worlds * 9, dtype=wp.float32, device=DEVICE)
    wp.launch(
        kernel,
        dim=worlds,
        inputs=[
            arr(d["body_q"], wp.transform), arr(d["timestep"], wp.int32), arr(d["episode_step"], wp.int32),
            wp.array(d["reference_joint_q"], dtype=wp.float32, device=DEVICE),
            arr(d["reference_wrist_pos_w"], wp.vec3), arr(d["reference_wrist_quat_w"], wp.quat),
            arr(d["wrist_body_ids"], wp.int32), arr(d["wrist_local_pos"], wp.vec3), arr(d["wrist_local_quat"], wp.quat),
            arr(d["object_body_ids_w"], wp.int32), arr(d["object_ref_pos_w"], wp.vec3), arr(d["object_ref_quat_w"], wp.quat),
            0, d["frames"], bodies, d["hands"], 1, 50,
            0.70, 1.50, 0.15, 1.50, 0.10, 1.50, 1,
        ],
        outputs=[*outs_i, *outs_f, packed],
        device=DEVICE,
    )
    return [a.numpy() for a in (*outs_i, *outs_f, packed)]


NAMES = ["terminated", "truncated", "pelvis_done", "palm_done", "object_done", "pelvis_position_error",
         "pelvis_orientation_error", "palm_position_error", "palm_orientation_error", "object_position_error",
         "object_orientation_error", "packed_diagnostics"]


def test_k1(upstream, patched) -> None:
    rng = np.random.default_rng(7)
    worlds, bodies = 512, 9
    d = make_inputs(rng, worlds, bodies)
    old = run_kernel(upstream.evaluate_recon_body_termination, d, worlds, bodies)
    new = run_kernel(patched.evaluate_recon_body_termination, d, worlds, bodies)
    for name, a, b in zip(NAMES, old, new):
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes(), f"K1：{name} 在全部有限时不逐位相同"
    assert 0 < old[0].sum() < worlds, "合成数据要有失败也有不失败的世界"
    assert (d["episode_step"] <= 50).any() and (d["episode_step"] > 50).any()
    print(f"K1 CPU：{worlds} 个世界全部有限，12 个输出逐位相同（其中 {int(old[0].sum())} 个世界被判终止）")

    bad = {}
    q = d["body_q"].copy()
    q[3 * bodies + 0, 0] = np.nan          # 世界 3：骨盆平移 NaN
    q[10 * bodies + 6, 4] = np.inf         # 世界 10：一个手指刚体的旋转 inf（不是骨盆、手腕、物体）
    q[20 * bodies + bodies - 1, 1] = np.nan  # 世界 20：物体平移 NaN
    q[30 * bodies + 2, 2] = 2.0e3          # 世界 30：一个刚体离原点 2000 m（有限）
    q[40 * bodies + 2, 2] = 999.0          # 世界 40：999 m，不算
    for w in (3, 10, 20, 30):
        bad[w] = True
    d2 = dict(d, body_q=q)
    old2 = run_kernel(upstream.evaluate_recon_body_termination, d2, worlds, bodies)
    new2 = run_kernel(patched.evaluate_recon_body_termination, d2, worlds, bodies)
    out = dict(zip(NAMES, new2))
    for w in bad:
        assert out["terminated"][w] == 1 and out["pelvis_done"][w] == 1, w
        assert out["palm_done"][w] == 0 and out["object_done"][w] == 0, w
        assert out["pelvis_position_error"][w] == -1.0, w
        for name in NAMES[6:11]:
            assert out[name][w] == 0.0, (w, name)
        p = out["packed_diagnostics"][w * 9:(w + 1) * 9]
        assert p.tolist() == [1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0], (w, p.tolist())
    keep = np.array([w not in bad for w in range(worlds)])
    for name, a, b in zip(NAMES[:11], old2, new2):
        assert a[keep].tobytes() == b[keep].tobytes(), f"K1：没注入的世界 {name} 变了"
    pk_old, pk_new = old2[11].reshape(worlds, 9), new2[11].reshape(worlds, 9)
    assert pk_old[keep].tobytes() == pk_new[keep].tobytes()
    assert np.isfinite(new2[11]).all(), "坏世界的诊断输出必须全部有限"
    flagged_upstream = int(old2[2][3])
    print(f"K1 CPU：注入的 4 个世界都被判骨盆失败、误差写成 -1 与 0；999 m 那个不算；其余世界与上游逐位相同（上游对世界 3 的 pelvis_done 是 {flagged_upstream}）")


def test_k2(base) -> None:
    rng = np.random.default_rng(11)
    worlds, dofs = 6, 5
    v = rng.normal(size=(worlds, dofs)).astype(np.float32)
    v[1, 2] = np.nan      # 在掩码里、有 NaN：整行清零
    v[2, 0] = np.inf      # 不在掩码里：不动
    v[3, 4] = 1.0e30      # 在掩码里、很大但有限：不动
    v[4, 1] = -np.inf     # 在掩码里、有 -inf：整行清零
    mask = np.array([1, 1, 0, 1, 1, 0], dtype=np.int32)
    values = wp.array(v, dtype=wp.float32, device=DEVICE)
    wp.launch(base.clear_nonfinite_solver_rows, dim=worlds, inputs=[wp.array(mask, dtype=wp.int32, device=DEVICE)],
              outputs=[values], device=DEVICE)
    got = values.numpy()
    for w in range(worlds):
        if w in (1, 4):
            assert (got[w] == 0.0).all(), w
        else:
            assert got[w].tobytes() == v[w].tobytes(), w
    zero = wp.zeros((worlds, 0), dtype=wp.float32, device=DEVICE)
    good = wp.zeros((worlds, dofs), dtype=wp.float32, device=DEVICE)
    wrong = wp.zeros((worlds + 1, dofs), dtype=wp.float32, device=DEVICE)
    assert base._solver_rows_to_clear(SimpleNamespace(), worlds) == ()
    rows = base._solver_rows_to_clear(SimpleNamespace(mjw_data=SimpleNamespace(qacc_warmstart=good, act=zero)), worlds)
    assert len(rows) == 1 and rows[0] is good
    try:
        base._solver_rows_to_clear(SimpleNamespace(mjw_data=SimpleNamespace(qacc_warmstart=wrong)), worlds)
    except ValueError:
        pass
    else:
        raise AssertionError("世界数对不上应当报错")
    print("K2：只清在重置掩码里、含非有限值的行（整行清零）；很大的有限值与掩码外的世界不动；取行的函数三种情形都对")


def main() -> int:
    wp.init()
    from flash_chord.envs import base
    from flash_chord.lifecycle import termination as patched

    upstream = load_upstream()
    assert "nonfinite" not in Path(UPSTREAM).read_text(), "对照用的上游 termination.py 不应带防护"
    assert "nonfinite" in Path(patched.__file__).read_text(), f"挂的不是打了防护补丁的副本：{patched.__file__}"
    test_k1(upstream, patched)
    test_k2(base)
    print("NFG_KERNELS_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
