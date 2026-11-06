"""非有限防护的端到端闸 T6（守 docs/contracts/policy.md 的 POLICY-I006；端到端地覆盖 I001、I003、I004）。计划 agent-doc/plan-ep7-eval-and-guards-2026-10-02.md
第 5 版的 B.4。包一层运行未改动的训练入口 scripts/train_flash_sac.py（参数与 stages/policy/fc_train.sh 相同：G1 配方、ep7 的 Tier 2 参考、种子 42），
只多一处：第 <注入步> 次 RLEnv.step 之前，经 wp.launch 把世界 7 的关节速度全写成 NaN（同 nfg_env_probe.py 的 t2a）。训练机上、仓库根执行：
  V2D_GPU=1 V2D_FC_SRC=<fc_src/v040_p185_nfg> scripts/dev/record ep7/nfg_t6_<机器> scripts/dev/in-flash-chord \
    "python /repo/stages/policy/nfg_train_inject.py /repo/runs/nfg_t6/<名字>"
默认 2048 个世界、1300 次采集（回放 409,600 条在第 200 次采集满，第 300 次之后注入，再跑 1000 次），W&B offline（只写输出目录、不联网、不传 checkpoint）。
跑完之后在同一进程里核：最后的 policy 文件全部有限，打印 T6_POLICY_FINITE。事件行（NONFINITE_GUARD_EVENT）与停训行（NONFINITE_GUARD_STOP）由 runner 打在
标准输出里；W&B 本地记录用 stages/policy/wandb_local_history.py 解析。输出目录已存在就退出。最后一行 NFG_T6_DONE。
"""
import argparse
import runpy
import sys
from pathlib import Path

import numpy as np
import warp as wp

TRAIN = "/workspace/scripts/train_flash_sac.py"
PARQUET = ("/repo/runs/motion/tier2/whole_body/soma/sequence_id=2026-05-29_15-20-43_iron_round_table_02/"
           "robot_name=g1/data.parquet")


@wp.kernel
def write_span(values: wp.array(dtype=wp.float32), start: int, value: float):
    i = wp.tid()
    values[start + i] = value


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--worlds", type=int, default=2048)
    ap.add_argument("--collections", type=int, default=1300)
    ap.add_argument("--inject-after", type=int, default=300)
    ap.add_argument("--world", type=int, default=7)
    a = ap.parse_args()
    out = Path(a.out)
    if out.exists():
        print(f"{out} 已存在，不覆盖", flush=True)
        return 2

    from flash_chord.envs import rl
    from flash_chord.lifecycle import termination

    assert "nonfinite" in Path(termination.__file__).read_text(), f"挂的不是打了防护的副本：{termination.__file__}"
    calls = {"n": 0, "injected_at": None}
    original_step = rl.RLEnv.step

    def step(self, action=None):
        calls["n"] += 1
        if calls["n"] == a.inject_after + 1:
            nv = self.model.joint_dof_count // self.world_count
            wp.launch(write_span, dim=nv, inputs=[self.state_0.joint_qd, a.world * nv, float("nan")], device=self.device)
            calls["injected_at"] = calls["n"]
            print(f"T6_INJECT 第 {calls['n']} 次 RLEnv.step 之前，世界 {a.world} 的 {nv} 个关节速度写成 NaN "
                  f"（此前环境步数 {a.inject_after * self.world_count}）", flush=True)
        return original_step(self, action)

    rl.RLEnv.step = step
    steps = a.collections * a.worlds
    name = out.name
    sys.argv = [
        TRAIN, "experiment=g1_recon_body_flash_sac", f"task.parquet='{PARQUET}'", f"scene.world_count={a.worlds}",
        f"training.total_environment_steps={steps}", "reset.seed=42", f"output_dir={out}",
        "logging.mode=offline", "logging.project=v2d-selftest", f"logging.experiment={name}", f"logging.run_name={name}",
        "logging.upload_checkpoints=false", "logging.upload_training_state=false", f"hydra.run.dir={out}/hydra",
    ]
    print(f"T6：{a.worlds} 个世界、{a.collections} 次采集（{steps} 步），第 {a.inject_after} 次采集之后注入世界 {a.world}；参数 {sys.argv[1:]}", flush=True)
    try:
        runpy.run_path(TRAIN, run_name="__main__")
    except SystemExit as exc:
        if exc.code not in (None, 0):
            raise
    print(f"训练入口返回；RLEnv.step 共调用 {calls['n']} 次，注入在第 {calls['injected_at']} 次之前", flush=True)
    if calls["injected_at"] is None:
        print("没有注入（采集次数不够）", flush=True)
        return 1

    from safetensors.numpy import load_file

    policies = sorted(out.glob("policy_*.safetensors"), key=lambda p: int(p.stem.split("_")[-1]))
    if not policies:
        print(f"{out} 下没有 policy 文件", flush=True)
        return 1
    tensors = load_file(str(policies[-1]))
    bad = [k for k, v in tensors.items() if np.issubdtype(v.dtype, np.floating) and not np.isfinite(v).all()]
    print(f"最后的 policy {policies[-1].name}：{len(tensors)} 个张量，非有限的 {bad[:10]}", flush=True)
    if bad:
        return 1
    print("T6_POLICY_FINITE", flush=True)
    print("NFG_T6_DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
