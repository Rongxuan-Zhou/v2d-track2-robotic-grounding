"""生成 Tier 1 变体 A 补跑（ep0、ep15、ep5，seed 42 与 43，mf4090）的期望表、冒烟队列、正式队列与导出清单。只用标准库。
计划 agent-doc/plan-t1-r3x-2026-10-08.md。

这 3 条现在交的是 ml40 上 1000 万 buffer 训的基线 seed 42（t1-ep<N>-w2048-cap10m-ml40）。新 run 在 mf4090 上训，buffer 400 万
（与第 3 步 Tier 1 试验相同，这张卡放不下 1000 万）：期望表行取 ml40 基线的 sequence_id、回放、帧数、支撑面、容差，buffer 列改成 4000000；
队列行取基线的 parquet、世界数与覆盖，前面加 training.replay.capacity=4000000，后面加种子与变体 A 两项。
基线行的查找与核对（期望表与队列各恰好一种、冒烟与正式一致、步数、没有种子与变体的键、种子 42）用 make_trial_queue.py 的 baseline()。
命名：t1-ep<N>-w2048-cap4m-r3a-mf4090（seed 42）、t1-ep<N>-w2048-cap4m-r3a-s43-mf4090（seed 43）。
顺序：先 3 条 seed 42（ep0、ep15、ep5），再 3 条 seed 43；链被截断时，每条 episode 先有一个变体。
输出：stages/policy/queues/mf4090_r3x{.expect.tsv,_smoke.txt,.txt} 与 stages/evaluate/exports/cand_t1_r3x_mf4090.tsv（export_chain.sh 的清单）。
用法（仓库根）：python3 stages/policy/make_r3x_queue.py [--root <仓库根>]
测试：stages/policy/test_make_r3x_queue.py。
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_trial_queue import HEADER, SEED43, SMOKE_STEPS, STEPS, VARIANT, baseline  # noqa: E402

PLAN = "agent-doc/plan-t1-r3x-2026-10-08.md"
EPISODES = [0, 15, 5]
BASE = "t1-ep{ep}-w2048-cap10m-ml40"
CAP = "4000000"
ENV_FORMAL = "V2D_WANDB_TAG=r3x V2D_MAX_RETRY=0 V2D_GPU_APPS=1 V2D_WANDB_UPLOAD_CKPT=0 V2D_KEEP_STATES=1"
ENV_SMOKE = "V2D_WANDB_PROJECT=v2d-selftest V2D_WANDB_TAG=smoke V2D_MAX_RETRY=0 V2D_GPU_APPS=1 V2D_WANDB_UPLOAD_CKPT=0"


def run_name(ep: int, seed: int) -> str:
    return f"t1-ep{ep}-w2048-cap4m-r3a-{'s43-' if seed == 43 else ''}mf4090"


def make(root: Path) -> dict:
    qdir = root / "stages" / "policy" / "queues"
    rows = {ep: baseline(qdir, "ml40", BASE.format(ep=ep)) for ep in EPISODES}
    for ep, (row, formal) in rows.items():
        keys = {o.split("=", 1)[0].lstrip("+") for o in formal["overrides"]}
        if row[8] != "10000000" or "training.replay.capacity" in keys:
            raise SystemExit(f"make_r3x_queue: {BASE.format(ep=ep)} 的 buffer 不是默认的 1000 万（期望表 {row[8]}，覆盖 {formal['overrides']}）")
    exp, smoke, formal_lines, cand = [HEADER], [], [], []
    for seed in (42, 43):
        for ep in EPISODES:
            row, formal = rows[ep]
            run = run_name(ep, seed)
            exp.append("\t".join([run, f"smoke-{run}", *row[2:8], CAP, str(seed), "no"]))
            over = " ".join([f"training.replay.capacity={CAP}", *formal["overrides"], *(SEED43 if seed == 43 else []), *VARIANT])
            tail = f"{formal['parquet']} {formal['worlds']}"
            formal_lines.append(f"{ENV_FORMAL} {run} {tail} {STEPS} {over}")
            smoke.append(f"{ENV_SMOKE} smoke-{run} {tail} {SMOKE_STEPS} {over}")
            cand.append(f"1\t{ep}\t{run}\t{STEPS}")
    head = (f"# Tier 1 变体 A 补跑：ep0、ep15、ep5 的 seed 42 与 seed 43（计划 {PLAN}，mf4090，buffer 400 万）；变体 A = {' '.join(VARIANT)}；"
            f"训练挂 v040_p185_nfg（链的环境给 V2D_ALLOW_FC_SRC=1 V2D_FC_SRC=<副本>，不写进队列行）。make_r3x_queue.py 生成，不要手改。")
    q = qdir / "mf4090_r3x"
    return {
        q.with_name("mf4090_r3x.expect.tsv"): "\n".join(exp) + "\n",
        q.with_name("mf4090_r3x_smoke.txt"): "\n".join([head, *smoke]) + "\n",
        q.with_name("mf4090_r3x.txt"): "\n".join([head, *formal_lines]) + "\n",
        root / "stages" / "evaluate" / "exports" / "cand_t1_r3x_mf4090.tsv":
            "\n".join([f"# Tier 1 变体 A 补跑的候选导出（计划 {PLAN}）：档位 episode run 步数。make_r3x_queue.py 生成，不要手改。", *cand]) + "\n",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    a = ap.parse_args(argv)
    for path, text in make(Path(a.root)).items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"写 {path.relative_to(a.root)}（{text.count(chr(10))} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
