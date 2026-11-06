"""生成第 3 步配方试验的期望表、冒烟队列、正式队列与试验清单（计划 agent-doc/plan-step3-recipe-trial-2026-10-04.md 第 2 版第 3 节）。只用标准库。

每条试验 run 都从它的基线 run（现在导出的那条，seed 42）派生：期望表行取基线的 sequence_id、回放、帧数、支撑面、容差、buffer，
队列行取基线的 parquet、世界数与覆盖（例如 scene.support=false、training.replay.capacity=4000000）；基线行在这台机器的
stages/policy/queues/<机器>_*.expect.tsv、<机器>_*.txt、<机器>_*_smoke.txt 里按 run 名找，找不到或找到互相不一致的就报错。
  A 段：5 条试验 episode 的变体 A seed 42，加对照 episode 的变体 A seed 42
  BC 段：每条试验 episode 的基线 seed 43、变体 A seed 43（按 episode 交错，段被截断时也是成对的）
命名：基线 t<档>-ep<N>-w2048-cap<…>-<机器> → 变体 …-r3a-<机器>、变体 seed 43 …-r3a-s43-<机器>、基线 seed 43 …-s43-<机器>。
seed 43 的行末加 reset.seed=43 training.seed=43；变体 A 的行末加 VARIANT 两项。训练副本由链的环境给，不写进队列行。
输出：stages/policy/queues/<机器>_r3t<档>{a,bc}{.expect.tsv,_smoke.txt,.txt} 与 stages/evaluate/trials/r3_t<档>_<机器>.json（trial_verdict.py 读）。
用法（仓库根）：python3 stages/policy/make_trial_queue.py --tier 2 --machine ml40 [--root <仓库根>]
测试：stages/policy/test_make_trial_queue.py。
"""
import argparse
import json
import sys
from pathlib import Path

PLAN = "agent-doc/plan-step3-recipe-trial-2026-10-04.md"
TRIALS = {   # (档位, 机器) → 试验 episode、对照 episode、基线 run 名的样式
    (1, "mf4090"): {"episodes": [6, 18, 20, 27, 29], "canary": 8, "base": "t1-ep{ep}-w2048-cap4m-mf4090"},
    (2, "ml40"): {"episodes": [8, 12, 20, 24, 26], "canary": 22, "base": "t2-ep{ep}-w2048-cap10m-ml40"},
}
VARIANT = ["objective.object_position_std=0.05", "termination.object_orientation_threshold=0.6"]
SEED43 = ["reset.seed=43", "training.seed=43"]
ENV_FORMAL = "V2D_WANDB_TAG=r3 V2D_MAX_RETRY=0 V2D_GPU_APPS=1 V2D_WANDB_UPLOAD_CKPT=0 V2D_KEEP_STATES=1"
ENV_SMOKE = "V2D_WANDB_PROJECT=v2d-selftest V2D_WANDB_TAG=smoke V2D_MAX_RETRY=0 V2D_GPU_APPS=1 V2D_WANDB_UPLOAD_CKPT=0"
HEADER = "run\tsmoke\tsequence_id\tplayback\tsrc_frames\tsupport\tpos_tol\tori_tol\tcapacity\tseed\txeval"
STEPS, SMOKE_STEPS = "349999104", "4096000"


def derived(base: str, kind: str) -> str:
    prefix, machine = base.rsplit("-", 1)
    return {"var42": f"{prefix}-r3a-{machine}", "var43": f"{prefix}-r3a-s43-{machine}", "base43": f"{prefix}-s43-{machine}"}[kind]


def find_one(rows: list, what: str) -> str:
    uniq = sorted(set(rows))
    if len(uniq) != 1:
        raise SystemExit(f"make_trial_queue: {what} 找到 {len(uniq)} 种：{uniq}")
    return uniq[0]


def parse_line(line: str) -> dict:
    tok = line.split()
    i = 0
    while i < len(tok) and "=" in tok[i] and tok[i].split("=", 1)[0].isupper():
        i += 1
    run, parquet, worlds, steps = tok[i:i + 4]
    return {"run": run, "parquet": parquet, "worlds": worlds, "steps": steps, "overrides": tok[i + 4:]}


def baseline(qdir: Path, machine: str, base: str) -> tuple:
    # 期望表行比较时不看最后一列 xeval：同一条 run 可能在两张表里各有一行、只差 xeval（如 t1-ep10-w2048-cap10m-ml40 在 day2 与 day2b，
    # 计划 agent-doc/plan-next-training-2026-10-12.md 审查 M2）。去掉 xeval 后仍不止一种就照旧报错。返回的行不含 xeval，调用方只用到第 0–9 列。
    hit = [ln for f in sorted(qdir.glob(f"{machine}_*.expect.tsv")) for ln in f.read_text().splitlines() if ln.split("\t")[0] == base]
    bad = [ln for ln in hit if len(ln.split("\t")) != len(HEADER.split("\t"))]
    if bad:
        raise SystemExit(f"make_trial_queue: {base} 的期望表行不是 {len(HEADER.split(chr(9)))} 列：{bad}")
    exp = [ln.rsplit("\t", 1)[0] for ln in hit]
    row = find_one(exp, f"{base} 的期望表行").split("\t")
    def qlines(files, run):
        return [ln.strip() for f in files for ln in f.read_text().splitlines()
                if not ln.lstrip().startswith("#") and run in ln.split()]
    formal_files = [f for f in sorted(qdir.glob(f"{machine}_*.txt")) if not f.name.endswith("_smoke.txt")]
    formal = parse_line(find_one(qlines(formal_files, base), f"{base} 的正式队列行"))
    smoke = parse_line(find_one(qlines(sorted(qdir.glob(f"{machine}_*_smoke.txt")), f"smoke-{base}"), f"smoke-{base} 的冒烟队列行"))
    if formal["steps"] != STEPS or smoke["steps"] != SMOKE_STEPS:
        raise SystemExit(f"make_trial_queue: {base} 的步数是 {formal['steps']} / {smoke['steps']}")
    if formal["parquet"] != smoke["parquet"] or formal["overrides"] != smoke["overrides"]:
        raise SystemExit(f"make_trial_queue: {base} 的冒烟与正式队列行 parquet 或覆盖不一致")
    keys = {o.split("=", 1)[0].lstrip("+") for o in formal["overrides"]}
    if keys & {"reset.seed", "training.seed", "objective.object_position_std", "termination.object_orientation_threshold"}:
        raise SystemExit(f"make_trial_queue: {base} 的基线覆盖里已有种子或变体的键：{formal['overrides']}")
    if row[9] != "42":
        raise SystemExit(f"make_trial_queue: {base} 的期望表种子是 {row[9]}，不是 42")
    return row, formal


def make(tier: int, machine: str, root: Path) -> dict:
    t = TRIALS[(tier, machine)]
    qdir = root / "stages" / "policy" / "queues"
    eps = t["episodes"] + [t["canary"]]
    base = {ep: t["base"].format(ep=ep) for ep in eps}
    rows = {ep: baseline(qdir, machine, base[ep]) for ep in eps}

    def entry(ep: int, kind: str):
        row, formal = rows[ep]
        run = derived(base[ep], kind)
        seed = "43" if kind in ("var43", "base43") else "42"
        extra = (SEED43 if seed == "43" else []) + (VARIANT if kind.startswith("var") else [])
        exp = "\t".join([run, f"smoke-{run}", *row[2:9], seed, "no"])
        tail = " ".join([formal["parquet"], formal["worlds"]])
        over = " ".join(formal["overrides"] + extra)
        f_line = f"{ENV_FORMAL} {run} {tail} {STEPS} {over}".rstrip()
        s_line = f"{ENV_SMOKE} smoke-{run} {tail} {SMOKE_STEPS} {over}".rstrip()
        return exp, s_line, f_line

    seg = {"a": [(ep, "var42") for ep in t["episodes"]] + [(t["canary"], "var42")],
           "bc": [(ep, k) for ep in t["episodes"] for k in ("base43", "var43")]}
    desc = {"a": f"A 段：5 条试验 episode 与对照 ep{t['canary']} 的变体 A seed 42",
            "bc": "BC 段：每条试验 episode 的基线 seed 43 与变体 A seed 43（段间门判 CONTINUE 才开）"}
    files = {}
    for s, items in seg.items():
        ents = [entry(ep, k) for ep, k in items]
        head = (f"# 第 3 步配方试验 {desc[s]}（计划 {PLAN} 第 2 版，Tier {tier}，{machine}）；变体 A = {' '.join(VARIANT)}；"
                f"训练挂 v040_p185_nfg（链的环境给 V2D_ALLOW_FC_SRC=1 V2D_FC_SRC=<副本>，不写进队列行）。make_trial_queue.py 生成，不要手改。")
        stem = f"{machine}_r3t{tier}{s}"
        files[qdir / f"{stem}.expect.tsv"] = "\n".join([HEADER] + [e[0] for e in ents]) + "\n"
        files[qdir / f"{stem}_smoke.txt"] = "\n".join([head] + [e[1] for e in ents]) + "\n"
        files[qdir / f"{stem}.txt"] = "\n".join([head] + [e[2] for e in ents]) + "\n"
    spec = {"plan": PLAN, "tier": tier, "machine": machine, "steps": int(STEPS), "worlds": 4096,
            "episodes": [{"ep": ep, "base42": base[ep], **{k: derived(base[ep], k) for k in ("var42", "base43", "var43")}}
                         for ep in t["episodes"]],
            "canary": {"ep": t["canary"], "base42": base[t["canary"]], "var42": derived(base[t["canary"]], "var42")}}
    files[root / "stages" / "evaluate" / "trials" / f"r3_t{tier}_{machine}.json"] = json.dumps(spec, ensure_ascii=False, indent=1) + "\n"
    return files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, required=True)
    ap.add_argument("--machine", required=True)
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    a = ap.parse_args(argv)
    if (a.tier, a.machine) not in TRIALS:
        raise SystemExit(f"make_trial_queue: 计划里没有 Tier {a.tier} 在 {a.machine} 上的试验")
    for path, text in make(a.tier, a.machine, Path(a.root)).items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"写 {path.relative_to(a.root)}（{text.count(chr(10))} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
