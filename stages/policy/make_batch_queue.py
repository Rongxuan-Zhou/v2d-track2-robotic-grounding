"""按 episode 序号生成一段批量训练的期望表与冒烟 / 正式队列（逐帧回放、官方 G1 配方；只用标准库，Mac 上跑）。

Tier 1（默认）：序列号取自 results/batch_summary_v040.tsv（Tier 1 行），源帧数取自 results/day2/sequence_stats_tier1.txt，
只有地面的序列（results/day2/empty_support_tier1.txt）加 scene.support=false。
Tier 2（--tier 2，计划 agent-doc/plan-t2-batch-2026-10-01.md）：序列号与帧数取自 results/batch_summary_v040_t2v54.tsv 的 Tier 2 行
（v5.4 净室参考重定向后 parquet 的帧数），不读 Tier 1 的统计。支撑面取自两台训练机各自落盘的
results/t2/support_kinds_tier2_{mf4090,ml40}.txt（stages/policy/support_kinds.py；两份必须逐行一致）：empty 加 scene.support=false，
带 scale 的方盒（v0.4.0 会建成 1 m 立方体）拒绝排队。parquet 在 runs/motion/tier2 下，run 名 t2-…，正式 W&B tag t2batch，
另加 V2D_KEEP_STATES=1（续训不用，只留最新一份 state，省 mf4090 的磁盘）。
两档格式相同：冒烟 4,096,000 步、W&B project v2d-selftest、tag smoke；正式 349,999,104 步；都带
V2D_MAX_RETRY=0 V2D_GPU_APPS=1 V2D_WANDB_UPLOAD_CKPT=0。ml40 用配方默认的 1000 万 buffer（不写覆盖），mf4090 写 400 万覆盖。
方盒重训（--box185，计划 agent-doc/plan-box6-and-eval-runtime-2026-10-01.md）：ep1、ep7、ep28 两档各一条，同一段里可以混排两档，
ep 写成 t<档位>:<序号>。只接受这三条带 scale 方盒的序列（Tier 2 另按 Tier 2 的支撑面清单核），支撑面开着；run 名加 -p185
（与第 1 批 v0.4.0 几何下发散的旧目录区分），正式行 tag box185、V2D_KEEP_STATES=1。训练要挂 #185 补丁副本：V2D_FC_SRC 由链的环境给
（同时设 V2D_ALLOW_FC_SRC=1），不写进队列行。
用法：python3 stages/policy/make_batch_queue.py [--tier 2] <ml40|mf4090> <段名，如 batch1a> <计划文件> <ep 序号…>
      python3 stages/policy/make_batch_queue.py --box185 <ml40|mf4090> <段名> <计划文件> <t1:1 t2:28 …>
  写出 stages/policy/queues/<机器>_<段名>.expect.tsv、…_smoke.txt、….txt；已存在就拒绝覆盖。V2D_QUEUE_DIR 可改输出目录（测试用）。
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CAP = {"ml40": (10000000, "cap10m", None), "mf4090": (4000000, "cap4m", "training.replay.capacity=4000000")}
SMOKE_ENV = "V2D_WANDB_PROJECT=v2d-selftest V2D_WANDB_TAG=smoke V2D_MAX_RETRY=0 V2D_GPU_APPS=1 V2D_WANDB_UPLOAD_CKPT=0"
FULL_ENV = {1: "V2D_WANDB_TAG=batch V2D_MAX_RETRY=0 V2D_GPU_APPS=1 V2D_WANDB_UPLOAD_CKPT=0",
            2: "V2D_WANDB_TAG=t2batch V2D_MAX_RETRY=0 V2D_GPU_APPS=1 V2D_WANDB_UPLOAD_CKPT=0 V2D_KEEP_STATES=1"}
BOX_ENV = "V2D_WANDB_TAG=box185 V2D_MAX_RETRY=0 V2D_GPU_APPS=1 V2D_WANDB_UPLOAD_CKPT=0 V2D_KEEP_STATES=1"
BOX_EPS = {1, 7, 28}   # 带 xformOp:scale 的方盒支撑面：v0.4.0 建成 1 m 立方体，#185（上游 8c778f9）之后按标注尺寸建


def tier1_tables() -> tuple:
    """(ep → 序列, 序列 → 源帧数, 只有地面的序列集合, 带 scale 方盒的序列集合)。Tier 1 沿用第 1 批的来源，方盒不在这里拦。"""
    seq_of = {}
    for line in (ROOT / "results/batch_summary_v040.tsv").read_text().splitlines()[1:]:
        f = line.split("\t")
        if f[0] == "1":
            seq_of[int(f[1])] = f[2]
    frames = {}
    for line in (ROOT / "results/day2/sequence_stats_tier1.txt").read_text().splitlines():
        f = line.split("\t")
        if len(f) >= 12 and f[1].isdigit():
            frames[f[0]] = int(f[1])
    empty = {l.strip() for l in (ROOT / "results/day2/empty_support_tier1.txt").read_text().splitlines() if l.strip()}
    return seq_of, frames, empty, set()


def support_kinds(path: Path) -> dict:
    """support_kinds.py 落盘结果里的 序列 → (形状清单, 有没有带 scale 的方盒, md5)；跳过 record 的头与注释。"""
    out = {}
    for line in path.read_text().splitlines():
        f = line.split("\t")
        if len(f) == 4 and f[0] != "sequence_id" and not f[0].startswith("#"):
            out[f[0]] = (f[1], f[2] == "yes", f[3])
    return out


def tier2_tables() -> tuple:
    seq_of, frames = {}, {}
    for line in (ROOT / "results/batch_summary_v040_t2v54.tsv").read_text().splitlines()[1:]:
        f = line.split("\t")
        if f[0] == "2":
            seq_of[int(f[1])] = f[2]
            frames[f[2]] = int(f[3])
    kinds = {h: support_kinds(ROOT / f"results/t2/support_kinds_tier2_{h}.txt") for h in ("mf4090", "ml40")}
    if kinds["mf4090"] != kinds["ml40"] or len(kinds["mf4090"]) != 30:
        raise SystemExit("两台训练机的 Tier 2 支撑面清单不一致或不足 30 条，先查 results/t2/support_kinds_tier2_*.txt")
    k = kinds["mf4090"]
    return seq_of, frames, {s for s, v in k.items() if v[0] == "empty"}, {s for s, v in k.items() if v[1]}


def box185(argv: list) -> int:
    """方盒重训的队列：两档可混排，ep 写成 t<档位>:<序号>，只接受 ep1、ep7、ep28。"""
    if len(argv) < 4 or argv[0] not in CAP:
        raise SystemExit(__doc__)
    machine, label, plan, *tokens = argv
    capacity, tag, cap_override = CAP[machine]
    rows = []
    for tok in tokens:
        if not (len(tok) > 3 and tok[0] == "t" and tok[1] in "12" and tok[2] == ":" and tok[3:].isdigit()):
            raise SystemExit(f"--box185 的 ep 要写成 t<档位>:<序号>（档位 1 或 2）：{tok}")
        rows.append((int(tok[1]), int(tok[3:])))
    if len(set(rows)) != len(rows):
        raise SystemExit("同一条 ep 写了两次")
    tables = {}
    for tier, ep in rows:
        if ep not in BOX_EPS:
            raise SystemExit(f"--box185 只排带 scale 方盒的 ep1、ep7、ep28，不排 ep{ep}")
        if tier not in tables:
            tables[tier] = tier1_tables() if tier == 1 else tier2_tables()
        seq_of, frames, empty, boxes = tables[tier]
        if seq_of[ep] in empty or (tier == 2 and seq_of[ep] not in boxes):
            raise SystemExit(f"t{tier} ep{ep}（{seq_of[ep]}）的支撑面清单里不是带 scale 的方盒，核对后再排")
    q = Path(os.environ.get("V2D_QUEUE_DIR", ROOT / "stages/policy/queues"))
    out = {k: q / f"{machine}_{label}{k}" for k in (".expect.tsv", "_smoke.txt", ".txt")}
    for p in out.values():
        if p.exists():
            raise SystemExit(f"{p} 已存在，拒绝覆盖")
    header = (f"# 方盒重训 {label}（计划 {plan}）：训练挂 #185 补丁副本（链的环境给 V2D_ALLOW_FC_SRC=1 V2D_FC_SRC=<副本>，不写进队列行），"
              f"官方 G1 配方、逐帧回放、{capacity // 10000} 万 buffer（{machine}）；方盒支撑面开着。")
    expect = ["run\tsmoke\tsequence_id\tplayback\tsrc_frames\tsupport\tpos_tol\tori_tol\tcapacity\tseed\txeval"]
    smoke_lines, full_lines = [header], [header]
    for tier, ep in rows:
        seq_of, frames, _, _ = tables[tier]
        seq = seq_of[ep]
        run = f"t{tier}-ep{ep}-w2048-{tag}-p185-{machine}"
        parquet = f"/repo/runs/motion/tier{tier}/whole_body/soma/sequence_id={seq}/robot_name=g1/data.parquet"
        extra = f" {cap_override}" if cap_override else ""
        expect.append(f"{run}\tsmoke-{run}\t{seq}\tframe\t{frames[seq]}\ttrue\t0.015\t0.086\t{capacity}\t42\tno")
        smoke_lines.append(f"{SMOKE_ENV} smoke-{run} {parquet} 2048 4096000{extra}")
        full_lines.append(f"{BOX_ENV} {run} {parquet} 2048 349999104{extra}")
    out[".expect.tsv"].write_text("\n".join(expect) + "\n")
    out["_smoke.txt"].write_text("\n".join(smoke_lines) + "\n")
    out[".txt"].write_text("\n".join(full_lines) + "\n")
    for p in out.values():
        print(f"写出 {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}")
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if argv[:1] == ["--box185"]:
        return box185(argv[1:])
    tier = 1
    if argv[:1] == ["--tier"]:
        tier, argv = int(argv[1]), argv[2:]
    if tier not in FULL_ENV or len(argv) < 4 or argv[0] not in CAP:
        raise SystemExit(__doc__)
    machine, label, plan, *eps = argv
    capacity, tag, cap_override = CAP[machine]
    seq_of, frames, empty, boxes = tier1_tables() if tier == 1 else tier2_tables()
    q = Path(os.environ.get("V2D_QUEUE_DIR", ROOT / "stages/policy/queues"))
    out = {k: q / f"{machine}_{label}{k}" for k in (".expect.tsv", "_smoke.txt", ".txt")}
    for p in out.values():
        if p.exists():
            raise SystemExit(f"{p} 已存在，拒绝覆盖")
    ref = "v5.4 净室参考（runs/motion/tier2）、" if tier == 2 else ""
    header = (f"# Tier {tier} 批量 {label}（计划 {plan}）：{ref}官方 G1 配方、逐帧回放、{capacity // 10000} 万 buffer（{machine}）；"
              "只有地面的序列 scene.support=false。\n")
    expect = ["run\tsmoke\tsequence_id\tplayback\tsrc_frames\tsupport\tpos_tol\tori_tol\tcapacity\tseed\txeval"]
    smoke_lines, full_lines = [header.rstrip("\n")], [header.rstrip("\n")]
    for ep in (int(e) for e in eps):
        seq = seq_of[ep]
        if seq in boxes:
            raise SystemExit(f"ep{ep}（{seq}）是带 scale 的方盒支撑面：v0.4.0 会建成 1 m 立方体，等组委会答复再排")
        floor = seq in empty
        run = f"t{tier}-ep{ep}-w2048-{tag}-{machine}"
        parquet = f"/repo/runs/motion/tier{tier}/whole_body/soma/sequence_id={seq}/robot_name=g1/data.parquet"
        extra = " ".join(x for x in (cap_override, "scene.support=false" if floor else None) if x)
        expect.append(f"{run}\tsmoke-{run}\t{seq}\tframe\t{frames[seq]}\t{'false' if floor else 'true'}\t0.015\t0.086\t{capacity}\t42\tno")
        smoke_lines.append(f"{SMOKE_ENV} smoke-{run} {parquet} 2048 4096000" + (f" {extra}" if extra else ""))
        full_lines.append(f"{FULL_ENV[tier]} {run} {parquet} 2048 349999104" + (f" {extra}" if extra else ""))
    out[".expect.tsv"].write_text("\n".join(expect) + "\n")
    out["_smoke.txt"].write_text("\n".join(smoke_lines) + "\n")
    out[".txt"].write_text("\n".join(full_lines) + "\n")
    for p in out.values():
        print(f"写出 {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
