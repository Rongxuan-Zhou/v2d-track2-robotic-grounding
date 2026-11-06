"""从已训过的 run 照抄队列行，只换参考：Tier 3 动力学修复版试点（计划 agent-doc/plan-t3dyn-pilot-2026-10-14.md）。只用标准库。

每条新 run 对应一条源 run：best_hm 上同一集、同一配置的 t3-ep<N>-w2048-cap4m-r3a-hm-mf4090。源 run 在 stages/policy/queues/ 下的期望表、
冒烟队列、正式队列里必须各恰好一行（注释行不算）。新行由源行改三处得到，别处一字不改：
  1. run 名：期望表前两列、冒烟行的 smoke-<run>、正式行的 <run> 换成 t3-ep<N>-w2048-cap4m-r3a-<版本>-<机器>；
  2. 参考：冒烟与正式行 parquet 开头的 /repo/runs/motion/tier3_hm/ 换成 /repo/runs/motion/tier3_<版本>/（整行恰好一处，序列名不变）；
  3. W&B 标签：正式行的 V2D_WANDB_TAG=<源标签> 换成 V2D_WANDB_TAG=t3dyn（冒烟行是 smoke，不动）。
buffer、世界数、支撑面覆盖、变体 A 两项、种子都照源行；换到 ml40 也不改（400 万 buffer，与 best_hm 那批同配置，计划第 2 节）。
改完核：正式行步数 349999104；冒烟行与正式行的 parquet、世界数与覆盖相同；新 run 名带 -<版本>-（precheck_expect.sh 要求），
不与别的队列文件里已有的 run 重名。任何一条不过就退出，什么文件都不写。
段（段名以 t3p、t3q 开头：arm_var_chain.sh 按前缀认机器，t3p* 是 mf4090，t3q* 是 ml40）：
  t3pdyn  mf4090：ep15、ep3、ep16 的 best_dyn4（runs/motion/tier3_dyn4）
  t3qdyn  ml40：ep12 的 best_dyn4 与 best_dyn4s（runs/motion/tier3_dyn4、runs/motion/tier3_dyn4s）
  t3qdyn_s43  ml40：同 t3qdyn 两条的 seed 43（试点计划第 4 节 Q2 要补的种子；计划 agent-doc/plan-overnight-2026-10-15.md）。run 名在机器名前加 -s43，
          冒烟与正式行在变体 A 两项之前加 reset.seed=43 training.seed=43（同 make_var_queue.py --seed），期望表种子列写 43；源行已有种子覆盖就拒绝。
输出同 make_var_queue.py：stages/policy/queues/<机器>_<段名>{.expect.tsv,_smoke.txt,.txt} 与 stages/evaluate/exports/cand_t3_<段名>_<机器>.tsv。
--check：不写文件，核仓库里这一段的四个文件与现在生成的逐字相同，并逐条列出新行与源行不同的 token（期望表按列），最后一行 COPY_CHECK_OK 或 COPY_CHECK_FAIL。
用法（仓库根）：python3 stages/policy/make_copy_queue.py t3pdyn|t3qdyn|t3qdyn_s43 [--root <仓库根>] [--check]
测试：stages/policy/test_make_copy_queue.py。
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_trial_queue import HEADER, STEPS  # noqa: E402

PLAN = "agent-doc/plan-t3dyn-pilot-2026-10-14.md"
SRC_RUN = "t3-ep{ep}-w2048-cap4m-r3a-hm-mf4090"
SRC_MOTION = "/repo/runs/motion/tier3_hm/"
NEW_RUN = "t3-ep{ep}-w2048-cap4m-r3a-{ver}-{machine}"
NEW_RUN_SEED = "t3-ep{ep}-w2048-cap4m-r3a-{ver}-s{seed}-{machine}"
VAR_FIRST = "objective.object_position_std="
NEW_MOTION = "/repo/runs/motion/tier3_{ver}/"
TAG = "t3dyn"
SEGS = {
    "t3pdyn": {"machine": "mf4090", "items": [(15, "dyn4"), (3, "dyn4"), (16, "dyn4")],
               "desc": "Tier 3 动力学修复版试点：ep15、ep3、ep16 的参考换成 best_dyn4（mf4090，配置照抄 best_hm 那批：变体 A seed 42、"
                       "buffer 400 万、支撑面按 best_hm）"},
    "t3qdyn": {"machine": "ml40", "items": [(12, "dyn4"), (12, "dyn4s")],
               "desc": "Tier 3 动力学修复版试点：ep12 的参考换成 best_dyn4 与 best_dyn4s 各一条（ml40，配置照抄 best_hm 在 mf4090 上那条："
                       "变体 A seed 42、buffer 400 万、支撑面按 best_hm）"},
    "t3qdyn_s43": {"machine": "ml40", "items": [(12, "dyn4"), (12, "dyn4s")], "seed": 43,
                   "desc": "Tier 3 动力学修复版试点补种子：ep12 的 best_dyn4 与 best_dyn4s 各一条 seed 43（ml40，配置同 t3qdyn，只加种子两项；"
                           "试点计划第 4 节 Q2）"},
}
ENV = re.compile(r"^[A-Z_][A-Z0-9_]*=")


def lines_of(files) -> list:
    """(文件, 行) 列表：去掉注释行与空行。"""
    out = []
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.lstrip().startswith("#"):
                out.append((f, line))
    return out


def one(pairs, pred, what: str):
    hits = [(f, x) for f, x in pairs if pred(x)]
    if len(hits) != 1:
        raise SystemExit(f"make_copy_queue: {what} 应当恰好一行，找到 {len(hits)} 行（{', '.join(sorted({f.name for f, _ in hits})) or '无'}）")
    return hits[0][1]


def rewrite(line: str, old_run: str, new_run: str, ver: str, tag: str = None, seed: int = None) -> list:
    """队列行按 token 改 run 名、parquet 目录与（给了 tag 时）W&B 标签，给了 seed 时在变体 A 两项之前加种子两项；
    返回新 token 列表与 (parquet, 世界数, 步数, 覆盖)。"""
    tok = line.split()
    if " ".join(tok) != line:
        raise SystemExit(f"make_copy_queue: {old_run} 的队列行里有多余的空白，不照抄")
    j = 0
    while j < len(tok) and ENV.match(tok[j]):
        j += 1
    if tok[j:j + 1] != [old_run] or len(tok) < j + 4:
        raise SystemExit(f"make_copy_queue: 队列行的 run 不是 {old_run}：{line[:120]}")
    if line.count(SRC_MOTION) != 1 or not tok[j + 1].startswith(SRC_MOTION):
        raise SystemExit(f"make_copy_queue: {old_run} 的 parquet 不在 {SRC_MOTION} 下，或整行不止一处：{tok[j + 1]}")
    tok[j] = new_run
    tok[j + 1] = NEW_MOTION.format(ver=ver) + tok[j + 1][len(SRC_MOTION):]
    tags = [i for i in range(j) if tok[i].startswith("V2D_WANDB_TAG=")]
    if len(tags) != 1:
        raise SystemExit(f"make_copy_queue: {old_run} 的队列行里 V2D_WANDB_TAG 不是恰好一个")
    if tag is not None:
        tok[tags[0]] = f"V2D_WANDB_TAG={tag}"
    elif tok[tags[0]] != "V2D_WANDB_TAG=smoke":
        raise SystemExit(f"make_copy_queue: {old_run} 的冒烟行标签不是 smoke：{tok[tags[0]]}")
    if seed is not None:
        if any(x.startswith(("reset.seed=", "training.seed=")) for x in tok[j + 4:]):
            raise SystemExit(f"make_copy_queue: {old_run} 的队列行已有种子覆盖，不再加")
        at = [i for i in range(j + 4, len(tok)) if tok[i].startswith(VAR_FIRST)]
        if len(at) != 1:
            raise SystemExit(f"make_copy_queue: {old_run} 的队列行里 {VAR_FIRST} 不是恰好一个，不知道种子插在哪")
        tok[at[0]:at[0]] = [f"reset.seed={seed}", f"training.seed={seed}"]
    return tok, (tok[j + 1], tok[j + 2], tok[j + 3], tok[j + 4:])


def changed(a: list, b: list) -> str:
    """两行 token（或列）不同之处：等长时写成“旧 → 新”；不等长时按 difflib 列出替换、插入与删除。"""
    if len(a) == len(b):
        return "；".join(f"{x} → {y}" for x, y in zip(a, b) if x != y) or "相同"
    import difflib

    out = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes():
        if op == "replace":
            out.append(f"{' '.join(a[i1:i2])} → {' '.join(b[j1:j2])}")
        elif op == "insert":
            out.append(f"插入 {' '.join(b[j1:j2])}")
        elif op == "delete":
            out.append(f"删去 {' '.join(a[i1:i2])}")
    return "；".join(out)


def make(part: str, root: Path, report: list = None) -> dict:
    if part not in SEGS:
        raise SystemExit(f"make_copy_queue: 段名只许 {'、'.join(SEGS)}：{part}")
    s, qdir = SEGS[part], root / "stages" / "policy" / "queues"
    stem = f"{s['machine']}_{part}"
    mine = {qdir / f"{stem}.expect.tsv", qdir / f"{stem}_smoke.txt", qdir / f"{stem}.txt"}
    smoke_files = sorted(f for f in qdir.glob("*_smoke.txt") if f not in mine)
    formal_files = sorted(f for f in qdir.glob("*.txt") if not f.name.endswith("_smoke.txt") and f not in mine)
    expect_files = sorted(f for f in qdir.glob("*.expect.tsv") if f not in mine)
    formal, smoke, expect = lines_of(formal_files), lines_of(smoke_files), lines_of(expect_files)
    exp, sm, fo, cand, seen = [HEADER], [], [], [], set()
    for ep, ver in s["items"]:
        seed = s.get("seed")
        src = SRC_RUN.format(ep=ep)
        new = (NEW_RUN_SEED.format(ep=ep, ver=ver, seed=seed, machine=s["machine"]) if seed is not None
               else NEW_RUN.format(ep=ep, ver=ver, machine=s["machine"]))
        if new in seen or f"-{ver}-" not in new:
            raise SystemExit(f"make_copy_queue: {new} 重复，或 run 名里没有 -{ver}-")
        seen.add(new)
        for pairs, name in ((formal, new), (smoke, f"smoke-{new}")):
            if any(name in x.split() for _, x in pairs):
                raise SystemExit(f"make_copy_queue: 别的队列文件里已有 {name}，不重名")
        if any(x.split("\t")[0] == new for _, x in expect):
            raise SystemExit(f"make_copy_queue: 别的期望表里已有 {new}，不重名")
        frow = one(formal, lambda x: src in x.split(), f"正式队列里 {src}")
        srow = one(smoke, lambda x: f"smoke-{src}" in x.split(), f"冒烟队列里 smoke-{src}")
        erow = one(expect, lambda x: x.split("\t")[0] == src, f"期望表里 {src}")
        ftok, (fpq, fw, fsteps, fover) = rewrite(frow, src, new, ver, TAG, seed)
        stok, (spq, sw, _, sover) = rewrite(srow, f"smoke-{src}", f"smoke-{new}", ver, seed=seed)
        if fsteps != STEPS or (fpq, fw, fover) != (spq, sw, sover):
            raise SystemExit(f"make_copy_queue: {src} 的正式步数不是 {STEPS}，或冒烟行与正式行的 parquet、世界数、覆盖不同")
        cols = orig = erow.split("\t")
        if len(cols) != len(HEADER.split("\t")) or cols[1] != f"smoke-{src}":
            raise SystemExit(f"make_copy_queue: 期望表里 {src} 这一行的列数或冒烟名不对：{erow}")
        if seed is not None:
            if cols[9] != "42":
                raise SystemExit(f"make_copy_queue: 期望表里 {src} 的种子列是 {cols[9]}，不是 42")
            cols = [*cols[:9], str(seed), *cols[10:]]
        exp.append("\t".join([new, f"smoke-{new}", *cols[2:]]))
        if report is not None:
            report += [f"{new}（源 {src}）", f"  正式行：{changed(frow.split(), ftok)}", f"  冒烟行：{changed(srow.split(), stok)}",
                       f"  期望表：{changed(orig, exp[-1].split(chr(9)))}"]
        fo.append(" ".join(ftok))
        sm.append(" ".join(stok))
        cand.append(f"3\t{ep}\t{new}\t{STEPS}")
    items = "、".join(f"ep{ep} {ver}" for ep, ver in s["items"])
    seeded = "" if s.get("seed") is None else f"，并在变体 A 两项之前加 reset.seed={s['seed']} training.seed={s['seed']}"
    head = (f"# {s['desc']}；这一段 {part}：{items}；计划 {PLAN}；源 run 是 best_hm 上同集的 {SRC_RUN.format(ep='<N>')}，只换 run 名、"
            f"参考目录与 W&B 标签{seeded}；训练挂 v040_p185_nfg（链的环境给 V2D_ALLOW_FC_SRC=1 V2D_FC_SRC=<副本>，不写进队列行）。"
            f"make_copy_queue.py 生成，不要手改。")
    return {
        qdir / f"{stem}.expect.tsv": "\n".join(exp) + "\n",
        qdir / f"{stem}_smoke.txt": "\n".join([head, *sm]) + "\n",
        qdir / f"{stem}.txt": "\n".join([head, *fo]) + "\n",
        root / "stages" / "evaluate" / "exports" / f"cand_t3_{part}_{s['machine']}.tsv":
            "\n".join([f"# {s['desc']}，段 {part} 的候选导出（计划 {PLAN}）：档位 episode run 步数。"
                       + (f"本段改种子：seed {s['seed']}。" if s.get("seed") is not None else "")
                       + "make_copy_queue.py 生成，不要手改。", *cand]) + "\n",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("part")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    if a.check:
        report = []
        files = make(a.part, Path(a.root), report)
        bad = [p for p, text in files.items() if not p.is_file() or p.read_text(encoding="utf-8") != text]
        print("\n".join(report))
        for p in files:
            print(f"{'不同或缺' if p in bad else '相同'} {p.relative_to(a.root)}")
        print("COPY_CHECK_FAIL" if bad else "COPY_CHECK_OK")
        return 1 if bad else 0
    for path, text in make(a.part, Path(a.root)).items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        print(f"写 {path.relative_to(a.root)}（{text.count(chr(10))} 行）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
