#!/usr/bin/env python3
"""从已有队列里某条 run 的那一行派生 r3o（变体 A 加物体朝向项）的冒烟、正式队列、期望表与导出清单，一次一段（只用标准库）。
计划 agent-doc/plan-t3auc1-2026-11-03.md 第 1 节：行首（run 名之前）加 V2D_OBJ_ORI_WEIGHT=1.0 V2D_OBJ_ORI_STD=0.4，run 名 -r3a- → -r3o-，
正式行的 W&B 标签换成段名；其余字段逐字照抄。源写 <源队列名>:<run>，例如 ml40_t3qb2n12:t3-ep12-w2048-cap10m-r3a-gb5b2n12hdyn7-ml40。
源后面加 +10m（只许 400 万的源）：去掉 training.replay.capacity=4000000、run 名 -cap4m- → -cap10m-、期望表 capacity 写 10000000，
run 名最后一段机器名换成 --machine（ep0 gb4 的 1000 万臂由 mf4090 那一行派生到 ml40）。
输出 stages/policy/queues/<机器>_<段名>{.txt,_smoke.txt,.expect.tsv} 与 stages/evaluate/exports/cand_t3_<段名>_<机器>.tsv；目标已存在就拒绝。
用法：python3 stages/policy/derive_r3o_rows.py --seg <段名> --machine <机器> [--root <仓库根>] <源>[+10m] …   最后一行 DERIVE_R3O_OK <段名> <条数>
"""

import argparse
import re
import sys
from pathlib import Path

ORI = "V2D_OBJ_ORI_WEIGHT=1.0 V2D_OBJ_ORI_STD=0.4"
CAP4 = "training.replay.capacity=4000000"


def fail(msg):
    print(f"DERIVE_R3O_FAIL {msg}")
    sys.exit(1)


def rows(path, key):
    """path 里第 key 列（空格分隔的 token）等于 run 的非注释行，要恰好一条。"""
    if not path.is_file():
        fail(f"缺 {path}")
    return [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln and not ln.startswith("#")]


def pick(lines, token, where):
    hit = [ln for ln in lines if token in ln.split()]
    if len(hit) != 1:
        fail(f"{where} 里 {token} 有 {len(hit)} 行，要恰好 1 行")
    return hit[0]


def new_name(run, ten, machine):
    if run.count("-r3a-") != 1 or "-r3o-" in run:
        fail(f"{run} 要恰好有一个 -r3a-、没有 -r3o-")
    out = run.replace("-r3a-", "-r3o-")
    if ten:
        if out.count("-cap4m-") != 1:
            fail(f"+10m 只许 400 万的源：{run}")
        out = out.replace("-cap4m-", "-cap10m-")
        out = re.sub(r"-[a-z0-9]+$", f"-{machine}", out)
    elif not out.endswith(f"-{machine}"):
        fail(f"{run} 的机器名不是 {machine}（不加 +10m 时机器不变）")
    return out


def queue_line(line, old_tok, new_tok, seg, ten, formal):
    toks = line.split(" ")
    if toks.count(old_tok) != 1 or "V2D_OBJ_ORI_" in line:
        fail(f"源行不合要求：{line[:80]}")
    i = toks.index(old_tok)
    toks[i] = new_tok
    toks[i:i] = ORI.split(" ")
    if formal:
        tags = [k for k, t in enumerate(toks) if t.startswith("V2D_WANDB_TAG=")]
        if len(tags) != 1:
            fail(f"正式行要恰好一个 V2D_WANDB_TAG：{line[:80]}")
        toks[tags[0]] = f"V2D_WANDB_TAG={seg}"
    if ten:
        if toks.count(CAP4) != 1:
            fail(f"+10m 的源行要恰好一个 {CAP4}")
        toks.remove(CAP4)
    elif any(t.startswith("training.replay.capacity=") for t in toks) and CAP4 not in toks:
        fail("源行的 buffer 覆盖不认识")
    return " ".join(toks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", required=True)
    ap.add_argument("--machine", required=True)
    ap.add_argument("--root", default=".")
    ap.add_argument("srcs", nargs="+")
    a = ap.parse_args()
    root = Path(a.root)
    qd, ed = root / "stages/policy/queues", root / "stages/evaluate/exports"
    outs = {k: qd / f"{a.machine}_{a.seg}{k}" for k in (".txt", "_smoke.txt", ".expect.tsv")}
    cand = ed / f"cand_t3_{a.seg}_{a.machine}.tsv"
    for p in list(outs.values()) + [cand]:
        if p.exists():
            fail(f"{p} 已存在")
    formal, smoke, expect, cands, header, srcs = [], [], [], [], None, []
    for spec in a.srcs:
        ten = spec.endswith("+10m")
        base, _, run = spec.removesuffix("+10m").partition(":")
        if not run:
            fail(f"源要写 <源队列名>:<run>：{spec}")
        m = re.match(r"t3-ep(\d+)-", run)
        if not m:
            fail(f"只派生 Tier 3 的 run：{run}")
        new = new_name(run, ten, a.machine)
        f_src = pick(rows(qd / f"{base}.txt", run), run, f"{base}.txt")
        s_src = pick(rows(qd / f"{base}_smoke.txt", run), f"smoke-{run}", f"{base}_smoke.txt")
        e_lines = rows(qd / f"{base}.expect.tsv", run)
        header = header or e_lines[0]
        if e_lines[0] != header:
            fail("几份源期望表的表头不同")
        e_src = [ln for ln in e_lines[1:] if ln.split("\t")[0] == run]
        if len(e_src) != 1:
            fail(f"{base}.expect.tsv 里 {run} 有 {len(e_src)} 行")
        cols = e_src[0].split("\t")
        hdr = header.split("\t")
        if cols[1] != f"smoke-{run}":
            fail(f"期望表的 smoke 列不是 smoke-{run}")
        cols[0], cols[1] = new, f"smoke-{new}"
        if ten:
            ci = hdr.index("capacity")
            if cols[ci] != "4000000":
                fail("+10m 的源期望表 capacity 不是 4000000")
            cols[ci] = "10000000"
        formal.append(queue_line(f_src, run, new, a.seg, ten, True))
        smoke.append(queue_line(s_src, f"smoke-{run}", f"smoke-{new}", a.seg, ten, False))
        expect.append("\t".join(cols))
        steps = f_src.split(" ")[f_src.split(" ").index(run) + 3]
        cands.append(f"3\t{int(m.group(1))}\t{new}\t{steps}")
        srcs.append(f"{base}:{run}{' +10m' if ten else ''}")
    note = (f"# {a.seg}: r3o derived by stages/policy/derive_r3o_rows.py (plan agent-doc/plan-t3auc1-2026-11-03.md) from "
            + ", ".join(srcs) + "; only the two V2D_OBJ_ORI_ variables, -r3a- -> -r3o-, the formal W&B tag"
            + (" and, for +10m rows, the 10M buffer and the machine name" if any(s.endswith("+10m") for s in srcs) else "") + " differ.")
    outs[".txt"].write_text(note + "\n" + "\n".join(formal) + "\n", encoding="utf-8")
    outs["_smoke.txt"].write_text(note + "\n" + "\n".join(smoke) + "\n", encoding="utf-8")
    outs[".expect.tsv"].write_text(header + "\n" + "\n".join(expect) + "\n", encoding="utf-8")
    cand.write_text(f"# {a.seg} candidate export (derive_r3o_rows.py; plan agent-doc/plan-t3auc1-2026-11-03.md): tier episode run steps\n"
                    + "\n".join(cands) + "\n", encoding="utf-8")
    for p in list(outs.values()) + [cand]:
        print(p)
    print(f"DERIVE_R3O_OK {a.seg} {len(formal)}")


if __name__ == "__main__":
    main()
