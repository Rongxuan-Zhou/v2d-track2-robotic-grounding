"""从清单里现有 run 的队列行派生 r3o 段（计划 agent-doc/plan-t1r3o-2026-10-31.md 第 3 版）。

make_var_queue.py 从 10/9 清单绕回非变体 A 的基线再派生：ep5、ep15 的基线没有期望表行，ep0、6、27、29 清单用的是 seed 43、基线覆盖里
已带种子，都生成不了。这里直接取清单 run 自己的正式行、冒烟行与期望表行，只改三处：
  1. run 名（与 smoke- 名）里的 -r3a- 换成 -r3o-；
  2. V2D_WANDB_TAG 换成段名（冒烟行的 V2D_WANDB_TAG=smoke 不动）；
  3. 在 run 名之前插入 V2D_OBJ_ORI_WEIGHT=1.0 V2D_OBJ_ORI_STD=0.4。
其余逐词照抄（容量、种子、支撑面、变体 A 两项），所以新 run 与对照只差物体朝向项。每条来源 run 在 stages/policy/queues/ 里
必须正好有一种正式行、一种冒烟行、一种期望表行（去掉 V2D_WANDB_TAG 后比），否则退出。
重生成同一段之前先删掉它的四份输出（正式、冒烟、期望表、导出清单），否则新 run 已在队列里会被拒绝。
用法（仓库根）：python3 stages/policy/make_clone_queue.py <段名> <机器> <来源 run> [<来源 run> ...]
"""
import re
import sys
from pathlib import Path

ORI = ["V2D_OBJ_ORI_WEIGHT=1.0", "V2D_OBJ_ORI_STD=0.4"]
PLAN = "agent-doc/plan-t1r3o-2026-10-31.md"


def one(paths, pick, what):
    found = {}
    for p in paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.startswith("#") or not line.strip():
                continue
            if pick(line):
                found.setdefault(re.sub(r"V2D_WANDB_TAG=\S+ ?", "", line), (p, line))
    if len(found) != 1:
        sys.exit(f"{what}：找到 {len(found)} 种：{[str(v[0]) for v in found.values()]}")
    return next(iter(found.values()))


def rename(run):
    if "-r3a-" not in run or not run.startswith("t1-"):
        sys.exit(f"来源 run 要是 Tier 1 变体 A（-r3a-）：{run}")
    return run.replace("-r3a-", "-r3o-", 1)


def derive(line, src, dst, tag):
    t = line.split()
    if ORI[0].split("=")[0] in line or ORI[1].split("=")[0] in line:
        sys.exit(f"来源行已带物体朝向变量：{src}")
    if "objective.object_position_std=0.05" not in t or "termination.object_orientation_threshold=0.6" not in t:
        sys.exit(f"来源行不是变体 A：{src}")
    i = t.index(src)
    head = [f"V2D_WANDB_TAG={tag}" if (x.startswith("V2D_WANDB_TAG=") and tag) else x for x in t[:i]]
    return " ".join(head + ORI + [dst] + t[i + 1:])


def main(argv):
    if len(argv) < 3:
        sys.exit(__doc__)
    seg, machine, srcs = argv[0], argv[1], argv[2:]
    if not re.fullmatch(r"t1z[a-z][a-z0-9]*", seg):
        sys.exit(f"段名非法：{seg}")
    q = Path("stages/policy/queues")
    formal = [p for p in q.glob(f"{machine}_*.txt") if not p.name.endswith("_smoke.txt")]
    smoke = list(q.glob(f"{machine}_*_smoke.txt"))
    expect = list(q.glob(f"{machine}_*.expect.tsv"))
    rows = {"": [], "_smoke": [], ".expect": [], "cand": []}
    head_exp = None
    for src in srcs:
        dst = rename(src)
        if any(p.read_text(encoding="utf-8").count(f" {dst} ") for p in formal):
            sys.exit(f"{dst} 已在别的队列里")
        _, f = one(formal, lambda x, s=src: f" {s} " in f" {x} ", f"{src} 的正式行")
        _, s = one(smoke, lambda x, s=src: f" smoke-{s} " in f" {x} ", f"smoke-{src} 的冒烟行")
        pe, e = one(expect, lambda x, s=src: x.split("\t")[0] == s, f"{src} 的期望表行")
        h = pe.read_text(encoding="utf-8").splitlines()[0]
        if head_exp is None:
            head_exp = h
        elif h != head_exp:
            sys.exit("来源期望表的表头不一致")
        rows[""].append(derive(f, src, dst, seg))
        rows["_smoke"].append(derive(s, f"smoke-{src}", f"smoke-{dst}", None))
        cols = e.split("\t")
        if cols[1] != f"smoke-{src}":
            sys.exit(f"{src} 期望表行的冒烟列不符：{cols[1]}")
        rows[".expect"].append("\t".join([dst, f"smoke-{dst}"] + cols[2:]))
        ep = re.match(r"t1-ep(\d+)-", src).group(1)
        rows["cand"].append(f"1\t{ep}\t{dst}\t349999104")
    desc = (f"# Tier 1 r3o：清单 run 的行加物体朝向项（{'、'.join(srcs)} → -r3o-），其余逐词照抄；段 {seg}，{machine}；"
            f"计划 {PLAN} 第 3 版；r3o = 变体 A 加 {' '.join(ORI)}；训练挂 v040_p185_nfg_oo（链的环境给 V2D_ALLOW_FC_SRC=1 "
            f"V2D_FC_SRC=<副本>，不写进队列行）。make_clone_queue.py 生成，不要手改。")
    out = {q / f"{machine}_{seg}.txt": [desc] + rows[""],
           q / f"{machine}_{seg}_smoke.txt": [desc] + rows["_smoke"],
           q / f"{machine}_{seg}.expect.tsv": [head_exp] + rows[".expect"],
           Path("stages/evaluate/exports") / f"cand_t1_{seg}_{machine}.tsv": [desc.replace("。make_clone_queue.py", "。这是这一段的候选导出：档位 episode run 步数。make_clone_queue.py")] + rows["cand"]}
    for p, lines in out.items():
        p.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"写 {p}（{len(lines)} 行）")


if __name__ == "__main__":
    main(sys.argv[1:])
