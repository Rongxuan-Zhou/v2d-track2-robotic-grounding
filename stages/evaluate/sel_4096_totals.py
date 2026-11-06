"""一份逐条选择清单的 4096 世界合计（只用标准库；训练机上、仓库根执行，只读候选目录里的 eval_<步数>_w4096.json）。

导出那 1 局是一次抽样，榜上的分数就是它；4096 世界是同一个策略的期望。rxzhou 10/14 定的是两种分数并列报、主要按 4096 世界判，
select_t1.py 只记了 4096 世界的 MP-SR，这里补全四项（MP-SR = maniptrans_sr，SP-SR = spider_sr_uncentered，同 eval_table.py）。
可以在命令行上改几集（ep<N>=<run>），同时报改前改后，用来比较几种选法。

用法：python3 stages/evaluate/sel_4096_totals.py <候选目录> <清单.tsv> [ep<N>=<run> …]
清单每行：档位 集号 run 步数 机器（select_t1.py --out-tsv 的格式）。
"""

import json
import sys
from pathlib import Path

KEYS = (("AUC", "add_auc"), ("SP-SR", "spider_sr_uncentered"), ("MP-SR", "maniptrans_sr"), ("MPPE", "mppe_cm"))


def load(cand: Path, run: str, step: str) -> dict:
    p = cand / run / f"eval_{step}_w4096.json"
    d = json.loads(p.read_text())
    if d.get("world_count") != 4096 or d.get("non_finite_worlds"):
        raise SystemExit(f"{p}：world_count {d.get('world_count')}，non_finite_worlds {d.get('non_finite_worlds')}")
    return {name: float(d["metrics"][k]) for name, k in KEYS}


def totals(rows: dict) -> str:
    return "  ".join(f"{name} {sum(r[name] for r in rows.values()) / len(rows):.6f}" for name, _ in KEYS)


def main():
    cand, sel = Path(sys.argv[1]), sys.argv[2]
    base = {}
    for line in open(sel):
        f = line.rstrip("\n").split("\t")
        if len(f) >= 4 and f[0] in ("1", "2", "3"):
            base[int(f[1])] = (f[2], f[3])
    if len(base) != 30:
        raise SystemExit(f"{sel} 只有 {len(base)} 集")
    over = {}
    for a in sys.argv[3:]:
        k, run = a.split("=", 1)
        ep = int(k[2:] if k.startswith("ep") else k)   # 不用 str.removeprefix（3.9 才有），旧解释器也能跑
        over[ep] = (run, base[ep][1])
    rows = {ep: load(cand, *base[ep]) for ep in sorted(base)}
    for ep in sorted(base):
        print(f"  ep{ep:<3} {base[ep][0]}: " + "  ".join(f"{n} {rows[ep][n]:.4f}" for n, _ in KEYS))
    print(f"清单 {sel}（4096 世界，30 集平均）：{totals(rows)}")
    if over:
        new = dict(rows)
        for ep, (run, step) in sorted(over.items()):
            new[ep] = load(cand, run, step)
            print(f"  改 ep{ep}: {base[ep][0]} → {run}  " + "  ".join(f"{n} {rows[ep][n]:.4f}→{new[ep][n]:.4f}" for n, _ in KEYS))
        print(f"改后（4096 世界，30 集平均）：{totals(new)}")
    print("SEL_4096_TOTALS_DONE")


if __name__ == "__main__":
    main()
