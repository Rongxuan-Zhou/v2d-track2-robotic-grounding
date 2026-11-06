#!/usr/bin/env python3
"""交上去的 Tier 1 CSV 里每集的 run：交的那 1 局的结果与同一 checkpoint 4096 世界的成功率并列，再估“导出 N 局、交成功的那局”
能多出几条成功（Mac 上跑，只读已落盘的结果，只用标准库）。

假设（都不是测出来的，只用来估量级）：
  1. N 局是同一分布里独立的 N 次，每局成功的概率等于 4096 世界的成功率 p（SP 用 spider_sr_uncentered，MP 用 maniptrans_sr）；
  2. 交的那局本来没成功的集，换成成功的一局后 AUC 与 MPPE 取现在 1 局里 MP 成功的集的中位数（同 t1_gap.py 的“像成功的”）。
两种算法：甲，只用交上去的那个 checkpoint；乙，MP 没成功的集换成候选里 4096 世界 MP-SR 最高的 checkpoint（select 记录的候选行）。
这种做法要先查规则（计划 agent-doc/plan-first-place-2026-10-14.md 第 1 节），这里只算数。
用法：python3 tools/t1_multiworld_estimate.py --select <select 记录> [--n 1,4,16,64] [--eps 14,19]
  --eps：乙只算这几集（例如“每集补 N 个新种子”只补 ep14、ep19 时）；N 读作每集独立抽几次（新种子或多导的局）。
"""
import argparse
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "stages" / "evaluate"))
from eval_table import row  # noqa: E402

CHOSEN = re.compile(r"^CHOSEN ep(\d+): (\S+)（现在 \S+）add_auc=(\S+) spider_sr=(\S+) maniptrans_sr=(\S+) mppe_cm=(\S+)")
CAND = re.compile(r"^  ep(\d+)\s+(\S+): add_auc=\S+ spider_sr=\S+ maniptrans_sr=\S+ mppe_cm=\S+ \| 4096 世界 MP-SR (\S+)")
N_EP = 30


def eval_file(run: str) -> Path:
    for d in (ROOT / "results" / "day2", ROOT / "results" / "ml40" / "day2"):
        p = d / f"eval_{run}_s349999104.txt"
        if p.exists():
            return p
    raise SystemExit(f"找不到 {run} 的 4096 世界评测")


def rates(run: str) -> tuple:
    r = row(eval_file(run))
    return float(r[4]), float(r[5])


def pbinom_at_least(ps: list, k: int) -> float:
    """独立伯努利之和至少为 k 的概率（泊松二项分布，动态规划）。"""
    dist = [1.0]
    for p in ps:
        new = [0.0] * (len(dist) + 1)
        for i, q in enumerate(dist):
            new[i] += q * (1 - p)
            new[i + 1] += q * p
        dist = new
    return sum(dist[k:])


def estimate(label: str, fails: list, base: dict, auc_ok: float, mppe_ok: float, ns: list) -> None:
    print(f"\n== {label}：MP 没成功、4096 世界 MP-SR 大于 0 的集：" +
          ("、".join(f"ep{e['ep']} {e['run']}（MP {e['p_mp']:.4f}，SP {e['p_sp']:.4f}）" for e in fails) or "没有"))
    for n in ns:
        pm = [1 - (1 - e["p_mp"]) ** n for e in fails]
        ps = [1 - (1 - e["p_sp"]) ** n if e["sp"] == 0.0 else 0.0 for e in fails]
        d_auc = sum(q * max(0.0, auc_ok - e["auc"]) for q, e in zip(pm, fails)) / N_EP
        d_mppe = sum(q * max(0.0, e["mppe"] - mppe_ok) for q, e in zip(pm, fails)) / N_EP
        print(f"N={n:<3d} 期望多出 MP 成功 {sum(pm):.2f} 条（至少多 1 条的概率 {pbinom_at_least(pm, 1):.3f}，"
              f"至少多 2 条 {pbinom_at_least(pm, 2):.3f}）；SP 成功 {sum(ps):.2f} 条；"
              f"AUC 期望 +{d_auc:.4f}（到 {base['auc'] + d_auc:.4f}）；MPPE 期望 −{d_mppe:.3f} cm（到 {base['mppe'] - d_mppe:.3f}）")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", required=True)
    ap.add_argument("--n", default="1,4,16,64")
    ap.add_argument("--eps")
    a = ap.parse_args()
    only = {int(x) for x in a.eps.split(",")} if a.eps else None
    eps, best = {}, {}
    for line in open(a.select, encoding="utf-8", errors="replace"):
        m = CAND.match(line)
        if m:
            ep, run, p = int(m.group(1)), m.group(2), float(m.group(3))
            if ep not in best or p > best[ep][1]:
                best[ep] = (run, p)
        m = CHOSEN.match(line)
        if m:
            ep, run = int(m.group(1)), m.group(2)
            auc, sp, mp, mppe = (float(x) for x in m.groups()[2:])
            p_mp, p_sp = rates(run)
            eps[ep] = {"ep": ep, "run": run, "auc": auc, "sp": sp, "mp": mp, "mppe": mppe, "p_mp": p_mp, "p_sp": p_sp}
    if len(eps) != N_EP:
        raise SystemExit(f"CHOSEN 行 {len(eps)} 条，要 {N_EP} 条")
    ok = [e for e in eps.values() if e["mp"] == 1.0]
    auc_ok, mppe_ok = statistics.median(e["auc"] for e in ok), statistics.median(e["mppe"] for e in ok)
    print("集\trun\t1局_SP\t1局_MP\t1局_AUC\t1局_MPPE\t4096_SP\t4096_MP")
    for ep in sorted(eps):
        e = eps[ep]
        print(f"{ep}\t{e['run']}\t{e['sp']:.0f}\t{e['mp']:.0f}\t{e['auc']:.3f}\t{e['mppe']:.2f}\t{e['p_sp']:.3f}\t{e['p_mp']:.3f}")
    base = {k: sum(e[k] for e in eps.values()) / N_EP for k in ("auc", "sp", "mp", "mppe")}
    print(f"\n现在（1 局）：AUC {base['auc']:.4f}，SP {base['sp'] * N_EP:.0f} 条，MP {base['mp'] * N_EP:.0f} 条，MPPE {base['mppe']:.3f} cm")
    print(f"“像成功的”取值：AUC {auc_ok:.3f}、MPPE {mppe_ok:.2f} cm（现在 1 局 MP 成功的 {len(ok)} 集的中位数）")
    ns = [int(x) for x in a.n.split(",")]
    fails = [e for _, e in sorted(eps.items()) if e["mp"] == 0.0 and e["p_mp"] > 0]
    estimate("甲（只用交上去的 checkpoint）", fails, base, auc_ok, mppe_ok, ns)
    fails2 = []
    for ep, e in sorted(eps.items()):
        if e["mp"] != 0.0 or (only is not None and ep not in only):
            continue
        run = best[ep][0] if ep in best and best[ep][1] > e["p_mp"] else e["run"]
        p_mp, p_sp = rates(run) if run != e["run"] else (e["p_mp"], e["p_sp"])
        if p_mp > 0:
            fails2.append({**e, "run": run, "p_mp": p_mp, "p_sp": p_sp})
    estimate("乙（MP 没成功的集换成候选里 4096 世界 MP-SR 最高的 checkpoint）", fails2, base, auc_ok, mppe_ok, ns)
    print("T1_MULTIWORLD_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
