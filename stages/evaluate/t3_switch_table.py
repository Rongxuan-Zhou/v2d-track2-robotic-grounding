"""Tier 3 换集的逐集并列表：现选与候选的 4096 世界值（取自 gb3_switch.py 的记录）与导出那 1 个世界的值（取自两份导出 JSON）（Mac 上、仓库根执行，只用标准库）。
计划 agent-doc/plan-t3-gb3-switch-2026-10-17.md 第 4 节第 3 条。对照的都是各自的重建参考，不是干净真值；Tier 3 的导出世界值也只是代理量。
用法：python3 stages/evaluate/t3_switch_table.py <现选导出目录> <候选导出目录> <gb3_switch 记录>…   最后一行 T3_SWITCH_TABLE_DONE
"""

import json
import re
import sys
from pathlib import Path

LINE = re.compile(r"^ep(\d+) (\S+)（现选 (\S+)：MPPE ([\d.]+) cm、SP-SR ([\d.]+)、MP-SR ([\d.]+)）(?:［[^］]*］)?：新 MPPE ([\d.]+) cm、SP-SR ([\d.]+)、MP-SR ([\d.]+)、ADD-AUC ([\d.]+)；规则 S（[^）]*）：(\S+)$")


def exported(d: Path, ep: int, run: str):
    j = json.loads((d / f"episode_{ep:06d}.json").read_text())
    if f"/{run}/policy_" not in str(j.get("checkpoint", "")):
        sys.exit(f"{d} 的 ep{ep} 不是 {run} 的导出：{j.get('checkpoint')}")
    m = j["metrics"]
    return m["spider_sr_uncentered"], m["maniptrans_sr"], m["mppe_cm"], m["add_auc"]


def main():
    cur, cand, recs = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
    rows = []
    for r in recs:
        for line in Path(r).read_text(encoding="utf-8").splitlines():
            m = LINE.match(line)
            if m:
                rows.append(m.groups())
    if not rows:
        sys.exit("记录里没有可读的 ep 行")
    print("集\t判\t现选 run\t4096 世界 MPPE / SP-SR / MP-SR\t导出 SP / MP / MPPE / AUC\t候选 run\t4096 世界 MPPE / SP-SR / MP-SR / AUC\t导出 SP / MP / MPPE / AUC")
    for ep, crun, orun, om, osp, omp, nm, nsp, nmp, nauc, verdict in sorted(rows, key=lambda g: int(g[0])):
        e = int(ep)
        ce = cand / f"episode_{e:06d}.json"
        o = exported(cur, e, orun)
        n = exported(cand, e, crun) if ce.is_file() else None
        fmt = lambda x: f"{x[0]:.0f} / {x[1]:.0f} / {x[2]:.2f} / {x[3]:.3f}"  # noqa: E731
        print(f"ep{e}\t{verdict}\t{orun}\t{om} / {osp} / {omp}\t{fmt(o)}\t{crun}\t{nm} / {nsp} / {nmp} / {nauc}\t{fmt(n) if n else '（没拉回）'}")
    print("T3_SWITCH_TABLE_DONE")


if __name__ == "__main__":
    main()
