#!/usr/bin/env python3
"""并排比几份 tools/t3_episode_remote.sh 记录里同一相对路径的 sha256（只读、只用标准库）。计划 agent-doc/plan-t3dyn-pilot-2026-10-14.md 第 1 节。

每份记录取“可读 … sha256 <hex> <相对路径>”行（parquet、元数据、支撑面）与“支撑面读不了、不存在或路径不合要求：<相对路径>”行（记成“没有”）。
按相对路径逐行打：各份记录里的 sha256 前 12 位（记录里没有这一行就打“—”），以及第一份与其余各份是否相同。
用法：python3 tools/t3_record_compare.py <标签>=<记录> <标签>=<记录> ...    例：hm=results/ops/t3_episodes_ml40_1014_hm4.txt dyn4=…
"""
import re
import sys
from pathlib import Path

OK = re.compile(r"^可读 .* sha256 ([0-9a-f]{64}) (\S+)$")
NO = re.compile(r"^支撑面读不了、不存在或路径不合要求：(\S+)$")


def read(path: Path) -> dict:
    out = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        m = OK.match(line.strip())
        if m:
            out[m.group(2)] = m.group(1)
            continue
        m = NO.match(line.strip())
        if m:
            out[m.group(1)] = "没有"
    return out


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) < 2 or not all("=" in a for a in args):
        raise SystemExit("用法：t3_record_compare.py <标签>=<记录> <标签>=<记录> ...")
    recs = [(a.split("=", 1)[0], read(Path(a.split("=", 1)[1]))) for a in args]
    paths = sorted(set().union(*(r for _, r in recs)))
    print("相对路径\t" + "\t".join(lab for lab, _ in recs) + "\t与 " + recs[0][0] + " 相同")
    for p in paths:
        vals = [r.get(p, "—") for _, r in recs]
        short = [v[:12] if len(v) == 64 else v for v in vals]
        same = [lab for (lab, _), v in zip(recs[1:], vals[1:]) if v == vals[0] and len(v) == 64]
        print(f"{p}\t" + "\t".join(short) + "\t" + ("、".join(same) or "无"))
    print(f"COMPARE_DONE（{len(paths)} 个路径，{len(recs)} 份记录）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
