"""Tier 3 重交评估的读表（计划 agent-doc/plan-t3resub-2026-10-29.md 第 1 节 D1、D2、D4）。只读已落盘的记录，不用 Tier 1 的数据。

D1：每个换入的集，旧 run（已交清单 t3m1022a）与新 run 的 4096 世界与导出那一局，都对各自的参考。
D2：已交清单 30 集导出那一局对自己参考的 SP、MP 成功集数（与榜上集数对照在计划里写）。
D4：新清单 SP、MP 集数的区间。
用法：python3 stages/evaluate/t3_resub_table.py   最后一行 T3_RESUB_TABLE_DONE
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SUBMITTED = "stages/evaluate/exports/t3_select_t3m1022a.tsv"
STEP = "349999104"
NEW = {
    5: "t3-ep5-w2048-cap10m-r3a-gb5-s43-ml40",
    10: "t3-ep10-w2048-cap10m-r3a-gb5-s43-ml40",
    14: "t3-ep14-w2048-cap10m-r3a-gb5-s43-ml40",
    23: "t3-ep23-w2048-cap10m-r3a-gb5-ml40",
    15: "t3-ep15-w2048-cap10m-r3a-gb5hr3sdyn7-ml40",
    12: "t3-ep12-w2048-cap10m-r3a-gb5b2n12hdyn7-ml40",
    27: "t3-ep27-w2048-cap10m-r3a-gb5hr2b2dyn7-ml40",
}


def summary(path: Path) -> dict:
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line.startswith("summary_json "):
            d = json.loads(line[len("summary_json "):])
            d = {**d, **d.get("metrics", {})}
            return {
                "sp": d.get("spider_sr", d.get("spider_sr_uncentered")),
                "mp": d.get("maniptrans_sr"),
                "mppe": d.get("mppe_cm"),
                "auc": d.get("add_auc"),
            }
    return {}


def find(pattern: str, exclude: str = "_check"):
    hits = sorted(p for p in REPO.glob(f"results/**/{pattern}") if exclude not in p.name)
    return hits[0] if hits else None


def ev4096(run: str, step: str) -> dict:
    p = find(f"eval_{run}_s{step}.txt")
    return summary(p) if p else {}


def export(run: str, step: str) -> dict:
    p = find(f"cand_t3_*_{run}_s{step}.txt")
    return summary(p) if p else {}


def fmt(d: dict) -> str:
    if not d:
        return "无记录"
    return f"SP {d['sp']:.3f} / MP {d['mp']:.3f} / MPPE {d['mppe']:.2f} / AUC {d['auc']:.3f}"


def main() -> int:
    old = {}
    for line in (REPO / SUBMITTED).read_text(encoding="utf-8").splitlines():
        t, ep, run, step, _ = line.split("\t")
        old[int(ep)] = (run, step)

    print("== D2 已交清单 t3m1022a：导出那一局对自己参考")
    sp_ok, mp_ok, missing = [], [], []
    exp_old = {}
    for ep in sorted(old):
        run, step = old[ep]
        d = export(run, step)
        exp_old[ep] = d
        if not d:
            missing.append(ep)
            continue
        if d["sp"] >= 0.5:
            sp_ok.append(ep)
        if d["mp"] >= 0.5:
            mp_ok.append(ep)
    print(f"SP 成功 {len(sp_ok)} 集：{sp_ok}")
    print(f"MP 成功 {len(mp_ok)} 集：{mp_ok}")
    print(f"没有导出记录：{missing}")

    print("== D1 换入的集（4096 世界 | 导出那一局，都对各自参考）")
    print("集\t旧 run\t旧 4096\t旧 导出\t新 run\t新 4096\t新 导出")
    new_sp, new_mp = set(sp_ok), set(mp_ok)
    for ep in sorted(NEW):
        orun, ostep = old[ep]
        nrun = NEW[ep]
        n4, ne = ev4096(nrun, STEP), export(nrun, STEP)
        print(f"ep{ep}\t{orun}@{ostep}\t{fmt(ev4096(orun, ostep))}\t{fmt(exp_old[ep])}\t{nrun}\t{fmt(n4)}\t{fmt(ne)}")
        new_sp.discard(ep)
        new_mp.discard(ep)
        if ne and ne["sp"] >= 0.5:
            new_sp.add(ep)
        if ne and ne["mp"] >= 0.5:
            new_mp.add(ep)

    swapped = set(NEW)
    out_sp = sorted(set(sp_ok) & swapped)
    out_mp = sorted(set(mp_ok) & swapped)
    gain_sp = sorted(new_sp & swapped)
    gain_mp = sorted(new_mp & swapped)
    print("== D4 区间（榜上现为 SP 10 集、MP 3 集）")
    print(f"换出的旧 run 里导出对自己参考 SP 成功：{out_sp}；MP 成功：{out_mp}")
    print(f"换入的新 run 导出对自己参考 SP 成功：{gain_sp}；MP 成功：{gain_mp}")
    print(f"新清单导出对自己参考：SP {len(new_sp)} 集，MP {len(new_mp)} 集")
    print(f"SP 区间：下界 {10 - len(out_sp)}，上界 {10 + len(gain_sp)}")
    print(f"MP 区间：下界 {3 - len(out_mp)}，上界 {3 + len(gain_mp)}")
    print("T3_RESUB_TABLE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
