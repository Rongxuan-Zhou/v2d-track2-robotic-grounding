"""把一份提交清单里各集的 4096 世界评测结果，与 stages/evaluate/ref_hand_survey.py 的参考手部盘点按序列对上（Mac 上跑，只用标准库，只读本地 results/）。
评测结果取 results/[ml40/]day2/eval_<run>_s<step>.txt 里的 summary_json（parquet 路径给出序列名）。只报不判。
每集一行：ep、run、MP-SR、SP-SR、MPPE、ADD-AUC、网格半径、最大抬高、单手托举帧、无手悬空帧、双手接触帧；最后按 MP-SR 是否 < 0.5 分两组报中位数。
用法：python3 tools/t2_hand_join.py <清单 tsv> <ref_hand_survey 记录>   最后一行 T2_HAND_JOIN_DONE
"""
import json
import statistics
import sys
from pathlib import Path
from typing import Optional


def survey(path: str) -> dict:
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        p = line.split("\t")
        if len(p) == 11 and p[0].startswith("20"):
            out[p[0]] = {"radius": float(p[2]), "lift": float(p[8]), "one": int(p[9]), "none": int(p[10]), "both": int(p[7]), "T": int(p[1])}
    return out


def eval_record(run: str, step: str, machine: str) -> Optional[dict]:
    for d in ([f"results/{machine}/day2", "results/day2"] if machine != "mf4090" else ["results/day2", "results/mf4090/day2"]):
        f = Path(d) / f"eval_{run}_s{step}.txt"
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip().startswith("summary_json"):
                    return json.loads(line.split("summary_json", 1)[1])
    return None


def main() -> int:
    sel, sv = sys.argv[1], survey(sys.argv[2])
    rows = []
    print("ep\trun\tMP-SR\tSP-SR\tMPPE cm\tADD-AUC\t半径 m\t最大抬高 m\t单手托举帧\t无手悬空帧\t双手接触帧")
    for line in Path(sel).read_text(encoding="utf-8").splitlines():
        p = line.split("\t")
        if len(p) < 5:
            continue
        ep, run, step, machine = p[1], p[2], p[3], p[4]
        j = eval_record(run, step, machine)
        if j is None:
            print(f"{ep}\t{run}\t没有评测记录")
            continue
        seq = j["parquet"].split("sequence_id=")[1].split("/")[0]
        s = sv.get(seq)
        if s is None:
            print(f"{ep}\t{run}\t盘点里没有 {seq}")
            continue
        r = {"ep": int(ep), "mp": j["maniptrans_sr"], "sp": j["spider_sr_uncentered"], "mppe": j["mppe_cm"], "auc": j["add_auc"], **s}
        rows.append(r)
        print(f"{ep}\t{run}\t{r['mp']:.3f}\t{r['sp']:.3f}\t{r['mppe']:.2f}\t{r['auc']:.3f}\t{s['radius']:.3f}\t{s['lift']:+.3f}\t{s['one']}\t{s['none']}\t{s['both']}")
    for name, grp in (("MP-SR < 0.5", [r for r in rows if r["mp"] < 0.5]), ("MP-SR ≥ 0.5", [r for r in rows if r["mp"] >= 0.5])):
        if grp:
            med = {k: statistics.median(r[k] for r in grp) for k in ("radius", "lift", "one", "none", "both", "mppe")}
            print(f"{name}：{len(grp)} 集，ep {[r['ep'] for r in grp]}；中位 半径 {med['radius']:.3f}、最大抬高 {med['lift']:+.3f}、单手托举 {med['one']:.0f}、"
                  f"无手悬空 {med['none']:.0f}、双手接触 {med['both']:.0f}、MPPE {med['mppe']:.2f}")
    print("T2_HAND_JOIN_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
