#!/usr/bin/env python3
"""Tier 2 现在交的清单逐集并列：导出那 1 局（导出 JSON，对自己的 v5.4 参考）与 4096 世界评测（评测记录的 summary_json）的四项（Mac 上、仓库根执行，只用标准库）。
只用 Tier 2 的数据。用途：找“4096 世界 MP-SR 不低、导出那一局却失败”的集，补 seed 有机会把导出那一局翻成成功；也看 AUC 的余量。
评测记录在 results/day2/eval_<run>_s<步数>.txt（mf4090）或 results/ml40/day2/eval_<run>_s<步数>.txt（ml40）。
用法：python3 tools/t2_export_vs_4096.py <选择清单.tsv> <导出目录>   最后一行 T2_EXPORT_VS_4096_DONE
"""

import json
import sys
from pathlib import Path

KEYS = ("spider_sr_uncentered", "maniptrans_sr", "add_auc", "mppe_cm")


def eval_metrics(run: str, step: str):
    for d in ("results/day2", "results/ml40/day2"):
        p = Path(d) / f"eval_{run}_s{step}.txt"
        if p.is_file():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("summary_json "):
                    j = json.loads(line[len("summary_json "):])
                    return {k: j.get(k) for k in KEYS}, str(p)
    return None, None


def main():
    sel, exp = Path(sys.argv[1]), Path(sys.argv[2])
    rows = [l.split("\t") for l in sel.read_text().splitlines() if l.startswith("2\t")]
    if len(rows) != 30:
        sys.exit(f"清单里 Tier 2 有 {len(rows)} 行")
    tot_x = {k: 0.0 for k in KEYS}
    tot_w = {k: 0.0 for k in KEYS}
    print("集\trun\t导出 SP / MP / AUC / MPPE\t4096 世界 SP / MP / AUC / MPPE\t补 seed 候选")
    cand = []
    for t, ep, run, step, *_ in rows:
        e = int(ep)
        j = json.loads((exp / f"episode_{e:06d}.json").read_text())
        if f"/{run}/policy_{step}." not in str(j.get("checkpoint", "")):
            sys.exit(f"ep{e} 的导出不是 {run} 第 {step} 步：{j.get('checkpoint')}")
        x = {k: float(j["metrics"][k]) for k in KEYS}
        w, src = eval_metrics(run, step)
        if w is None or any(w[k] is None for k in KEYS):
            sys.exit(f"找不到 {run} 第 {step} 步的 4096 世界评测记录")
        for k in KEYS:
            tot_x[k] += x[k]
            tot_w[k] += float(w[k])
        flag = ""
        if x["maniptrans_sr"] < 0.5 and w["maniptrans_sr"] >= 0.2:
            flag = f"MP：4096 世界 {w['maniptrans_sr']:.3f}，导出失败"
            cand.append((w["maniptrans_sr"], e, run))
        print(f"ep{e}\t{run}\t{x['spider_sr_uncentered']:.0f} / {x['maniptrans_sr']:.0f} / {x['add_auc']:.3f} / {x['mppe_cm']:.2f}\t"
              f"{w['spider_sr_uncentered']:.3f} / {w['maniptrans_sr']:.3f} / {w['add_auc']:.3f} / {w['mppe_cm']:.2f}\t{flag}")
    print("合计平均（导出，对自己参考）：" + "  ".join(f"{k} {tot_x[k] / 30:.5f}" for k in KEYS))
    print("合计平均（4096 世界）：" + "  ".join(f"{k} {tot_w[k] / 30:.5f}" for k in KEYS))
    cand.sort(reverse=True)
    exp_succ = sum(c[0] for c in cand)
    print(f"导出失败、4096 世界 MP-SR ≥ 0.2 的集 {len(cand)} 个：" + "，".join(f"ep{e}（{p:.3f}）" for p, e, _ in cand)
          + f"；各补一个 seed、新 seed 与原 seed 同样好时，期望多成功 {exp_succ:.2f} 集")
    print("T2_EXPORT_VS_4096_DONE")


if __name__ == "__main__":
    main()
