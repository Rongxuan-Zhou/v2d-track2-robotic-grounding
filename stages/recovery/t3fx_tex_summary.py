"""t3fix B1 的汇总（训练机上执行，用 t3recon 的 gate 环境）。计划 agent-doc/plan-t3fix-2026-10-27.md 第 3.1 版第 1 节 B1、判据 G0。

读 t3_fix_tex.py 的 <tex>/epNN.json 与 epNN.npz：
  1. 逐集：门槛、候选、（a）能分的候选与自对照、可见像素中位、负对照 accept 数、阳性对照找到的段、真实 accept 段数、各位姿的中位分；
  2. G0：全部集的负对照 accept 合计（任一变体）；能分的集数；
  3. 真实 accept 的段逐个列出（帧段、是否没被拿、换成谁、两个变体下当前与胜出的中位、差、块一致性）；
  4. 真实判定的 refuse 理由计数（合并后与各变体）。
用法：python t3fx_tex_summary.py <tex 目录>   最后一行 T3FX_TEX_SUMMARY_DONE <G0 ok|fail> <能分的集数>
"""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def main() -> None:
    tex = Path(sys.argv[1])
    rows = [json.loads(p.read_text()) for p in sorted(tex.glob("ep*.json"))]
    null_total, distinct_eps = 0, 0
    print("== 逐集")
    for d in rows:
        if "skip" in d:
            print(f"{d['episode']}：不进 B1（{d['skip']}）")
            continue
        if not d.get("gate_ok") or "decisions" not in d:
            print(f"{d['episode']}：门槛 {d.get('gate_iou_p50')}（点数 {d.get('points')}），{'未过' if not d.get('gate_ok') else '只跑了门槛'}")
            continue
        null_total += d["null_variant_accepts"]
        distinct_eps += bool(d["distinct"])
        z = np.load(tex / f"{Path(d['episode']).name[:4]}.npz")
        med = {k.split("__", 1)[1]: round(float(np.nanmedian(z[k])), 3) for k in z.files if k.startswith("i__") and np.isfinite(z[k]).any()}
        print(f"{d['episode']}：门槛 {d['gate_iou_p50']}，候选 {len(d['candidates'])}，能分 {len(d['distinct'])}（自对照 i {d['distinct_ncc']['reseed']['i']}、ii {d['distinct_ncc']['reseed']['ii']}），"
              f"可见像素中位 {d['visible_px_p50']}，段 {d['segments']}，负对照 accept {d['null_variant_accepts']}，阳性对照 {d['positive_found']}，真实 accept {d['accepts']['real']}")
        print(f"  中位分（i）：{med}")
    print("== G0 与停规则")
    print(f"负对照 accept 合计（任一变体）：{null_total}；能分的集：{distinct_eps}")
    print("== 真实 accept 的段")
    reasons, per_variant = Counter(), Counter()
    for d in rows:
        for r in d.get("decisions", {}).get("real", []):
            reasons[r["reason"]] += 1
            for v, x in r["variants"].items():
                per_variant[f"{v}:{x['reason']}"] += 1
            if r["decision"] == "accept":
                vv = {v: (round(x["median"]["current"], 3), round(x["median"][r["name"]], 3), round(x["gap"], 3), round(x["agree"], 2), x["evidence"])
                      for v, x in r["variants"].items()}
                print(f"{d['episode']} 帧 {r['start']}–{r['end']} {'没被拿' if r['unheld'] else '被拿/混合'} → {r['name']}（变体：当前、胜出、差、块一致、证据）{vv}")
    print("== 真实判定的 refuse 理由")
    print(f"合并后：{dict(reasons)}")
    print(f"各变体：{dict(per_variant)}")
    print(f"T3FX_TEX_SUMMARY_DONE {'ok' if null_total == 0 else 'fail'} {distinct_eps}")


if __name__ == "__main__":
    main()
