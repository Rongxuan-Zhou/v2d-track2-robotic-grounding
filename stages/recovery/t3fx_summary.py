"""t3fix A 与 C 的汇总（训练机上执行，只用标准库）。计划 agent-doc/plan-t3fix-2026-10-27.md 第 1 节 A5、C2 与判据 G0。

读 t3_fix_sil.py 的 <sil>/epNN.json 与 t3_fix_onset.py 的 <onset>/epNN.json：
  1. G0：MIN_GAIN 阶梯上每一档，30 集负对照的 accept 单元数合计；取让它为 0 的最小一档（都不为 0 就打 A_NOT_USABLE）；
  2. 取定那一档下真实候选的 accept 单元，逐个列出（集、帧段、是否被拿、证据帧数、当前与胜出候选的中位 IoU）；
  3. 逐集一行：候选、负对照、单元数、当前中位 IoU；
  4. C：逐集各类段数，以及 drift、hand_late 段逐个列出。
用法：python3 stages/recovery/t3fx_summary.py <sil 目录> <onset 目录>   最后一行 T3FX_SUMMARY_DONE <MIN_GAIN 或 none>
"""

import json
import sys
from pathlib import Path


def main() -> None:
    sil_dir, onset_dir = Path(sys.argv[1]), Path(sys.argv[2])
    sil = {p.stem: json.loads(p.read_text()) for p in sorted(sil_dir.glob("ep*.json"))}
    onset = {p.stem: json.loads(p.read_text()) for p in sorted(onset_dir.glob("ep*.json"))}
    print(f"A：{len(sil)} 集；C：{len(onset)} 集")
    ladder = sorted({mg for d in sil.values() for mg in d["decisions"]}, key=float)
    print("== G0 负对照（accept 单元数，30 集合计）与真实候选")
    chosen = None
    for mg in ladder:
        null = sum(r["decision"] == "accept" for d in sil.values() for r in d["decisions"][mg]["null"])
        real = sum(r["decision"] == "accept" for d in sil.values() for r in d["decisions"][mg]["real"])
        print(f"MIN_GAIN {mg}：负对照 {null}，真实 {real}")
        if chosen is None and null == 0:
            chosen = mg
    if chosen is None:
        print("A_NOT_USABLE：阶梯上每一档负对照都有误判")
    else:
        print(f"取 MIN_GAIN {chosen}")
        print("== 取定一档下真实候选的 accept 单元")
        for ep, d in sil.items():
            for r in d["decisions"][chosen]["real"]:
                if r["decision"] == "accept":
                    print(f"{d['episode']} 帧 {r['start']}–{r['end']} {'被拿' if r['held'] else '没被碰'} 证据 {r['evidence']} "
                          f"当前 {r['median']['current']:.3f} → {r['name']} {r['median'][r['name']]:.3f}")
        print("== 取定一档下各集 refuse 的理由")
        for ep, d in sil.items():
            reasons = {}
            for r in d["decisions"][chosen]["real"]:
                reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
            print(f"{d['episode']}：{reasons}")
    print("== 逐集")
    for ep, d in sil.items():
        print(f"{d['episode']} 帧 {d['frames']} 测 {d['measured']} 单元 {d['units']} 被拿 {d['held_share']} 当前中位 IoU {d['iou_current_p50']} "
              f"对称 {d['symmetries']}{'+连续轴' if d['continuous_axis'] else ''} 候选 {len(d['candidates'])} 负对照 {','.join(d['null'])}")
    print("== C 分类")
    for ep, d in onset.items():
        print(f"{d['episode']}：段 {d['runs']} drift {d['drift']} hand_late {d['hand_late']} still {d['still']} locked {d['locked']} 判不了 {d['inconclusive']}")
        for r in d["rows"]:
            if r["class"] in ("drift", "hand_late"):
                print(f"  {r['class']} 帧 {r['start']}–{r['end']} 参考移动 {r['ref_move_m']} m 质心 {r.get('shift_px_max')} px "
                      f"IoU {r.get('iou_min')} 面积 {r.get('area_min')}–{r.get('area_max')} 已锁 {r.get('locked_frames')} 锁后变差 {r.get('freeze_worse_frames')}")
    print(f"T3FX_SUMMARY_DONE {chosen or 'none'}")


if __name__ == "__main__":
    main()
