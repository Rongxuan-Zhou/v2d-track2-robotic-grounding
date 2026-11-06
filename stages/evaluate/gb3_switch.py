"""规则 S：标定内参参考上的候选 run 换不换现选的 run（训练机上、仓库根执行，只用标准库；只读评测 JSON，不读真值）。
计划 agent-doc/plan-t3-gb3-switch-2026-10-17.md 判据 G2：4096 世界、对照各自参考、同一评测器，候选 MPPE ≤ 现选 MPPE + margin_cm，
且候选 SP-SR ≥ sp_floor，记“换”。现选 run 的值写死在规格里（来源记录逐项注明），出结果前登记。
规格文件（JSON）：{"step": 349999104, "margin_cm": 4.0, "sp_floor": 0.5, "source": "<现选值的来源>",
  "pairs": [{"ep": 20, "new": <候选 run>, "ref_dir": "tier3_gb3", "cur": <现选 run>, "cur_mppe": 5.77, "cur_sp": 0.8911, "cur_mp": 0.0046,
             "report_only": false, "note": "<可选：照抄进这一集的输出行>"}, ...]}
新值取 runs/flash_chord/<候选 run>/eval_<步数>_w4096.json（评测链写的；MPPE = mppe_cm，SP-SR = spider_sr_uncentered，MP-SR = maniptrans_sr），
要求 world_count 为 4096、没有非有限世界、JSON 里记的参考在 /runs/motion/<ref_dir>/ 下，不合就“不判”。report_only 的集照算、只报，不计入“换”。
规格可选 "sp_drop_max"（计划 agent-doc/plan-t3g4-2026-10-19.md 第 2 版 G2）：另要求候选 SP-SR ≥ 现选 SP-SR − sp_drop_max；不给时行为同以前。
用法：python3 stages/evaluate/gb3_switch.py <规格>；最后一行 GB3_SWITCH_DONE <换的集数> <不判的集数>。
测试：stages/evaluate/test_gb3_switch.py。
"""
import json
import math
import sys
from pathlib import Path


def judge(p: dict, d: dict, step: int, margin: float, floor: float, drop=None) -> tuple:
    """返回 (结论, 说明)；结论是 换、不换、不判 之一。"""
    m = d.get("metrics", d)
    bad = []
    if int(m.get("world_count", d.get("world_count", 0))) != 4096:
        bad.append("世界数不是 4096")
    if d.get("non_finite_worlds"):
        bad.append(f"非有限世界 {len(d['non_finite_worlds'])} 个")
    if f"/runs/motion/{p['ref_dir']}/" not in str(d.get("parquet")):
        bad.append(f"参考不在 {p['ref_dir']} 下：{d.get('parquet')}")
    vals = {}
    for k in ("mppe_cm", "spider_sr_uncentered", "maniptrans_sr", "add_auc"):
        try:
            vals[k] = float(m[k])
        except (KeyError, TypeError, ValueError):
            bad.append(f"缺 {k}")
    if not bad and not all(math.isfinite(v) for v in vals.values()):
        bad.append("指标不是有限数")
    if bad:
        return "不判", "；".join(bad)
    mppe, sp = vals["mppe_cm"], vals["spider_sr_uncentered"]
    lo = floor if drop is None else max(floor, p["cur_sp"] - drop)
    ok = mppe <= p["cur_mppe"] + margin and sp >= lo
    note = (f"新 MPPE {mppe:.2f} cm、SP-SR {sp:.4f}、MP-SR {vals['maniptrans_sr']:.4f}、ADD-AUC {vals['add_auc']:.3f}；"
            f"规则 S（MPPE ≤ {p['cur_mppe'] + margin:.2f} 且 SP-SR ≥ {lo:.4f}）：{'换' if ok else '不换'}")
    return ("换" if ok else "不换"), note


def main() -> int:
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    step, margin, floor = spec["step"], float(spec["margin_cm"]), float(spec["sp_floor"])
    drop = float(spec["sp_drop_max"]) if "sp_drop_max" in spec else None
    print(f"规则 S：margin {margin} cm、SP-SR 下限 {floor}" + (f"、且不低于现选减 {drop}" if drop is not None else "") + f"；现选值来源：{spec.get('source', '')}")
    switched, skipped = 0, 0
    for p in spec["pairs"]:
        j = Path("runs/flash_chord") / p["new"] / f"eval_{step}_w4096.json"
        head = (f"ep{p['ep']} {p['new']}（现选 {p['cur']}：MPPE {p['cur_mppe']:.2f} cm、SP-SR {p['cur_sp']:.4f}、MP-SR {p['cur_mp']:.4f}）"
                + ("［只报］" if p.get("report_only") else "") + (f"［{p['note']}］" if p.get("note") else ""))
        if not j.exists():
            print(f"{head}：没有 {j}，不判")
            skipped += 1
            continue
        verdict, note = judge(p, json.loads(j.read_text()), step, margin, floor, drop)
        print(f"{head}：{note}" if verdict != "不判" else f"{head}：{note}，不判")
        if verdict == "不判":
            skipped += 1
        elif verdict == "换" and not p.get("report_only"):
            switched += 1
    print(f"GB3_SWITCH_DONE {switched} {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
