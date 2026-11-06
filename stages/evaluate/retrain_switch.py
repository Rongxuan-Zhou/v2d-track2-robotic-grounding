"""重训的 run 换不换（判据 5 的同一写法；训练机上、仓库根执行，只用标准库；只读评测 JSON，不读真值）。
计划 agent-doc/plan-t3-kfix-retrain-2026-10-16.md “判据”：新 run 对新参考、旧 run 对旧参考，都是 4096 世界。
规格文件（JSON）：{"step": 349999104, "pairs": [{"ep": 0, "new": <新 run>, "ref_dir": "tier3_kfix", "old": <旧 run>, "old_mp": 0.9651, "old_sp": 0.9922}, ...]}。
旧值写死在规格里（取自 10/11 的 4096 世界评测，与 fc_8c778f9 一致，results/t3kfix/reexport_checks_mf4090.txt），出结果前登记。
新值取 runs/flash_chord/<新 run>/eval_<步数>_w4096.json（评测链写的；MP-SR = maniptrans_sr，SP-SR = spider_sr_uncentered），
要求 world_count 为 4096、没有非有限世界、JSON 里记的参考在 /runs/motion/<ref_dir>/ 下，不合就“不判”。
旧 MP-SR > 0：新 MP-SR ≥ 旧 − 0.05 且新 SP-SR ≥ 旧 − 0.05 记“换”；旧 MP-SR 为 0：只看 SP-SR ≥ 旧 − 0.05。
用法：python3 stages/evaluate/retrain_switch.py <规格>；最后一行 RETRAIN_SWITCH_DONE <换的集数> <不判的集数>。
"""
import json
import sys
from pathlib import Path


def main() -> int:
    spec = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    step, switched, skipped = spec["step"], 0, 0
    for p in spec["pairs"]:
        j = Path("runs/flash_chord") / p["new"] / f"eval_{step}_w4096.json"
        head = f"ep{p['ep']} {p['new']}（旧 {p['old']}：MP-SR {p['old_mp']:.3f}、SP-SR {p['old_sp']:.3f}）"
        if not j.exists():
            print(f"{head}：没有 {j}，不判")
            skipped += 1
            continue
        d = json.loads(j.read_text())
        m = d.get("metrics", d)
        bad = []
        if int(m.get("world_count", d.get("world_count", 0))) != 4096:
            bad.append("世界数不是 4096")
        if d.get("non_finite_worlds"):
            bad.append(f"非有限世界 {len(d['non_finite_worlds'])} 个")
        if f"/runs/motion/{p['ref_dir']}/" not in str(d.get("parquet")):
            bad.append(f"参考不在 {p['ref_dir']} 下：{d.get('parquet')}")
        if bad:
            print(f"{head}：{'；'.join(bad)}，不判")
            skipped += 1
            continue
        mp, sp = float(m["maniptrans_sr"]), float(m["spider_sr_uncentered"])
        if p["old_mp"] > 0:
            ok = mp >= p["old_mp"] - 0.05 and sp >= p["old_sp"] - 0.05
            rule = f"MP-SR ≥ {p['old_mp'] - 0.05:.4f} 且 SP-SR ≥ {p['old_sp'] - 0.05:.4f}"
        else:
            ok = sp >= p["old_sp"] - 0.05
            rule = f"旧 MP-SR 为 0，只看 SP-SR ≥ {p['old_sp'] - 0.05:.4f}"
        print(f"{head}：新 MP-SR {mp:.4f}、SP-SR {sp:.4f}、ADD-AUC {float(m['add_auc']):.3f}、MPPE {float(m['mppe_cm']):.2f} cm；"
              f"判据 5（{rule}）：{'换' if ok else '不换'}")
        switched += ok
    print(f"RETRAIN_SWITCH_DONE {switched} {skipped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
