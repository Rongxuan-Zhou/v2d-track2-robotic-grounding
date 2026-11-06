"""t3g5 的判据 R1 逐集判（只用标准库；计划 agent-doc/plan-t3g5-2026-10-21.md 第 2 版第 2 节）。
读 fc_eval 落的评测文本（results/day2/eval_<run>_s<步数>.txt 的 summary_json 行），4096 世界、对自己参考：
SP-SR（spider_sr_uncentered）≥ 0.5 且 MPPE ≤ 10 cm 记“跟上”，其余记“仍跟丢”；文件缺或没有 summary_json 记“没有评测结果”。
用法：python3 stages/evaluate/t3g5_r1.py <评测文本>...   最后一行 T3G5_R1_DONE 跟上 a / 共 b
"""
import json
import re
import sys


def verdict(sp: float, mppe: float) -> str:
    return "跟上" if sp >= 0.5 and mppe <= 10.0 else "仍跟丢"


def main(paths: list[str]) -> int:
    ok = 0
    print("判据 R1（SP-SR ≥ 0.5 且 MPPE ≤ 10 cm，对自己参考、4096 世界）：")
    for f in paths:
        try:
            with open(f, encoding="utf-8") as fh:
                line = next(x for x in fh if "summary_json" in x)
        except (OSError, StopIteration):
            print(f"  {f}：没有评测结果")
            continue
        d = json.loads(line.split("summary_json", 1)[1].strip())
        m = re.search(r"flash_chord/([^/]+)/policy_", d["checkpoint"])
        run = m.group(1) if m else d["checkpoint"]
        sp, mppe = float(d["spider_sr_uncentered"]), float(d["mppe_cm"])
        v = verdict(sp, mppe)
        ok += v == "跟上"
        print(f"  {run}：SP-SR {sp:.3f}、MPPE {mppe:.2f} cm、MP-SR {float(d['maniptrans_sr']):.3f}、ADD-AUC {float(d['add_auc']):.3f} → {v}")
    print(f"T3G5_R1_DONE 跟上 {ok} / 共 {len(paths)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
