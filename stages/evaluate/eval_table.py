"""把若干份评测文本并成一张表（Mac 上跑，只用标准库，只读已落盘的结果）。

每份文件要有 summary_json 行（fc_eval 打的，见 results/day2/eval_<run>_s<步数>.txt）；run 名取自其中 checkpoint 的路径。
列：run、步数、世界数、帧数、MP-SR（maniptrans_sr）、SP-SR（spider_sr_uncentered）、ADD-AUC、MPPE（cm）、平均位置误差（cm）、
平均朝向误差（°）、CHORD SR、失败原因里骨盆 / 物体 / 手掌的比例、checkpoint sha256 前 12 位。成功率与误差都是对全部世界取平均。
--mean：最后多一行，各指标对所给文件逐条等权平均（用原始值算，不用表里四舍五入过的数）。
用法：python3 stages/evaluate/eval_table.py [--mean] <评测文本> ...
测试：stages/evaluate/test_eval_table.py。
"""
import json
import re
import sys
from pathlib import Path

# 指标列：（取值函数，小数位）；--mean 对这些列求平均
METRICS = [
    (lambda j, f: j["maniptrans_sr"], 3), (lambda j, f: j["spider_sr_uncentered"], 3), (lambda j, f: j["add_auc"], 3),
    (lambda j, f: j["mppe_cm"], 2), (lambda j, f: 100 * j["object_position_error_m"], 2), (lambda j, f: j["object_orientation_error_deg"], 1),
    (lambda j, f: j["chord_sr"], 3), (lambda j, f: f.get("pelvis", 0.0), 3), (lambda j, f: f.get("object", 0.0), 3), (lambda j, f: f.get("palm", 0.0), 3),
]


def load(path: Path) -> dict:
    m = re.search(r"summary_json (\{.*\})", path.read_text(encoding="utf-8"))
    if not m:
        raise SystemExit(f"eval_table: {path} 里没有 summary_json 行")
    return json.loads(m.group(1))


def values(j: dict) -> list[float]:
    fail = j["reference_tracking"]["failure_cause_fraction"]
    return [float(get(j, fail)) for get, _ in METRICS]


def row(path: Path) -> list[str]:
    j = load(path)
    run = Path(j["checkpoint"]).parent.name
    return ([run, str(j["environment_steps"]), str(j["world_count"]), str(j["step_count"])]
            + [f"{v:.{d}f}" for v, (_, d) in zip(values(j), METRICS)] + [j["checkpoint_sha256"][:12]])


def main(argv: list[str]) -> int:
    mean = "--mean" in argv
    paths = [a for a in argv if a != "--mean"]
    if not paths:
        raise SystemExit(__doc__)
    head = ["run", "步数", "世界", "帧", "MP-SR", "SP-SR", "ADD-AUC", "MPPE_cm", "位置_cm", "朝向_deg", "CHORD_SR",
            "失败_骨盆", "失败_物体", "失败_手掌", "sha256"]
    print("\t".join(head))
    for p in paths:
        print("\t".join(row(Path(p))))
    if mean:
        vals = [values(load(Path(p))) for p in paths]
        avg = [sum(v[i] for v in vals) / len(vals) for i in range(len(METRICS))]
        print("\t".join([f"平均（{len(vals)} 条，等权）", "", "", ""] + [f"{a:.{d}f}" for a, (_, d) in zip(avg, METRICS)] + [""]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
