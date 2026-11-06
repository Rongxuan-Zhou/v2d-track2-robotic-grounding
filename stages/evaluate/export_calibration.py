"""导出那 1 局与 4096 世界评测是不是同一个分布：逐候选拿 4096 世界的成功率当概率，与导出那一局的成败对照（只用标准库）。

导出的 1 局如果是同一策略的一次公平抽样，全部候选上“4096 世界成功率之和”应与“导出成功的个数”在二项分布的噪声内相等，MPPE 也应
大致落在 4096 世界均值两侧。系统性偏离说明导出与评测的设置不一样（初始状态、随机种子、settling 等），那就是一个偏差，要查。

用法：python3 stages/evaluate/export_calibration.py <候选目录> <select_t1 的 scores.json>
候选目录下每个 run 一个子目录，里面有 eval_349999104_w4096.json（stage_t1_candidates.sh 抄的）。
"""

import json
import math
import sys
from pathlib import Path

PAIRS = (("SP-SR", "spider_sr", "spider_sr_uncentered"), ("MP-SR", "maniptrans_sr", "maniptrans_sr"))


def main():
    cand, scores = Path(sys.argv[1]), json.load(open(sys.argv[2]))
    rows = []
    for run, c in sorted(scores.items()):
        j = cand / run / "eval_349999104_w4096.json"
        m = json.loads(j.read_text())["metrics"]
        rows.append((run, c, m))
    print(f"候选 {len(rows)} 个")
    for name, ek, wk in PAIRS:
        p = [r[2][wk] for r in rows]
        y = [r[1][ek] for r in rows]
        mu, var = sum(p), sum(q * (1 - q) for q in p)
        z = (sum(y) - mu) / math.sqrt(var) if var > 0 else float("nan")
        print(f"{name}：4096 世界成功率之和 {mu:.2f}（标准差 {math.sqrt(var):.2f}），导出成功 {sum(y):.0f} 个，z = {z:+.2f}")
        # 分档看：4096 世界成功率在 (0.2, 0.8) 之间的候选，导出的成败最能说明问题
        mid = [(r[0], q, v) for r, q, v in zip(rows, p, y) if 0.2 < q < 0.8]
        if mid:
            print(f"  其中 4096 世界成功率在 0.2–0.8 的 {len(mid)} 个：期望 {sum(q for _, q, _ in mid):.2f}，导出成功 {sum(v for _, _, v in mid):.0f}")
            for run, q, v in mid:
                print(f"    {run}: 4096 世界 {q:.3f}，导出 {v:.0f}")
    d = [r[1]["mppe_cm"] - r[2]["mppe_cm"] for r in rows]
    d.sort()
    n = len(d)
    print(f"MPPE 导出减 4096 世界均值（cm）：中位 {d[n // 2]:+.2f}，四分位 {d[n // 4]:+.2f} / {d[3 * n // 4]:+.2f}，"
          f"导出更大的 {sum(1 for x in d if x > 0)} 个、更小的 {sum(1 for x in d if x < 0)} 个")
    big = sorted(((r[1]["mppe_cm"] - r[2]["mppe_cm"], r[0], r[1]["mppe_cm"], r[2]["mppe_cm"]) for r in rows), reverse=True)[:8]
    for diff, run, e, w in big:
        print(f"    {run}: 导出 {e:.2f}，4096 世界 {w:.2f}（差 {diff:+.2f}）")
    print("EXPORT_CALIBRATION_DONE")


if __name__ == "__main__":
    main()
