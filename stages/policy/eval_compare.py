"""比较两份 FlashCHORD 评测 JSON 的关键指标（只用标准库）。

用途：同一个 policy 换机器、换世界数或换存储之后，复核评测是否一致。例如 ml40 的 docker 存储迁移之后，
用新存储上的镜像以 64 个世界重评冒烟 policy，与迁移前 256 个世界的结果比。评测是确定性的、各世界初始状态
相同，世界间差异只来自数值误差，所以不同世界数的 ADD-AUC 应当几乎相同。

用法：python3 stages/policy/eval_compare.py <新 JSON> <参照 JSON> [--max-add-auc-diff 0.005] [--max-chord-z 3]
--max-chord-z：另要求两份 chord_sr 的两比例 z 检验 |z| 小于它（2026-09-28 为 Day 2 的恒等覆盖核对加的，不给就不查）。
两份 JSON 的 checkpoint sha256 不同（比的不是同一个 policy），或 ADD-AUC 差的绝对值不小于阈值时，退出码 1。
"""
import argparse
import json
import sys

KEYS = ["add_auc", "chord_sr", "maniptrans_sr", "spider_sr_uncentered", "mppe_cm",
        "object_position_error_m", "object_orientation_error_deg"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("new")
    ap.add_argument("ref")
    ap.add_argument("--max-add-auc-diff", type=float, default=0.005)
    ap.add_argument("--max-chord-z", type=float)
    a = ap.parse_args()
    with open(a.new) as f:
        n = json.load(f)
    with open(a.ref) as f:
        r = json.load(f)
    same = n.get("checkpoint_sha256") == r.get("checkpoint_sha256")
    print(f"new {a.new}: {n.get('world_count')} worlds; ref {a.ref}: {r.get('world_count')} worlds; "
          f"same checkpoint sha256: {same} ({str(n.get('checkpoint_sha256'))[:16]}…)")
    for k in KEYS:
        x, y = n["metrics"].get(k), r["metrics"].get(k)
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            print(f"  {k:<30} {x:.6g} vs {y:.6g}  diff {x - y:+.6g}")
    d = abs(n["metrics"]["add_auc"] - r["metrics"]["add_auc"])
    ok = same and d < a.max_add_auc_diff
    if a.max_chord_z is not None:
        p1, p2, n1, n2 = n["metrics"]["chord_sr"], r["metrics"]["chord_sr"], n["world_count"], r["world_count"]
        pp = (p1 * n1 + p2 * n2) / (n1 + n2)
        z = (p1 - p2) / (max(pp * (1 - pp), 1e-12) * (1 / n1 + 1 / n2)) ** 0.5
        print(f"chord_sr {p1:.4f} vs {p2:.4f}: |z| {abs(z):.2f} < {a.max_chord_z}: {abs(z) < a.max_chord_z}")
        ok = ok and abs(z) < a.max_chord_z
    print(f"|ADD-AUC diff| {d:.6f} < {a.max_add_auc_diff}: {d < a.max_add_auc_diff}; RESULT {'CONSISTENT' if ok else 'INCONSISTENT'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
