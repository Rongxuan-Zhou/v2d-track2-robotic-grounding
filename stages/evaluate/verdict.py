"""按事先登记的判据对评测结果下结论（只读 results/ 下的文本结果，只用标准库，任何机器都能跑）。

输入是 stages/policy/eval_summary.py 与 stages/evaluate/static_baseline.py 输出里的 `summary_json` 行。
Day 1 登记（agent-doc/experiments/2026-09-28-day1.md）：通过 = chord_sr > 0 且 maniptrans_sr > 0 且
ADD-AUC > “物体全程不动”基线（2026-09-27 14:38 MDT 修正，评测之前）；原判据的最后一项是 ADD-AUC > 0.465。
两套都算，另报每个 policy 相对“不动基线”的差，两个 policy 之间的差（描述性，不设阈值）。
--manifests 给两份 policy_inputs.py 的输出时，比较其中的 sha 行，说明跨机评测的输入是否逐字节一致。

用法：python3 stages/evaluate/verdict.py --static results/day1/static_baseline_ep20.txt \
        --eval results/ml40/day1/eval_<run>.txt --eval ... [--orig-add-bar 0.465] [--manifests A B]
（原判据只对 Day 1 ep20 有意义，snack_box 等后来登记的判据不传 --orig-add-bar。）
"""
import argparse
import json
import sys
from pathlib import Path

RATES = ("chord_sr", "maniptrans_sr", "spider_sr_uncentered")


def summary(path: str) -> dict:
    for line in Path(path).read_text().splitlines():
        s = line.strip()
        if s.startswith("summary_json "):
            return json.loads(s[len("summary_json "):])
    raise SystemExit(f"{path}: 没有 summary_json 行")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--static", required=True)
    ap.add_argument("--eval", action="append", required=True)
    ap.add_argument("--orig-add-bar", type=float, help="原判据的 ADD-AUC 门槛（Day 1 ep20 为 0.465）；不给就不报原判据")
    ap.add_argument("--manifests", nargs=2)
    ap.add_argument("--bar-check-diag", help="诊断报告（diag_report.py）：其中用评测器顶点复算的不动 ADD-AUC 与 --static 的差"
                                             "超过 --bar-tolerance 时改用复算值（Day 1b 登记的规则）")
    ap.add_argument("--bar-tolerance", type=float, default=0.002)
    a = ap.parse_args()

    st = summary(a.static)
    bar = st["add_auc"]
    if a.bar_check_diag:
        re_bar = None
        for line in Path(a.bar_check_diag).read_text().splitlines():
            if line.strip().startswith("summary_json "):
                re_bar = json.loads(line.strip()[len("summary_json "):])["static_add_auc"]
        if re_bar is None:
            raise SystemExit(f"{a.bar_check_diag}: 没有 summary_json 行")
        use = abs(re_bar - bar) > a.bar_tolerance
        print(f"门槛核对：--static {bar:.4f}；评测器顶点复算 {re_bar:.4f}（{a.bar_check_diag}）；差 {re_bar - bar:+.4f}，"
              f"{'超过' if use else '未超过'} {a.bar_tolerance} → 用{'复算值' if use else ' --static 的值'}")
        if use:
            bar = re_bar
    print(f"do-nothing baseline ({a.static}): add_auc {bar:.4f}, object_position_error {st['object_position_error_m'] * 100:.2f} cm, "
          f"object_orientation_error {st['object_orientation_error_deg']:.2f} deg, mppe {st['mppe_cm']:.2f} cm, "
          f"spider_sr {st['spider_sr_uncentered']:.3f}, maniptrans_sr {st['maniptrans_sr']:.3f}")
    rows = []
    for p in a.eval:
        m = summary(p)
        w = m["world_count"]
        name = Path(p).stem.removeprefix("eval_")
        crit = [("chord_sr", m["chord_sr"], 0.0), ("maniptrans_sr", m["maniptrans_sr"], 0.0), ("add_auc", m["add_auc"], bar)]
        revised = all(v > b for _, v, b in crit)
        original = None if a.orig_add_bar is None else (m["chord_sr"] > 0 and m["maniptrans_sr"] > 0 and m["add_auc"] > a.orig_add_bar)
        print(f"\n== {name}: {w} worlds, {m['environment_steps']:,} steps, checkpoint sha256 {m['checkpoint_sha256'][:16]}…")
        for k, v, b in crit:
            extra = f" ({round(v * w)} of {w} worlds)" if k in RATES else ""
            print(f"  {k:<14} {v:.6g} > {b:.6g}{extra}: {'PASS' if v > b else 'FAIL'}")
        if a.orig_add_bar is None:
            print(f"  VERDICT: {'PASS' if revised else 'FAIL'}")
        else:
            print(f"  add_auc vs the original bar: {m['add_auc']:.6g} > {a.orig_add_bar}: {'PASS' if m['add_auc'] > a.orig_add_bar else 'FAIL'}")
            print(f"  VERDICT revised criteria: {'PASS' if revised else 'FAIL'}; original criteria: {'PASS' if original else 'FAIL'}")
        print(f"  spider_sr_uncentered {m['spider_sr_uncentered']:.6g} ({round(m['spider_sr_uncentered'] * w)} of {w} worlds); "
              f"task-native failure causes {m['reference_tracking'].get('failure_cause_fraction')}")
        print(f"  minus do-nothing: add_auc {m['add_auc'] - bar:+.4f}; object_position_error {(m['object_position_error_m'] - st['object_position_error_m']) * 100:+.2f} cm; "
              f"object_orientation_error {m['object_orientation_error_deg'] - st['object_orientation_error_deg']:+.2f} deg; mppe {m['mppe_cm'] - st['mppe_cm']:+.2f} cm")
        rows.append((name, m))
    if len(rows) == 2:
        (n1, m1), (n2, m2) = rows
        print(f"\n{n1} minus {n2}:")
        for k in ("add_auc", "chord_sr", "maniptrans_sr", "spider_sr_uncentered", "mppe_cm", "object_position_error_m", "object_orientation_error_deg"):
            print(f"  {k:<30} {m1[k]:.6g} - {m2[k]:.6g} = {m1[k] - m2[k]:+.6g}")
    rc = 0
    if a.manifests:
        sha = [sorted(line for line in Path(f).read_text().splitlines() if line.startswith("sha ")) for f in a.manifests]
        same = sha[0] == sha[1] and bool(sha[0])
        print(f"\ninput manifests: {len(sha[0])} vs {len(sha[1])} files; byte-identical: {same} ({a.manifests[0]} vs {a.manifests[1]})")
        rc = 0 if same else 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
