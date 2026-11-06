"""Tier 1 逐条选：把加了新候选之后的一次选择与上一次逐项对比（只用标准库，宿主 python3 即可）。计划 agent-doc/plan-t1-r3x-2026-10-08.md 第 4 节。

核（任一不过，最后一行 COMPARE_FAIL）：
  C1 上一次的每个候选都在，四项分数与 4096 世界 MP-SR 和上一次相同（差 ≤ 1e-9）
  C2 新出现的候选都在新候选清单里（清单里没出现的只报，例如训练或导出失败）
  C3 清单涉及的 episode 以外，每条选中的 run 与上一次相同
  C4 清单涉及的 episode：选中的是上一次选中的或这条 episode 的新候选，且按登记规则不比上一次选中的差
  C5 打包打印的合计（PACKED，6 位小数）等于 30 条选中者分数的平均（差 ≤ 1e-6）
只报：清单涉及的 episode 上，上一次选中者与各新候选的四项分数、4096 世界 MP-SR、选中谁；新旧两份 CSV 的合计。
用法：python3 stages/evaluate/select_t1_compare.py --old-scores <json> --old-tsv <tsv> --new-scores <json> --new-tsv <tsv> \\
        --new-list <导出清单.tsv> --select-log <select 的 record 文件>
测试：stages/evaluate/test_select_t1_compare.py。
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from select_t1 import METRICS, sort_key  # noqa: E402

TOL = 1e-9


def read_tsv(p):
    out = {}
    for line in Path(p).read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            f = line.split("\t")
            out[int(f[1])] = f[2]
    return out


def mean_of(scores, chosen):
    return {m: sum(scores[r][m] for r in chosen.values()) / len(chosen) for m in METRICS}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    for k in ("--old-scores", "--old-tsv", "--new-scores", "--new-tsv", "--new-list"):
        ap.add_argument(k, required=True)
    ap.add_argument("--select-log", required=True, help="select_t1.sh 的 record 文件；C5 读其中的 PACKED 行（审查 M3：可选会让 C5 静默跳过）")
    a = ap.parse_args(argv)
    old, new = json.loads(Path(a.old_scores).read_text()), json.loads(Path(a.new_scores).read_text())
    old_sel, new_sel = read_tsv(a.old_tsv), read_tsv(a.new_tsv)
    listed = {}
    for line in Path(a.new_list).read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            t, ep, run, _ = line.split("\t")[:4]
            listed[run] = int(ep)
    eps = set(listed.values())
    bad = []

    diff = [r for r in old if r not in new or any(abs(new[r][m] - old[r][m]) > TOL for m in METRICS)
            or new[r].get("mp_sr_4096") != old[r].get("mp_sr_4096")]
    print(f"C1 上一次的 {len(old)} 个候选：缺或分数不同 {len(diff)} 个" + (f"：{diff[:5]}" if diff else ""))
    bad += ["C1"] if diff else []

    extra = sorted(set(new) - set(old))
    unknown = [r for r in extra if r not in listed]
    missing = sorted(set(listed) - set(new))
    print(f"C2 新候选 {len(extra)} 个（清单 {len(listed)} 个）；不在清单里的 {unknown}；清单里没有分数的（只报）{missing}")
    bad += ["C2"] if unknown else []

    if sorted(old_sel) != list(range(30)) or sorted(new_sel) != list(range(30)):
        print(f"C3 选择清单不是 30 条：上一次 {len(old_sel)}，这一次 {len(new_sel)}")
        bad.append("C3")
    else:
        moved = [ep for ep in range(30) if ep not in eps and new_sel[ep] != old_sel[ep]]
        print(f"C3 清单以外的 {30 - len(eps)} 条 episode：选中的 run 变了的 {moved}")
        bad += ["C3"] if moved else []
        for ep in sorted(eps):
            prev, now = old_sel[ep], new_sel[ep]
            if prev not in new or now not in new:
                print(f"C4 ep{ep}：上一次 {prev} 或这一次 {now} 没有分数")
                bad.append(f"C4 ep{ep}")
                continue
            ok = (now == prev or listed.get(now) == ep) and sort_key(new[now]) >= sort_key(new[prev])
            print(f"C4 ep{ep}：上一次 {prev}，这一次 {now}（{'换了' if now != prev else '没换'}）{'' if ok else ' 不过'}")
            bad += [] if ok else [f"C4 ep{ep}"]
            for r in [prev] + sorted(x for x, e in listed.items() if e == ep and x in new):
                c = new[r]
                print(f"    {'*' if r == now else ' '} {r}: " + " ".join(f"{m}={c[m]:.6f}" for m in METRICS) + f" | 4096 世界 MP-SR {c.get('mp_sr_4096')}")
        if any(r not in new for r in new_sel.values()):
            print("合计：这一次选中的 run 有的没有分数")
            print(f"COMPARE_FAIL {bad + ['C5']}")
            return 1
        before, after = mean_of(old, old_sel), mean_of(new, new_sel)
        print("合计（只报）：" + "；".join(f"{m} {before[m]:.4f} → {after[m]:.4f}" for m in METRICS))
        m = re.findall(r"^PACKED (.*)$", Path(a.select_log).read_text(errors="replace"), re.M)
        if not m:
            print("C5 select 记录里没有 PACKED 行")
            bad.append("C5")
        else:
            packed = {k: float(v) for k, v in (x.split("=") for x in m[-1].split())}
            d = max(abs(packed[k] - after[k]) for k in METRICS)
            print(f"C5 打包合计与选中者平均的最大差 {d:.2e}")
            bad += ["C5"] if d > 1e-6 else []
    print("COMPARE_OK" if not bad else f"COMPARE_FAIL {bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
