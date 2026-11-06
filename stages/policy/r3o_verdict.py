"""r3o 的最终判读（计划 agent-doc/plan-r3o-2026-10-15.md 2.2、2.3）：两段都评完之后，按一档里全部跟丢集判“有用 / 看不清 / 没用”。只用标准库，宿主机上跑。
规格 stages/policy/r3o_verdict.json（P1 = Tier 1 七集、P2 = Tier 2 八集；第二段的基线 10/16 出结果之前登记）。每集的判法与段间门相同
（stages/policy/r3o_gate.py 的 judge：MP-SR ≥ 0.3 修好；不是修好、且 MP-SR ≥ 0.1，或基线 SP-SR < 0.3 而 SP-SR ≥ 0.3，或失败_物体 ≤ 基线 − 0.2，有起色）。
  有用：修好 ≥ 2 集，且对照 MP-SR ≥ 门槛；
  看不清：修好 1 集，或修好 0 集但有起色 ≥ 2 集；
  没用：其余。
没有对照的一节（"control": null，如 C1 = 接触标记提前的 Tier 1 四集，计划 agent-doc/plan-t1-contact-anticipation-2026-10-16.md）总判另写：
  有用：修好 ≥ 1 集；看不清：修好 0 集但有起色 ≥ 2 集；没用：其余。
读 runs/flash_chord/<run>/eval_<步数>_w4096.json；缺文件、世界数不是 4096、有非有限世界就退出码 2（人来看）。Tier 2 先装 Tier 1 禁读。
用法：python3 stages/policy/r3o_verdict.py <P1|P2|C1> [--root .] [--steps 349999104] [--spec stages/policy/r3o_verdict.json]
最后一行 R3O_VERDICT 有用 / 看不清 / 没用，或 R3O_VERDICT_ERROR。测试：stages/policy/test_r3o_verdict.py。
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
from r3o_gate import judge, read_eval


def verdict(fixed: int, improved: int, ok_ctl: bool) -> str:
    if fixed >= 2 and ok_ctl:
        return "有用"
    if fixed == 1 or (fixed == 0 and improved >= 2):
        return "看不清"
    return "没用"


def verdict_no_control(fixed: int, improved: int) -> str:
    if fixed >= 1:
        return "有用"
    if improved >= 2:
        return "看不清"
    return "没用"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("part")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--steps", type=int, default=349999104)
    ap.add_argument("--spec", default=str(HERE / "r3o_verdict.json"))
    a = ap.parse_args(argv)
    spec = json.loads(Path(a.spec).read_text(encoding="utf-8"))
    if a.part.startswith("_") or a.part not in spec:
        print(f"r3o_verdict: 规格里没有 {a.part}")
        print("R3O_VERDICT_ERROR")
        return 2
    s = spec[a.part]
    if s["tier"] != 1:
        from libs.v2d_infra import tier_guard
        tier_guard.forbid_tier1()
    root = Path(a.root)
    try:
        ctl = read_eval(root, s["control"]["run"], a.steps) if s["control"] else None
        lost = {run: read_eval(root, run, a.steps) for run in s["lost"]}
    except (OSError, KeyError, ValueError) as err:
        print(f"r3o_verdict: 读不了评测或评测不干净：{type(err).__name__}: {err}")
        print("R3O_VERDICT_ERROR")
        return 2
    ok_ctl = ctl["mp"] >= s["control"]["floor"] if ctl else None
    print(f"== {a.part}（Tier {s['tier']}，步数 {a.steps}，跟丢集 {len(lost)}）")
    if ctl:
        print(f"  对照 {s['control']['run']}：MP-SR {ctl['mp']:.3f}（门槛 {s['control']['floor']}）{'过' if ok_ctl else '不过'}")
    else:
        print("  这一节没有对照（总判：修好 ≥ 1 集有用，修好 0 集但有起色 ≥ 2 集看不清，其余没用）")
    counts = {"修好": 0, "有起色": 0, "没用": 0}
    for run, e in lost.items():
        b = s["lost"][run]
        v = judge(e, b)
        counts[v] += 1
        print(f"  {run}：MP-SR {e['mp']:.3f}、SP-SR {e['sp']:.3f}、失败_物体 {e['obj']:.3f}（基线 {b['baseline_run']} 失败_物体 "
              f"{b['baseline_obj_fail']:.3f}、SP-SR {b['baseline_sp']:.3f}）→ {v}")
    out = verdict(counts["修好"], counts["有起色"], ok_ctl) if ctl else verdict_no_control(counts["修好"], counts["有起色"])
    print(f"  修好 {counts['修好']}、有起色 {counts['有起色']}、没用 {counts['没用']} → {out}")
    print(f"R3O_VERDICT {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
