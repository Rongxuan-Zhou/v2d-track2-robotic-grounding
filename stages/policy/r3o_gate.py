"""r3o 的段间门（计划 agent-doc/plan-r3o-2026-10-15.md 第 3 节、2.2、2.3）：第一段（t1oa / t2oa）评完之后，判第二段起不起。只用标准库，宿主机上跑。
读 stages/policy/r3o_gate.json 里这一段的对照 run、门槛与跟丢集的基线，再读 runs/flash_chord/<run>/eval_<步数>_w4096.json：
  每集：修好 = MP-SR（metrics.maniptrans_sr）≥ 0.3；有起色 = 不是修好，且 MP-SR ≥ 0.1、或 SP-SR（metrics.spider_sr_uncentered）≥ 0.3
        （只对基线 SP-SR < 0.3 的集算）、或失败_物体（reference_tracking.failure_cause_fraction.object）≤ 基线 − 0.2。
  门：对照的 MP-SR 低于门槛，并且跟丢集没有一集修好或有起色 → STOP（退出码 1）；否则 CONTINUE（退出码 0）。
  评测文件缺、世界数不是 4096、有非有限世界 → 退出码 2（不起，人来看）。t2 的段先装 Tier 1 禁读（只读 Tier 2 的评测）。
arm_after.sh 的 V2D_PRE_CMD 调它：退出码 0 才起下一段。最后一行 R3O_GATE_CONTINUE、R3O_GATE_STOP 或 R3O_GATE_ERROR。
用法：python3 stages/policy/r3o_gate.py <段：t1oa|t2oa> [--root .] [--steps 349999104] [--spec stages/policy/r3o_gate.json]
测试：stages/policy/test_r3o_gate.py。
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def read_eval(root: Path, run: str, steps: int) -> dict:
    j = json.loads((root / "runs/flash_chord" / run / f"eval_{steps}_w4096.json").read_text())
    if int(j["world_count"]) != 4096 or j.get("non_finite_worlds"):
        raise ValueError(f"{run}：世界数 {j['world_count']}、非有限世界 {len(j.get('non_finite_worlds') or [])} 个")
    m = j["metrics"]
    return {"mp": float(m["maniptrans_sr"]), "sp": float(m["spider_sr_uncentered"]),
            "obj": float(j["reference_tracking"]["failure_cause_fraction"]["object"])}


def judge(e: dict, base: dict) -> str:
    if e["mp"] >= 0.3:
        return "修好"
    if e["mp"] >= 0.1 or (base["baseline_sp"] < 0.3 and e["sp"] >= 0.3) or e["obj"] <= base["baseline_obj_fail"] - 0.2:
        return "有起色"
    return "没用"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("part")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--steps", type=int, default=349999104)
    ap.add_argument("--spec", default=str(ROOT / "stages/policy/r3o_gate.json"))
    a = ap.parse_args(argv)
    spec = json.loads(Path(a.spec).read_text())
    if a.part not in spec or a.part.startswith("_"):
        print(f"r3o_gate: 规格里没有段 {a.part}")
        print("R3O_GATE_ERROR")
        return 2
    s = spec[a.part]
    if s["tier"] != 1:
        from libs.v2d_infra import tier_guard
        tier_guard.forbid_tier1()
    root = Path(a.root)
    try:
        ctl = read_eval(root, s["control"]["run"], a.steps)
        lost = {run: read_eval(root, run, a.steps) for run in s["lost"]}
    except (OSError, KeyError, ValueError) as err:
        print(f"r3o_gate: 读不了评测或评测不干净：{type(err).__name__}: {err}")
        print("R3O_GATE_ERROR")
        return 2
    print(f"== {a.part}（Tier {s['tier']}，步数 {a.steps}）")
    ok_ctl = ctl["mp"] >= s["control"]["floor"]
    print(f"  对照 {s['control']['run']}：MP-SR {ctl['mp']:.3f}（门槛 {s['control']['floor']}）{'过' if ok_ctl else '不过'}")
    verdicts = {}
    for run, e in lost.items():
        base = s["lost"][run]
        verdicts[run] = judge(e, base)
        print(f"  {run}：MP-SR {e['mp']:.3f}、SP-SR {e['sp']:.3f}、失败_物体 {e['obj']:.3f}（基线 {base['baseline_run']} "
              f"失败_物体 {base['baseline_obj_fail']:.3f}、SP-SR {base['baseline_sp']:.3f}）→ {verdicts[run]}")
    gain = any(v in ("修好", "有起色") for v in verdicts.values())
    if not ok_ctl and not gain:
        print("  对照不过、跟丢集没有一集修好或有起色：不起第二段，记“r3o 伤到已跟住的集”")
        print("R3O_GATE_STOP")
        return 1
    if not ok_ctl:
        print("  对照不过，但跟丢集有修好或有起色的：照起第二段，第二批只铺到跟丢或低分的集")
    print("R3O_GATE_CONTINUE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
