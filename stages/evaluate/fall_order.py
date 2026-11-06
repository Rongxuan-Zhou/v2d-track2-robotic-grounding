"""Tier 1 变体 A 失败世界里，物体先丢还是骨盆先偏：读 fc_eval_dump.sh 存的逐步误差，按登记的判据分类（容器或带 numpy 的环境里跑）。

计划 agent-doc/plan-t1-fall-order-2026-10-07.md 第 2 版第 4 节：
- 门：同一条 run 的 npz 与新评 JSON 对齐（MP-SR 差 < 1e-9，误差数组形状 = (step_count, world_count, 6)）；新评与原评测 JSON 的 checkpoint sha256 相同、
  MP-SR 相差 ≤ 0.01；MPPE 相差只报（参考容差 max(0.3, 3·sd/√N) cm）；8 条 run 的骨盆、手掌、物体位置阈值与 active_after_step 都相同，
  物体朝向阈值变体 0.6、基线 1.5。任一不满足就不分类（退出码 1）。
- 失败世界：评测器逐世界 MP-SR 判失败，排除非有限世界。t_obj：物体在冻结期之后（step > active_after_step，与评测器相同）位置 > 0.10 m 或朝向 > 0.6 rad 的
  第一步（两边同一套门槛）；t_pel：骨盆位置或朝向超 checkpoint 阈值的第一步；t_palm 只报。
- 摔倒的失败世界：失败世界里 t_pel 存在的；物体先（t_obj ≤ t_pel − k）、骨盆先（t_pel ≤ t_obj − k，或没有 t_obj）、同时；k 主值 25 步，另报 10、50。
- 结论（变体两种子、两条 episode 合起来）：“物体先”≥ 60% 支持甲，“骨盆先”≥ 60% 支持乙；两条 episode 各自的多数类别必须与合计一致，否则不下结论；
  摔倒的失败世界少于 50 个只描述。只报：基线同样的分类与“变体 − 基线”的比例差、没摔的失败世界数、t_pel < 冻结期 + k 的世界数、
  t_obj 从第 0 步算的敏感性、骨盆超阈值后一直没回来的比例、各类别举 3 个世界编号。

用法：python stages/evaluate/fall_order.py <trial 清单 json> --episodes 18 27 [--steps 349999104] [--worlds 4096] [--root .]
"""

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

NAMES = ["pelvis_position_m", "pelvis_orientation_rad", "palm_position_m", "palm_orientation_rad", "object_position_m", "object_orientation_rad"]
OBJ_POS_M = 0.10
OBJ_ORI_RAD = 0.6
K_MAIN = 25
K_SENS = (10, 25, 50)
SHARE = 0.60
MIN_FALLEN = 50
REPRO_MP = 0.01
CURVE_OFFSETS = (-100, -75, -50, -25, 0)


def is_variant(run: str) -> bool:
    """变体 A（-r3a-）与 r3o（-r3o-，变体 A 加物体朝向项）的物体朝向终止阈值都是 0.6，其余是官方 1.5。"""
    return re.search(r"-r3[ao]-", run) is not None


def first_true(mask):
    """mask (T, W) → 每个世界第一个为真的步，没有为 −1。"""
    return np.where(mask.any(axis=0), mask.argmax(axis=0), -1)


def crossings(z, object_from_zero=False, obj_pos_m=OBJ_POS_M, obj_ori_rad=OBJ_ORI_RAD):
    te = np.asarray(z["tracking_error"], dtype=np.float32)
    thr = z["thresholds"]
    act = z["active_after_step"]
    steps = np.arange(te.shape[0])[:, None]

    def active(i):
        return steps > (-1 if (object_from_zero and i >= 4) else act[i])

    pel_mask = ((te[..., 0] > thr[0]) | (te[..., 1] > thr[1])) & active(0) & active(1)
    palm = ((te[..., 2] > thr[2]) | (te[..., 3] > thr[3])) & active(2) & active(3)
    obj = ((te[..., 4] > obj_pos_m) | (te[..., 5] > obj_ori_rad)) & active(4) & active(5)
    return te, first_true(obj), first_true(pel_mask), first_true(palm), pel_mask


def classify(t_obj, t_pel, k):
    obj_first = (t_obj >= 0) & (t_obj <= t_pel - k)
    pel_first = (t_obj < 0) | (t_pel <= t_obj - k)
    return obj_first, pel_first, ~obj_first & ~pel_first


def run_one(root, run, steps, worlds, is_var, repro_mp=REPRO_MP):
    o = Path(root) / "runs/flash_chord" / run
    newj = json.loads((o / f"eval_dump/eval_{steps}_w{worlds}.json").read_text())
    oldj = json.loads((o / f"eval_{steps}_w{worlds}.json").read_text())
    new, old = newj["metrics"], oldj["metrics"]
    z = dict(np.load(o / f"eval_dump/eval_{steps}_w{worlds}.npz", allow_pickle=False))
    problems = []
    if [str(n) for n in z["criteria_error_names"]] != NAMES or [str(n) for n in z["error_names"]] != NAMES:
        problems.append("误差名不符")
    if tuple(z["tracking_error"].shape) != (int(newj["step_count"]), int(newj["world_count"]), 6):
        problems.append(f"误差数组形状 {z['tracking_error'].shape} 与 JSON 的 ({newj['step_count']}, {newj['world_count']}, 6) 不符")
    if abs(float(np.mean(z["mp_success"])) - new["maniptrans_sr"]) >= 1e-9:
        problems.append("npz 的 MP-SR 与新评 JSON 不符")
    if newj.get("checkpoint_sha256") != oldj.get("checkpoint_sha256"):
        problems.append("checkpoint sha256 与原评测不同")
    d_mp = new["maniptrans_sr"] - old["maniptrans_sr"]
    if abs(d_mp) > repro_mp:
        problems.append(f"MP-SR 差 {d_mp:+.4f} 超过 {repro_mp}")
    want_ori = 0.6 if is_var else 1.5
    if abs(float(z["thresholds"][5]) - want_ori) > 1e-9:
        problems.append(f"物体朝向阈值 {z['thresholds'][5]} 不是 {want_ori}")
    d_mppe = new["mppe_cm"] - old["mppe_cm"]
    tol = max(0.3, 3 * old["object_position_error_std_m"] * 100 / math.sqrt(int(newj["world_count"])))
    te, t_obj, t_pel, t_palm, pel_mask = crossings(z)
    _, t_obj0, _, _, _ = crossings(z, object_from_zero=True)
    nf = set(int(w) for w in z["non_finite_worlds"])
    valid = np.array([w not in nf for w in range(te.shape[1])])
    failed = ~z["mp_success"] & valid
    fallen = failed & (t_pel >= 0)
    stays = pel_mask[-1]  # 最后一步骨盆仍超阈值
    return {
        "run": run, "problems": problems, "d_mp": d_mp, "d_mppe": d_mppe, "tol": tol, "mp_new": new["maniptrans_sr"], "mppe_new": new["mppe_cm"],
        "thr": z["thresholds"], "act": z["active_after_step"], "te": te, "t_obj": t_obj, "t_obj0": t_obj0, "t_pel": t_pel, "t_palm": t_palm,
        "valid": valid, "failed": failed, "fallen": fallen, "stays": stays, "pos_cm": z["mp_position_m"] * 100, "fps": float(z["control_fps"]),
    }


def shares(rows, k=K_MAIN, key="t_obj"):
    t_obj = np.concatenate([r[key][r["fallen"]] for r in rows])
    t_pel = np.concatenate([r["t_pel"][r["fallen"]] for r in rows])
    if not len(t_pel):
        return 0, float("nan"), float("nan"), float("nan")
    of, pf, si = classify(t_obj, t_pel, k)
    return len(t_pel), of.mean(), pf.mean(), si.mean()


def majority(of, pf, si):
    return max((of, "物体先"), (pf, "骨盆先"), (si, "同时"))[1]


def describe(rows, label):
    n_failed = sum(int(r["failed"].sum()) for r in rows)
    n, of, pf, si = shares(rows)
    print(f"[{label}] 失败世界 {n_failed}，摔倒（骨盆超阈值）{n}，没摔 {n_failed - n}")
    n_valid = sum(int(r["valid"].sum()) for r in rows)
    lost = sum(int((r["valid"] & (r["t_obj"] >= 0)).sum()) for r in rows)
    lost_fell = sum(int((r["valid"] & (r["t_obj"] >= 0) & (r["t_pel"] >= 0)).sum()) for r in rows)
    fell_all = sum(int((r["valid"] & (r["t_pel"] >= 0)).sum()) for r in rows)
    print(f"  全部 {n_valid} 个世界：物体丢过（超 0.10 m 或 0.6 rad）{lost}（{lost / n_valid:.1%}），其中后来摔倒 {lost_fell}（{lost_fell / max(lost, 1):.1%}）；"
          f"摔倒的世界共 {fell_all}（{fell_all / n_valid:.1%}）（只报）")
    if not n:
        return
    fps = rows[0]["fps"]
    for k in K_SENS:
        _, a, b, c = shares(rows, k)
        print(f"  k={k:<3}{'（主）' if k == K_MAIN else '（敏感性）'} 物体先 {a:.1%}  骨盆先 {b:.1%}  同时 {c:.1%}")
    _, a, b, c = shares(rows, K_MAIN, key="t_obj0")
    print(f"  敏感性：物体从第 0 步就判 物体先 {a:.1%}  骨盆先 {b:.1%}  同时 {c:.1%}")
    t_obj = np.concatenate([r["t_obj"][r["fallen"]] for r in rows])
    t_pel = np.concatenate([r["t_pel"][r["fallen"]] for r in rows])
    t_palm = np.concatenate([r["t_palm"][r["fallen"]] for r in rows])
    freeze = int(rows[0]["act"][4])
    print(f"  t_pel < 冻结期 + k（{freeze + K_MAIN} 步）的摔倒世界 {int((t_pel < freeze + K_MAIN).sum())} 个（这些按构造只能判骨盆先或同时）")
    both = t_obj >= 0
    if both.any():
        q = np.percentile((t_pel[both] - t_obj[both]) / fps, [10, 25, 50, 75, 90])
        print(f"  t_pel − t_obj（秒，两者都有的 {both.sum()} 个）分位 10/25/50/75/90：" + " / ".join(f"{v:+.2f}" for v in q))
    print(f"  摔倒但物体从未超门槛的 {int((~both).sum())} 个")
    stays = np.concatenate([r["stays"][r["fallen"]] for r in rows])
    print(f"  骨盆超阈值后到最后一步仍超阈值（像真摔，不是暂时走偏）{stays.mean():.1%}")
    ph = t_palm >= 0
    if ph.any():
        b2 = ph & both
        print(f"  手掌超阈值的 {ph.sum()} 个里早于骨盆 {(t_palm[ph] < t_pel[ph]).mean():.1%}"
              + (f"，早于物体 {(t_palm[b2] < t_obj[b2]).mean():.1%}（只报）" if b2.any() else ""))
    vals = {o: [] for o in CURVE_OFFSETS}
    for r in rows:
        for w in np.flatnonzero(r["fallen"]):
            tp = int(r["t_pel"][w])
            for o in CURVE_OFFSETS:
                if 0 <= tp + o < r["te"].shape[0]:
                    vals[o].append(float(r["te"][tp + o, w, 4]) * 100)
    print("  t_pel 前后物体位置误差中位（cm）：" + "，".join(f"{o / fps:+.1f}s {np.median(v):.1f}" for o, v in vals.items() if v))
    tot = sum(int(r["valid"].sum()) for r in rows)
    part = lambda m: sum(float(r["pos_cm"][m(r)].sum()) for r in rows) / tot
    print(f"  平均位置误差（cm，按全部世界摊）：成功 {part(lambda r: r['valid'] & ~r['failed']):.2f}，没摔的失败 {part(lambda r: r['failed'] & ~r['fallen']):.2f}，"
          f"摔倒 {part(lambda r: r['fallen']):.2f}；摔倒世界平均 {np.mean(np.concatenate([r['pos_cm'][r['fallen']] for r in rows])):.1f} cm")
    for r in rows:
        of_, pf_, si_ = classify(r["t_obj"], r["t_pel"], K_MAIN)
        ids = {name: np.flatnonzero(r["fallen"] & m)[:3].tolist() for name, m in (("物体先", of_), ("骨盆先", pf_), ("同时", si_))}
        print(f"  {r['run']} 举例（世界编号，给 viser）：{ids}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("--episodes", type=int, nargs="+", required=True)
    ap.add_argument("--steps", type=int, default=349999104)
    ap.add_argument("--worlds", type=int, default=4096)
    ap.add_argument("--root", default=".")
    ap.add_argument("--repro-mp", type=float, default=REPRO_MP,
                    help="MP-SR 复现门；默认是登记的 0.01，改它属于偏离登记，输出里会标出")
    a = ap.parse_args()
    if a.repro_mp != REPRO_MP:
        print(f"偏离登记：MP-SR 复现门从 {REPRO_MP} 改为 {a.repro_mp}")
    spec = json.loads(Path(a.spec).read_text())
    eps = [e for e in spec["episodes"] if e["ep"] in a.episodes]
    rows = {}
    print("门：")
    bad = False
    for e in eps:
        for key in ("base42", "base43", "var42", "var43"):
            r = run_one(a.root, e[key], a.steps, a.worlds, key.startswith("var"), a.repro_mp)
            rows[(e["ep"], key)] = r
            bad |= bool(r["problems"])
            print(f"  {r['run']}: MP-SR {r['mp_new']:.4f}（{r['d_mp']:+.4f}）MPPE {r['mppe_new']:.2f}（{r['d_mppe']:+.2f}，参考容差 {r['tol']:.2f}）"
                  f"{'；'.join([''] + r['problems']) or ' OK'}；失败 {int(r['failed'].sum())}，摔倒 {int(r['fallen'].sum())}")
    ref = next(iter(rows.values()))
    for r in rows.values():
        if not (np.allclose(r["thr"][:5], ref["thr"][:5]) and np.array_equal(r["act"], ref["act"])):
            print(f"  {r['run']}: 骨盆、手掌或物体位置阈值、active_after_step 与其他 run 不同")
            bad = True
    if bad:
        print("FALL_ORDER STOP：门不过，不分类")
        raise SystemExit(1)
    print(f"阈值（骨盆位置、骨盆朝向、手掌位置、手掌朝向、物体位置）{ref['thr'][:5].tolist()}，active_after_step {ref['act'].tolist()}")
    var = [r for (ep, k), r in rows.items() if k.startswith("var")]
    base = [r for (ep, k), r in rows.items() if k.startswith("base")]
    for e in eps:
        describe([rows[(e["ep"], "var42")], rows[(e["ep"], "var43")]], f"ep{e['ep']} 变体")
        describe([rows[(e["ep"], "base42")], rows[(e["ep"], "base43")]], f"ep{e['ep']} 基线（只报）")
    describe(base, "基线合计（只报）")
    describe(var, "变体合计（判定）")
    n, of, pf, si = shares(var)
    _, bof, bpf, bsi = shares(base)
    print(f"变体 − 基线（k={K_MAIN}）：物体先 {of - bof:+.1%}，骨盆先 {pf - bpf:+.1%}，同时 {si - bsi:+.1%}（只报）")
    pooled = majority(of, pf, si)
    per_ep = {e["ep"]: majority(*shares([rows[(e["ep"], "var42")], rows[(e["ep"], "var43")]])[1:]) for e in eps}
    if n < MIN_FALLEN:
        verdict = f"摔倒的失败世界只有 {n} 个（< {MIN_FALLEN}），只描述"
    elif any(v != pooled for v in per_ep.values()):
        verdict = f"不下结论：两条 episode 的多数类别不一致 {per_ep}"
    elif of >= SHARE:
        verdict = "支持甲：物体先丢，之后骨盆才超阈值"
    elif pf >= SHARE:
        verdict = "支持乙：骨盆先超阈值，物体在那之前还跟着"
    else:
        verdict = f"不下结论：两类都不到 60%（物体先 {of:.1%}、骨盆先 {pf:.1%}）"
    print(f"FALL_ORDER {verdict}")


if __name__ == "__main__":
    main()
