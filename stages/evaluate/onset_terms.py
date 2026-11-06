"""事后诊断（计划外，只报）：物体越线的那一步，是位置项（> 0.10 m）还是朝向项（> 阈值）超了；骨盆越线的那一步同样分位置与朝向。
容器里跑，要 numpy。读 fc_eval_dump.sh 的 npz 与新评 JSON，越线的定义照 fall_order.crossings（门与定义不改，这里不判门，门看 fail_onset 的输出）。
起因：逐帧终止分析里 t1-ep4、t3-ep4 的起点挤在冻结期刚结束（第 72、51 帧），参考里的球棒在地上没动；猜测是仿真里的球棒绕长轴滚，
朝向误差不计对称、先越线（计划 agent-doc/plan-frame-onset-2026-10-14.md 执行记录）。一次只收同一档；t2、t3 先装 Tier 1 禁读。
--pos-thr、--ori-thr 改物体越线的门槛（默认同 fall_order.py 的 0.10 m、0.6 rad）：r3o 计划第 0 步要按 MP-SR 的常数
（3 cm、30°）与中间值重读“先丢朝向”（agent-doc/plan-r3o-2026-10-15.md，审查第 2 条）。骨盆的门槛不变。
打印：每条 run 失败世界里 t_obj 存在的个数，其中只有位置超、只有朝向超、两者都超的个数；t_obj 那一步位置误差（cm）与朝向误差（度）的中位；
t_pel 同样分骨盆位置与朝向；按这组门槛、k = 25 时摔倒的失败世界（t_pel 存在的）里物体先、骨盆先、同时的比例（口径同 fall_order.shares，只报）。
自检：三类个数之和等于 t_obj 存在的个数，否则退出码 1。
用法：python stages/evaluate/onset_terms.py <run> [<run> ...] [--steps 349999104] [--worlds 4096] [--root .] [--pos-thr 0.10] [--ori-thr 0.6]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import fall_order as fo  # noqa: E402
from libs.v2d_infra import tier_guard  # noqa: E402


def split(te_at: np.ndarray, pos_thr: float, ori_thr: float):
    p, o = te_at[:, 0] > pos_thr, te_at[:, 1] > ori_thr
    return int((p & ~o).sum()), int((~p & o).sum()), int((p & o).sum())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--steps", type=int, default=349999104)
    ap.add_argument("--worlds", type=int, default=4096)
    ap.add_argument("--root", default=".")
    ap.add_argument("--pos-thr", type=float, default=fo.OBJ_POS_M)
    ap.add_argument("--ori-thr", type=float, default=fo.OBJ_ORI_RAD)
    a = ap.parse_args(argv)
    tiers = {r.split("-", 1)[0] for r in a.runs}
    if len(tiers) != 1 or not tiers <= {"t1", "t2", "t3"}:
        print(f"onset_terms: 一次只收同一档的 run，这次是 {sorted(tiers)}", file=sys.stderr)
        return 2
    if tiers != {"t1"}:
        tier_guard.forbid_tier1()
    bad = 0
    for run in a.runs:
        o = Path(a.root) / "runs/flash_chord" / run / "eval_dump"
        z = dict(np.load(o / f"eval_{a.steps}_w{a.worlds}.npz", allow_pickle=False))
        newj = json.loads((o / f"eval_{a.steps}_w{a.worlds}.json").read_text())
        te, t_obj, t_pel, _, _ = fo.crossings(z, obj_pos_m=a.pos_thr, obj_ori_rad=a.ori_thr)
        thr = z["thresholds"]
        nf = set(int(w) for w in z["non_finite_worlds"])
        valid = np.array([w not in nf for w in range(te.shape[1])])
        failed = ~z["mp_success"] & valid
        wo = np.flatnonzero(failed & (t_obj >= 0))
        wp = np.flatnonzero(failed & (t_pel >= 0))
        obj_at = np.stack([te[t_obj[wo], wo, 4], te[t_obj[wo], wo, 5]], -1) if len(wo) else np.zeros((0, 2))
        pel_at = np.stack([te[t_pel[wp], wp, 0], te[t_pel[wp], wp, 1]], -1) if len(wp) else np.zeros((0, 2))
        po, oo, bo = split(obj_at, a.pos_thr, a.ori_thr)
        pp, op, bp = split(pel_at, float(thr[0]), float(thr[1]))
        print(f"== {run}（step_count {newj['step_count']}，物体判越线用 位置 > {a.pos_thr} m 或朝向 > {a.ori_thr} rad）")
        if len(wo):
            print(f"  物体越线 {len(wo)} 个：只有位置 {po}，只有朝向 {oo}，两者 {bo}；越线那一步位置误差中位 {np.median(obj_at[:, 0]) * 100:.1f} cm、"
                  f"朝向误差中位 {np.degrees(np.median(obj_at[:, 1])):.1f}°；t_obj 中位第 {int(np.median(t_obj[wo]))} 步")
        if len(wp):
            print(f"  骨盆越线 {len(wp)} 个：只有位置 {pp}，只有朝向 {op}，两者 {bp}（门槛 {float(thr[0])} m、{float(thr[1])} rad）")
        fallen = failed & (t_pel >= 0)
        if fallen.any():
            of, pf, si = fo.classify(t_obj[fallen], t_pel[fallen], fo.K_MAIN)
            print(f"  k={fo.K_MAIN}：摔倒的失败世界 {int(fallen.sum())} 个（失败 {int(failed.sum())} 个），物体先 {of.mean():.1%}、骨盆先 {pf.mean():.1%}、"
                  f"同时 {si.mean():.1%}")
        if po + oo + bo != len(wo) or pp + op + bp != len(wp):
            print("  自检不过：三类个数之和与越线个数不等")
            bad += 1
    print("ONSET_TERMS_DONE" if not bad else f"ONSET_TERMS_SELFCHECK_FAIL {bad}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
