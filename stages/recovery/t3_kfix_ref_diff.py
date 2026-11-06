"""k-fix 第一批：新参考与旧参考逐集差多少（Mac 上跑，要 numpy、pyarrow；只读两份 Tier 3 parquet，不读真值，不读 Tier 1、Tier 2 轨迹）。
计划 agent-doc/plan-t3-kfix-2026-10-16.md 判据 5 的风险预期：现有策略训在旧参考上，换成新参考后要多走多远、物体挪多少。

两份参考的世界系原点都是主机位相机中心（世界步只绕原点转、再乘缩放），所以 ‖X‖ 就是离相机的距离：
  - 旧参考：runs/t3world/budget_v2/<集>/episode_*.parquet（另一位用户的 tier3_cari4d_f830_floor，root_v2 的来源；results/t3world/pull_budget_v2.txt）；
  - 新参考：runs/t3world/kfix_new_F/<集>/episode_*.parquet（ml40 上 tier3_kfix_r2/F/runs/tier3_kfix_F；results/t3world/pull_kfix_new_F.txt）。
逐集报（物体取两边都可见的帧）：
  - 物体离相机距离之比 ‖X_新‖/‖X_旧‖ 的中位与范围（推导的预期：(866.3/830)·(0.984/0.9856) ≈ 1.042 的深度比，再乘视线长度之比，合起来约 1.05）；
  - 物体位置差 ‖X_新 − X_旧‖ 的中位与 90% 分位，两者视线的夹角中位（标定内参含约 4.7° 的偏航，新参考不再转 frame refs）；
  - 人体根（observation.human.translation）逐帧位移之和之比（人走的路程），根位置差的中位。
用法：uv run --no-project --python 3.12 --with numpy --with pyarrow python stages/recovery/t3_kfix_ref_diff.py；最后一行 KFIX_REF_DIFF_DONE。
"""
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
EPS = ("ep00_tall_bar_stool", "ep18_blue_trash_can", "ep20_rice_cooker", "ep22_step_ladder", "ep24_watering_can",
       "ep26_skateboard", "ep29_rolling_luggage")


def load(d: Path) -> dict:
    f = next(d.glob("episode_*.parquet"))
    t = pq.read_table(f, columns=["observation.object.pose", "observation.object.visible", "observation.human.translation"])
    col = lambda c: np.stack(t.column(c).to_numpy(zero_copy_only=False))
    return {"obj": col("observation.object.pose")[:, :3].astype(float),
            "vis": np.asarray(t.column("observation.object.visible").to_numpy(zero_copy_only=False), bool),
            "root": col("observation.human.translation").astype(float)}


def path_len(x: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(x, axis=0), axis=1).sum())


def main() -> int:
    sys.path.insert(0, str(ROOT / "libs/v2d_infra"))
    from tier_guard import forbid_tier1
    forbid_tier1()
    old_root, new_root = ROOT / "runs/t3world/budget_v2", ROOT / "runs/t3world/kfix_new_F"
    print("集\t两边可见帧\t旧距离中位(m)\t距离比中位\t距离比范围(5–95%)\t物体位置差中位/90%(cm)\t视线夹角中位(°)\t人走路程比\t根位置差中位(cm)")
    allr, allp = [], []
    for ep in EPS:
        o, n = load(old_root / ep), load(new_root / ep)
        if len(o["vis"]) != len(n["vis"]):
            raise SystemExit(f"{ep}：帧数不同（旧 {len(o['vis'])}，新 {len(n['vis'])}）")
        v = o["vis"] & n["vis"]
        xo, xn = o["obj"][v], n["obj"][v]
        ro, rn = np.linalg.norm(xo, axis=1), np.linalg.norm(xn, axis=1)
        ratio = rn / ro
        disp = np.linalg.norm(xn - xo, axis=1)
        cosang = np.clip((xo * xn).sum(1) / (ro * rn), -1, 1)
        walk = path_len(n["root"]) / path_len(o["root"])
        rdisp = np.linalg.norm(n["root"] - o["root"], axis=1)
        allr.append(ratio)
        allp.append(walk)
        print(f"{ep}\t{int(v.sum())}\t{np.median(ro):.2f}\t{np.median(ratio):.4f}\t{np.percentile(ratio, 5):.4f}–{np.percentile(ratio, 95):.4f}\t"
              f"{100 * np.median(disp):.1f} / {100 * np.percentile(disp, 90):.1f}\t{np.degrees(np.median(np.arccos(cosang))):.2f}\t"
              f"{walk:.4f}\t{100 * np.median(rdisp):.1f}")
    r = np.concatenate(allr)
    print(f"7 集合起来：物体距离比中位 {np.median(r):.4f}（{100 * (np.median(r) - 1):+.1f}%）；人走路程比逐集 {min(allp):.4f}–{max(allp):.4f}，"
          f"中位 {np.median(allp):.4f}")
    print("KFIX_REF_DIFF_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
