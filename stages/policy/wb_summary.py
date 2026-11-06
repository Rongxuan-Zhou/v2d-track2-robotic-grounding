"""按课程阶段汇总 W&B 上训练曲线的关键指标（只读 W&B；在 FlashCHORD 容器里跑，经 wb_summary.sh 调用）。

每个 run 按 curriculum/stage_index 分段，取每段最后 25% 步数内的均值，另报全程吞吐中位数。
读取用 history(samples=N)：一次请求取回按步数均匀采样的行（N 不小于总行数时即全量）。不用 scan_history：
不带 keys 的全量逐页扫描从 mf4090 拉两条 run 超过 12 分钟未完成；带 keys 的版本在 wandb 0.22.2 会把每行
返回两次（均为实测）。

用法：stages/policy/wb_summary.sh <W&B run id>...    例：t1-ep20-w2048-cap1m-mf4090-s1
"""
import argparse
import statistics as st

import wandb

KEYS = [
    ("voc", "curriculum/voc_scale"),
    ("ff_p", "curriculum/immediate_first_frame_probability"),
    ("ref_prog", "episode/reference_progress_mean_per_completed_episode"),
    ("return", "episode/return_mean_per_completed_episode"),
    ("t_refend", "termination/reference_end_fraction_per_completed_episode"),
    ("t_object", "termination/object_fraction_per_completed_episode"),
    ("t_palm", "termination/palm_fraction_per_completed_episode"),
    ("t_pelvis", "termination/pelvis_fraction_per_completed_episode"),
    ("obj_rot", "tracking_error/object_orientation_rad_mean_env_max_per_step"),
    ("obj_pos", "tracking_error/object_position_m_mean_env_max_per_step"),
    ("palm_pos", "tracking_error/palm_position_m_mean_env_max_per_step"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--entity", default="zhou-rongx-northeastern-university")
    ap.add_argument("--project", default="v2d-track2")
    ap.add_argument("--samples", type=int, default=10000, help="history 采样行数上限")
    a = ap.parse_args()
    api = wandb.Api(timeout=120)
    for rid in a.runs:
        r = api.run(f"{a.entity}/{a.project}/{rid}")
        rows = sorted((x for x in r.history(samples=a.samples, pandas=False) if x.get("_step") is not None),
                      key=lambda x: x["_step"])
        steps = [x["_step"] for x in rows]
        print(f"== {rid}  state={r.state}  rows={len(rows)} (samples={a.samples})  steps {steps[0]:,}..{steps[-1]:,}"
              f"  duplicate steps {len(steps) - len(set(steps))}")
        print(f"  {'stage':>5} {'steps (M)':>15}" + "".join(f"{n:>9}" for n, _ in KEYS))
        stages = sorted({x["curriculum/stage_index"] for x in rows if isinstance(x.get("curriculum/stage_index"), (int, float))})
        for s in stages:
            sel = [x for x in rows if x.get("curriculum/stage_index") == s]
            lo, hi = sel[0]["_step"], sel[-1]["_step"]
            tail = [x for x in sel if x["_step"] >= lo + 0.75 * (hi - lo)]
            cells = []
            for _, k in KEYS:
                v = [x[k] for x in tail if isinstance(x.get(k), (int, float))]
                cells.append(f"{st.mean(v):>9.3f}" if v else f"{'-':>9}")
            print(f"  {int(s):>5} {lo / 1e6:>7.1f}-{hi / 1e6:<7.1f}" + "".join(cells))
        sps = [x["throughput/env_steps_per_second"] for x in rows if isinstance(x.get("throughput/env_steps_per_second"), (int, float))]
        print(f"  throughput median {st.median(sps):,.0f} env steps/s (second half {st.median(sps[len(sps) // 2:]):,.0f})")
        print("  (per stage: mean over the last 25% of that stage's steps)")


if __name__ == "__main__":
    main()
