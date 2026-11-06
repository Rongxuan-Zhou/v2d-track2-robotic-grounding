"""SOP 案例视频的清单：每段成片的来历、挑选规则、逐世界误差、各道门的数值、文件大小与 md5（容器里执行，经 scripts/dev/record 落盘）。

计划 agent-doc/plan-site-cases-video-2026-10-08.md 第 2 版第 4 节（判据）与审查第 5 条（钉住 step、sha256、parquet、世界编号、规则、逐世界误差、代码 md5）。
判据：每段 ≤ 3 MB；渲染帧全部有限、写入帧数 = 应写帧数；三道门（物体位置同步 ≤ 1e-6 m、刚体数与刚体名、重置位姿 ≤ 1 mm）都记录在案。全部满足打 CASES_OK。
用法：python case_manifest.py <run>:<类别,类别…> …   （在仓库根）
"""

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

MAX_BYTES = 3 * 1024 * 1024
CODE = ("stages/evaluate/case_clip.py", "stages/evaluate/case_compose.py", "stages/evaluate/case_clip.sh", "stages/evaluate/fall_order.py")


def md5(p):
    return hashlib.md5(Path(p).read_bytes()).hexdigest()


def main():
    ok = True
    for f in CODE:
        print(f"code md5 {md5(f)}  {f}")
    for spec in sys.argv[1:]:
        run, kinds = spec.split(":")
        d = Path("runs/site_cases") / run
        rec = json.loads((d / "record.json").read_text())
        rend = json.loads((d / "render.json").read_text())
        z = np.load(d / "record.npz", allow_pickle=False)
        parquet = next(json.loads(l)["parquet"] for l in open(f"runs/flash_chord/{run}/run_record.jsonl") if json.loads(l).get("event") == "start")
        print(f"== {run}：checkpoint sha256 {rec['checkpoint_sha256']}，步数 {rec['environment_steps']}，parquet {parquet}，"
              f"本次 {int(z['world_count'])} 世界 rollout MP-SR {float(z['mp_sr']):.4f}")
        written_ok = rend.get("written_frames") == rend.get("total_frames")
        ok &= written_ok
        print(f"   渲染：写入 {rend.get('written_frames')} / {rend.get('total_frames')} 帧，参考栏有限帧 {rend.get('right_finite')} / {rend['segments']['right'][1] - rend['segments']['right'][0]}")
        for kind in kinds.split(","):
            st = z[f"{kind}_stats"]
            pos, ori, succ, t_obj, t_pel, sync, jump, cut = st
            g = rend["gates"][kind]
            fin = rend.get(f"{kind}_finite")
            out = Path("runs/site_cases/out") / f"{run}_{kind}"
            mp4, jpg = out.with_suffix(".mp4"), out.with_suffix(".jpg")
            size = mp4.stat().st_size
            seg = rend["segments"][kind]
            clip_ok = size <= MAX_BYTES and fin == seg[1] - seg[0] and sync <= 1e-6 and g["reset_gap_m"] <= 1e-3 and jpg.exists()
            ok &= clip_ok
            print(f"   {kind}：世界 {int(z[f'{kind}_world'])}；规则 {z[f'{kind}_note']}；这一类中位数 {float(z[f'{kind}_median']):.4g}；"
                  f"MP 位置 {pos * 100:.2f} cm、朝向 {ori:.1f}°（门槛 3 cm / 30°）、成功 {bool(succ)}；t_obj {int(t_obj)}、t_pel {int(t_pel)}；"
                  f"同步差 {sync:.2g} m；骨盆最大单步位移 {jump:.3f} m；帧 {int(cut)}；刚体 {g['bodies']}、刚体名已核 {g['names_checked']}、重置差 {g['reset_gap_m']:.2g} m；"
                  f"有限帧 {fin}/{seg[1] - seg[0]}；{mp4.name} {size} B md5 {md5(mp4)}；{jpg.name} {jpg.stat().st_size if jpg.exists() else 0} B"
                  f"{'' if clip_ok else '  ← 不满足判据'}")
    print("CASES_OK" if ok else "CASES_FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
