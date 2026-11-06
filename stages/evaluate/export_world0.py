"""导出那 1 局是不是 4096 世界评测里的第 0 个世界（训练机上、仓库根执行，只读已落盘的结果，用宿主机的 numpy）。

背景：export_calibration.py 在 75 个候选上看导出成败与 4096 世界成功率在总量上对得上（z +0.54 / +0.67），但 ep14 两个 r3a seed 的 4096 世界
SP-SR 0.78 / 0.78，导出都失败（MPPE 78 / 85 cm）。如果导出那 1 局固定等于评测里某个固定世界（例如第 0 个：同一个种子、同一份初始扰动），
同一集的不同 policy 面对的是同一个“难度”，导出的成败在同一集的候选之间就不独立，补 seed 的价值要重估。

做法：对每个有逐世界 dump（fc_eval_dump.sh，runs/flash_chord/<run>/eval_dump/eval_<步数>_w4096.npz）又有 1 世界导出
（runs/flash_chord/<run>/export_8c778f9/s<步数>_w1/episode_*.json）的 run，比导出 JSON 的 object_position_error_m、
object_orientation_error_deg 与 dump 里每个世界的 mp_position_m、mp_orientation_deg：
  - 第 0 个世界的差；位置最接近导出的那个世界的序号与差；
  - 导出的位置误差在 4096 个世界里的分位（公平抽样时应在 0–1 上大致均匀）；
  - 导出的 SP / MP 成败与第 0 个世界的成败是否相同。
用法：python3 stages/evaluate/export_world0.py <步数> <run>…   最后一行 EXPORT_WORLD0_DONE
"""

import json
import sys
from pathlib import Path

import numpy as np


def one(run: str, step: str):
    o = Path("runs/flash_chord") / run
    npz = o / "eval_dump" / f"eval_{step}_w4096.npz"
    js = sorted((o / "export_8c778f9" / f"s{step}_w1").glob("episode_*.json"))
    if not npz.is_file() or len(js) != 1:
        print(f"{run}：缺 dump 或导出（dump {npz.is_file()}，导出 JSON {len(js)} 个），跳过")
        return None
    d = np.load(npz)
    pos, ori = np.asarray(d["mp_position_m"], dtype=np.float64), np.asarray(d["mp_orientation_deg"], dtype=np.float64)
    sp, mp = np.asarray(d["sp_success"]).astype(bool), np.asarray(d["mp_success"]).astype(bool)
    if pos.shape != (4096,) or ori.shape != (4096,) or sp.shape != (4096,) or mp.shape != (4096,):
        raise SystemExit(f"{run}：dump 的形状不是 (4096,)：{pos.shape} {ori.shape} {sp.shape} {mp.shape}")
    m = json.loads(js[0].read_text())["metrics"]
    ep, eo = float(m["object_position_error_m"]), float(m["object_orientation_error_deg"])
    esp, emp = bool(m["spider_sr_uncentered"] >= 0.5), bool(m["maniptrans_sr"] >= 0.5)
    near = int(np.argmin(np.abs(pos - ep) + np.abs(ori - eo) / 1000.0))
    pct = float(np.mean(pos < ep)) + 0.5 * float(np.mean(pos == ep))
    print(f"{run}：导出 位置 {ep:.6f} m 朝向 {eo:.4f}° SP/MP {esp:d}/{emp:d} | 第 0 个世界 {pos[0]:.6f} m {ori[0]:.4f}° SP/MP {sp[0]:d}/{mp[0]:d}"
          f"（差 {pos[0] - ep:+.2e} m）| 最接近的世界 #{near} {pos[near]:.6f} m {ori[near]:.4f}°（差 {pos[near] - ep:+.2e} m）"
          f"| 导出在 4096 个世界里的分位 {pct:.3f} | 4096 世界 SP {sp.mean():.3f} MP {mp.mean():.3f}")
    return {"w0_match": abs(pos[0] - ep) < 1e-5 and abs(ori[0] - eo) < 1e-3, "near_match": abs(pos[near] - ep) < 1e-5 and abs(ori[near] - eo) < 1e-3,
            "same_outcome_w0": (esp, emp) == (bool(sp[0]), bool(mp[0])), "pct": pct, "near": near}


def main():
    step, runs = sys.argv[1], sys.argv[2:]
    rows = [r for r in (one(run, step) for run in runs) if r is not None]
    n = len(rows)
    if n == 0:
        raise SystemExit("一个可比的 run 都没有")
    pcts = sorted(r["pct"] for r in rows)
    print(f"可比的 run {n} 个：导出与第 0 个世界逐位相同（位置差 < 1e-5 m 且朝向差 < 1e-3°）{sum(r['w0_match'] for r in rows)} 个；"
          f"与某个世界逐位相同 {sum(r['near_match'] for r in rows)} 个（世界序号 {sorted({r['near'] for r in rows if r['near_match']})}）；"
          f"SP/MP 成败与第 0 个世界相同 {sum(r['same_outcome_w0'] for r in rows)} 个")
    print("导出位置误差在 4096 个世界里的分位（排序后）：" + " ".join(f"{p:.2f}" for p in pcts))
    print("EXPORT_WORLD0_DONE")


if __name__ == "__main__":
    main()
