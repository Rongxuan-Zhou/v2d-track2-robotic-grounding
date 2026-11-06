"""一档参考的接触标记分段与越线时刻对照（训练机上经 scripts/dev/in-flash-chord 在容器里跑，只用 CPU；只读这一档的重定向参考）。
计划 agent-doc/plan-c10-extend-2026-10-16.md 第 3.1 节：Tier 2 的接触标记提前要用 Tier 2 自己的数据定参数（净室规矩），不照搬 Tier 1 的 c10。
--tier 2、3 时只收 runs/motion/tier<档>/ 下的参考（带后缀的目录、别的档都拒收），启动时装 Tier 1 原始数据禁读（libs/v2d_infra/tier_guard.forbid_tier1）；tier_guard 只拦原始数据目录 tier_1_multiview_caption，派生参考靠这里的路径检查（独立复核 MINOR 4）。
每集打印：每只手的首次接触帧、接触段数、段间空档长度；最早接触的那只手接触后 --window 帧内物体自转（各帧相对窗口第一帧的转角取最大，
物体朝向取 object_body_wxyz 的第 0 个部件）与平移；给了 --onset（集:越线步）的集再报 delay = 越线步 − 最早首次接触帧（逐帧回放下步号就是帧号）。
汇总按计划第 3.1 节登记的判据（数值在下面的常量里）：
  特征 1：跟丢集里 0 ≤ delay ≤ 40 的集数 ≥ 5（没给越线步的跟丢集算不满足）；
  特征 2：接触后窗口内物体自转的中位，跟丢集 ≥ 其余各集的 3 倍；
  特征 3：跟丢集里有接触的手，接触段数的中位 ≥ 3；三条过两条记“过”。
  lead = 跟丢集里 delay ≥ 0 的最小值，向上取到 5 的倍数，限在 [5, 20]；选集 = 0 ≤ delay ≤ 40 的跟丢集按 --mppe 从大到小取前 3。
episode 号取 runs/motion_src/<档目录>/<序列>/v2d_meta.json。
用法（容器内）：python /repo/stages/evaluate/contact_onset_table.py --tier 2 --lost 1,2,4,9,11,17,25,28 --onset 1:130,2:118 [--mppe 4:119.8]
      [--window 30] <parquet>...；最后几行 CONTACT_ONSET_FEATURES、CONTACT_ONSET_LEAD、CONTACT_ONSET_PICK，最后一行 CONTACT_ONSET_DONE。
测试：stages/evaluate/test_contact_onset_table.py。
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from libs.v2d_infra import tier_guard  # noqa: E402

DELAY_MAX = 40       # 特征 1：越线在首次接触后 40 帧以内
N_DELAY = 5          # 特征 1：8 个跟丢集里至少 5 集
ROT_RATIO = 3.0      # 特征 2：跟丢集的物体自转中位至少是其余各集的 3 倍
SEG_MED = 3          # 特征 3：跟丢集里有接触的手，接触段数中位 ≥ 3
LEAD_STEP, LEAD_MIN, LEAD_MAX = 5, 5, 20
N_PICK = 3
PATH_RE = re.compile(r"/runs/motion/(tier[123])/whole_body/soma/sequence_id=([^/]+)/robot_name=g1/data\.parquet$")


def segments(x: np.ndarray) -> list:
    """布尔序列里连续为真的段，[起, 止) 列表（同 stages/retarget/contact_relabel.py）。"""
    d = np.diff(np.r_[0, x.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = (q / np.linalg.norm(q, axis=-1, keepdims=True)).T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def max_angle_from_first(R: np.ndarray) -> float:
    tr = np.einsum("nij,ij->n", R, R[0])
    return float(np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))).max())


def pairs(text: str, cast) -> dict:
    out = {}
    for item in filter(None, (t.strip() for t in (text or "").split(","))):
        k, v = item.split(":")
        out[int(k)] = cast(v)
    return out


def lead_rule(delays: list):
    ok = [d for d in delays if d >= 0]
    if not ok:
        return None
    return int(min(LEAD_MAX, max(LEAD_MIN, math.ceil(min(ok) / LEAD_STEP) * LEAD_STEP)))


def read_one(p: str, tier: int, window: int) -> dict:
    m = PATH_RE.search(p)
    if not m or m.group(1) != f"tier{tier}":
        raise SystemExit(f"只收 runs/motion/tier{tier}/ 下的参考：{p}")
    meta = Path(p.split("/runs/motion/")[0]) / "runs" / "motion_src" / m.group(1) / m.group(2) / "v2d_meta.json"
    j = json.loads(meta.read_text())
    if int(j["tier"]) != tier:
        raise SystemExit(f"{meta} 的档位是 {j['tier']}，不是 {tier}")
    r = pq.read_table(p, columns=["hand_sides", "hand_contact_active", "object_root_position", "object_body_wxyz"]).to_pylist()[0]
    obj = np.asarray(r["object_root_position"], float)
    c = np.asarray(r["hand_contact_active"], float)
    sides = list(r["hand_sides"])
    if c.ndim != 2 or c.shape != (len(sides), len(obj)):
        raise SystemExit(f"{p}：hand_contact_active 形状 {c.shape}，应为 手 × 帧 = ({len(sides)}, {len(obj)})")
    Ro = quat_to_mat(np.asarray([b[0] for b in r["object_body_wxyz"]], float))
    hands = []
    for h, side in enumerate(sides):
        segs = segments(c[h] > 0.5)
        gaps = [s2 - e1 for (_, e1), (s2, _) in zip(segs, segs[1:])]
        hands.append({"side": side, "segs": segs, "gaps": gaps, "first": segs[0][0] if segs else None, "frames": int((c[h] > 0.5).sum())})
    firsts = [h["first"] for h in hands if h["first"] is not None]
    out = {"ep": int(j["episode"]), "hands": hands, "f0": min(firsts) if firsts else None, "rot": None, "mv": None}
    if out["f0"] is not None:
        f0 = out["f0"]; f1 = min(f0 + window, len(obj))
        out["rot"] = max_angle_from_first(Ro[f0:f1])
        out["mv"] = float(np.linalg.norm(obj[f0:f1] - obj[f0], axis=1).max() * 100)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", type=int, required=True, choices=(1, 2, 3))
    ap.add_argument("--lost", required=True)
    ap.add_argument("--onset", default="")
    ap.add_argument("--mppe", default="")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("parquets", nargs="+")
    a = ap.parse_args()
    if a.tier != 1:
        tier_guard.forbid_tier1()
        print("contact_onset_table：已装 Tier 1 原始数据禁读；派生参考只收 runs/motion/tier<档>/（路径检查）", flush=True)
    lost = {int(x) for x in a.lost.split(",") if x.strip()}
    onset, mppe = pairs(a.onset, int), pairs(a.mppe, float)
    rows = sorted((read_one(p, a.tier, a.window) for p in a.parquets), key=lambda r: r["ep"])
    eps = [r["ep"] for r in rows]
    if len(set(eps)) != len(eps) or not lost <= set(eps):
        raise SystemExit(f"episode 有重复，或跟丢集 {sorted(lost - set(eps))} 不在给的参考里")
    for r in rows:
        r["delay"] = onset[r["ep"]] - r["f0"] if r["ep"] in onset and r["f0"] is not None else None
        parts = []
        for h in r["hands"]:
            if h["first"] is None:
                parts.append(f"{h['side']} 没有接触")
                continue
            g = h["gaps"]
            parts.append(f"{h['side']} 首次接触第 {h['first']} 帧、接触 {h['frames']} 帧、{len(h['segs'])} 段、空档 "
                         + ("无" if not g else "、".join(map(str, g[:12])) + ("…" if len(g) > 12 else "")))
        win = "没有接触" if r["f0"] is None else f"接触后 {a.window} 帧内物体自转 {r['rot']:.1f}°、平移 {r['mv']:.1f} cm"
        d = "" if r["delay"] is None else f"；越线第 {onset[r['ep']]} 步，delay {r['delay']:+d}"
        print(f"ep{r['ep']}{'（跟丢）' if r['ep'] in lost else ''}：{'；'.join(parts)}；{win}{d}")
    L = [r for r in rows if r["ep"] in lost]
    O = [r for r in rows if r["ep"] not in lost]
    ok1 = [r["ep"] for r in L if r["delay"] is not None and 0 <= r["delay"] <= DELAY_MAX]
    nodelay = [r["ep"] for r in L if r["delay"] is None]
    f1 = len(ok1) >= N_DELAY
    print(f"特征 1：跟丢 {len(L)} 集里 0 ≤ delay ≤ {DELAY_MAX} 的 {len(ok1)} 集（{'、'.join(f'ep{e}' for e in ok1) or '无'}）"
          + (f"，没有越线步的 {'、'.join(f'ep{e}' for e in nodelay)}" if nodelay else "") + f"，要求 ≥ {N_DELAY} → {'过' if f1 else '不过'}")
    rl = [r["rot"] for r in L if r["rot"] is not None]
    ro = [r["rot"] for r in O if r["rot"] is not None]
    if rl and ro:
        ml, mo = float(np.median(rl)), float(np.median(ro))
        f2 = ml >= ROT_RATIO * mo
        print(f"特征 2：接触后 {a.window} 帧内物体自转中位 跟丢 {ml:.1f}°（{len(rl)} 集）、其余 {mo:.1f}°（{len(ro)} 集），要求跟丢 ≥ {ROT_RATIO:g} 倍 → {'过' if f2 else '不过'}")
    else:
        f2 = False
        print("特征 2：有一组没有接触的集，算不过")
    segn = [len(h["segs"]) for r in L for h in r["hands"] if h["segs"]]
    f3 = bool(segn) and float(np.median(segn)) >= SEG_MED
    print(f"特征 3：跟丢集里有接触的手 {len(segn)} 只，接触段数中位 {float(np.median(segn)) if segn else float('nan'):.1f}，要求 ≥ {SEG_MED} → {'过' if f3 else '不过'}")
    gl = [g for r in L for h in r["hands"] for g in h["gaps"]]
    go = [g for r in O for h in r["hands"] for g in h["gaps"]]
    for name, g in (("跟丢", gl), ("其余", go)):
        if g:
            print(f"只报：{name}集段间空档 {len(g)} 个，中位 {float(np.median(g)):.0f} 帧，≤ 15 帧的 {sum(x <= 15 for x in g)} 个")
    n = int(f1) + int(f2) + int(f3)
    print(f"CONTACT_ONSET_FEATURES {'过' if n >= 2 else '不过'} {n}/3")
    lead = lead_rule([r["delay"] for r in L if r["delay"] is not None])
    print(f"CONTACT_ONSET_LEAD {lead if lead is not None else '无'}")
    pick = sorted(ok1, key=lambda e: (-mppe.get(e, float("-inf")), e))[:N_PICK]
    print(f"CONTACT_ONSET_PICK {','.join(map(str, pick)) or '无'}"
          + ("（没给 --mppe，按集号）" if not mppe else "（按 MPPE：" + "、".join(f"ep{e} {mppe.get(e, float('nan')):.1f}" for e in pick) + "）"))
    print("CONTACT_ONSET_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
