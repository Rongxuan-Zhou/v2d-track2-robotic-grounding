"""Tier 3 参考的第 0 帧朝向表与 B2 选支（训练机上经 scripts/dev/in-flash-chord 在容器里跑，只用 CPU；要 numpy、pyarrow）。计划 agent-doc/plan-t3b2-2026-10-28.md 第 2.1 版第 1 步、判据 T0 与 B2 规则。
输入一份没对齐的帧参考目录（frefs_from_motion_t3.py 或 make_frame_refs_t3.py 的输出）与对称群文件（t3sil 分支 t3_sym_dump.py 的输出），逐集：
- 真值只用 Tier 1 第 0 帧（tier1_frame0.py，照常读，不接受 --swap-quat；负对照只在 align_frefs_t3.py 那一步做）；地面法向、人体根同 align_frefs_t3.py 的 our_frame0；
- 四个对齐方案（obj、yaw_obj、yaw_hum、pos）与自动规则各自对齐后，参考第 0 帧的朝向对 Tier 1 第 0 帧的差（度）；另报 align_frefs_t3 报表里的 转角差（对齐前）、倾斜、两个偏航，供与旧记录逐项比；
- B2（只对离散对称群：群元多于 1 个、没有连续轴）：在固定对齐（人体可用用 yaw_hum，否则 pos）下，对群里每个 g 算 第 0 帧位姿右乘 g（绕质心）之后的差，取最小的 g*；
  固定对齐下原差 > 30°、右乘 g* 后 ≤ 30° 且至少小 20°、Tier 1 第 0 帧物体可见，才选 g*，否则不改。选了的集再按自动规则重新对齐一次，新方案与新差只报；
- 近对称（只报）：绕三条主惯性轴每 5° 的转动里，固定对齐下第 0 帧差最小的那个，与它让表面点最大移动多少厘米。
网格核对：对称群文件里每集的 mesh_sha256 必须等于这一集 Tier 3 根（--t3）里 meta 的网格文件的 sha256。
--aligned <目录>：再读一份 align_frefs_t3.py 的输出（帧参考与 ALIGN.json），报对齐输出第 0 帧对 Tier 1 第 0 帧的差（判据 T0 的对照用）。
输出：标准输出逐集报表；--choice <文件> 写 B2 选中的集（g、质心、Tier 1 文件 sha256、规则版本、档位目录、序列、网格 sha256）。最后一行 FRAME0_TABLE_OK n。
用法：python t3_frame0_table.py <帧参考目录> --tier1 <Tier 1 数据根> --t3 <runs/tier3> --sym <对称群 json> [--aligned <目录>] [--choice <文件>]
测试：stages/evaluate/test_t3_frame0_table.py。
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

import align_frefs_t3 as al
import tier1_frame0

RULE = "t3b2-2.1"
NEAR_RULE = "t3b2n12-1.1"
NEAR_MAX_M = 0.04  # plan-t3b2n12 v1.1: set after seeing ep12's 3.53 cm; the next ep12 candidate moves 8.4 cm
NEAR_DEGS = {60, 90, 120, 180, 240, 270, 300}  # n-fold turns, n in {2, 3, 4, 6}
B2_FROM_DEG, B2_TO_DEG, B2_GAIN_DEG = 30.0, 30.0, 20.0
VARIANTS = ("obj", "yaw_obj", "yaw_hum", "pos")


def rot(axis, deg):
    return al.rotvec_to_R(np.asarray(axis, dtype=np.float64) * np.deg2rad(deg))


def right_mul(P, g, com_b):
    """整集右乘物体系旋转 g，绕质心（质心路径不变）：R′ = R g，p′ = p + R (c − g c)。P 是 [T,7]（xyz + xyzw）。"""
    out = P.copy()
    d = np.asarray(com_b, dtype=np.float64) - g @ np.asarray(com_b, dtype=np.float64)
    for i in range(len(P)):
        R = al.q_to_R(P[i, 3:])
        out[i, :3] = P[i, :3] + R @ d
        out[i, 3:] = al.R_to_q(R @ g)
    return out


def frame0_diff(P, R_T, R1):
    return al.angle_deg((R_T @ al.q_to_R(P[0, 3:])).T @ R1)


def b2_choose(d_fixed: dict, visible: bool) -> tuple:
    """d_fixed：群元下标 → 固定对齐下第 0 帧差（度），下标 0 是恒等。返回（选中的下标、理由）。"""
    k = min(d_fixed, key=d_fixed.get)
    d0, dk = d_fixed[0], d_fixed[k]
    if not visible:
        return 0, "Tier 1 第 0 帧物体不可见"
    if d0 <= B2_FROM_DEG:
        return 0, f"原差 {d0:.1f}° ≤ {B2_FROM_DEG:g}°"
    if k == 0:
        return 0, "恒等已是最小"
    if dk > B2_TO_DEG:
        return 0, f"最好的群元后仍差 {dk:.1f}° > {B2_TO_DEG:g}°"
    if d0 - dk < B2_GAIN_DEG:
        return 0, f"只小 {d0 - dk:.1f}° < {B2_GAIN_DEG:g}°"
    return k, f"{d0:.1f}° → {dk:.1f}°"


def near_choose(d0: float, cands: list, visible: bool) -> tuple:
    """B2n (plan-t3b2n12)：cands = [(名字, 差°)]，只含 n 重、表面移动 ≤ NEAR_MAX_M 的候选。返回（选中的名字或 None、理由）。"""
    if not visible:
        return None, "Tier 1 第 0 帧物体不可见"
    if not cands:
        return None, "没有 4 cm 内的 n 重候选"
    if d0 <= B2_FROM_DEG:
        return None, f"原差 {d0:.1f}° ≤ {B2_FROM_DEG:g}°"
    ok = [(n, d) for n, d in cands if d <= B2_TO_DEG]
    if not ok:
        return None, f"候选后最小仍差 {min(d for _, d in cands):.1f}° > {B2_TO_DEG:g}°"
    if len(ok) > 1:
        return None, f"{len(ok)} 个候选都 ≤ {B2_TO_DEG:g}°，拒绝"
    n, d = ok[0]
    if d0 - d < B2_GAIN_DEG:
        return None, f"只小 {d0 - d:.1f}° < {B2_GAIN_DEG:g}°"
    return n, f"{d0:.1f}° → {d:.1f}°"


def mesh_sha(t3: Path, tier_dir: str, ep: int) -> str:
    root = al.root_dir(t3, tier_dir)
    m = next(m for m in map(json.loads, (root / "meta" / "episodes_metadata.jsonl").read_text().splitlines()) if int(m["episode_index"]) == ep)
    return hashlib.sha256((root / m["mesh"]).read_bytes()).hexdigest()


def read_rows(d: Path) -> dict:
    rows = {}
    for m in sorted(d.glob("manifest_*.tsv")):
        with open(m, newline="") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                if int(r["episode"]) in rows:
                    raise SystemExit(f"ep{r['episode']} 在 {d} 的清单里不止一行")
                rows[int(r["episode"])] = r
    return rows


def poses(d: Path, r: dict) -> np.ndarray:
    with np.load(d / r["file"], allow_pickle=False) as z:
        return np.asarray(z["pose_xyzw"], dtype=np.float64)[:, 0, :]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("frefs", type=Path)
    ap.add_argument("--tier1", type=Path, required=True)
    ap.add_argument("--t3", type=Path, required=True)
    ap.add_argument("--sym", type=Path, required=True)
    ap.add_argument("--aligned", type=Path, default=None)
    ap.add_argument("--choice", type=Path, default=None)
    ap.add_argument("--near", default="", help="B2n（plan-t3b2n12）：这些集若不是离散对称，在 4 cm 内的 n 重近对称转动里选")
    a = ap.parse_args()
    if a.choice is not None and a.choice.exists():
        raise SystemExit(f"{a.choice} 已存在")
    sym = json.loads(a.sym.read_text())["episodes"]
    rows = read_rows(a.frefs)
    arows, aj = ({}, {}) if a.aligned is None else (read_rows(a.aligned), json.loads((a.aligned / "ALIGN.json").read_text()))
    print("ep\t档位目录\t自动方案\t转角差°(对齐前)\t倾斜°\t偏航物体°\t偏航人体°\t" + "\t".join(f"{v}后°" for v in VARIANTS)
          + "\t自动后°\t固定对齐\t对称\tB2\tB2理由\tB2后自动方案\tB2后自动差°\t近对称最好(轴/度/差°/移动cm)\t对齐输出差°")
    choice = {}
    near_eps = {int(x) for x in a.near.split(",") if x.strip()}
    for ep in sorted(rows):
        r = rows[ep]
        P = poses(a.frefs, r)
        truth = tier1_frame0.read(a.tier1, ep)
        n, human, vis0, seq_root = al.our_frame0(a.t3, r)
        if truth["sequence_id"] != r["sequence"] or seq_root != r["sequence"]:
            raise SystemExit(f"ep{ep}：序列对不上（Tier 1 {truth['sequence_id']}、根 {seq_root}、清单 {r['sequence']}）")
        s = sym.get(str(ep))
        if s is None:
            raise SystemExit(f"ep{ep}：对称群文件里没有这一集")
        if s["mesh_sha256"] != mesh_sha(a.t3, r["tier_dir"], ep):
            raise SystemExit(f"ep{ep}：对称群文件的网格与 {r['tier_dir']} 根里的网格 sha256 不同")
        R1 = al.q_to_R(truth["q_xyzw"])
        _, _, rep = al.align_episode(P, truth, n, human, "auto")
        diffs = {}
        for v in VARIANTS:
            if v == "yaw_hum" and human is None:
                diffs[v] = None
                continue
            R_T, _, _ = al.align_episode(P, truth, n, human, v)
            diffs[v] = frame0_diff(P, R_T, R1)
        auto_d = diffs[rep["choice"]]
        fixed = "yaw_hum" if human is not None else "pos"
        R_F, _, _ = al.align_episode(P, truth, n, human, fixed)
        G = [np.asarray(g, dtype=np.float64) for g in s["group"]]
        discrete = len(G) > 1 and s["continuous_axis"] is None
        kind = "连续轴" if s["continuous_axis"] is not None else (f"离散{len(G)}" if len(G) > 1 else "无")
        b2, why, after = "-", "不是离散对称", ("-", None)
        if discrete:
            d_fixed = {k: frame0_diff(right_mul(P[:1], g, s["com_b"]), R_F, R1) for k, g in enumerate(G)}
            k, why = b2_choose(d_fixed, truth["visible"])
            b2 = "恒等" if k == 0 else f"g{k}"
            if k:
                P2 = right_mul(P, G[k], s["com_b"])
                _, _, rep2 = al.align_episode(P2, truth, n, human, "auto")
                R_T2, _, _ = al.align_episode(P2, truth, n, human, rep2["choice"])
                after = (rep2["choice"], frame0_diff(P2, R_T2, R1))
                choice[str(ep)] = {"g": G[k].tolist(), "com_b": s["com_b"], "tier1_sha256": truth["file_sha256"], "rule": RULE,
                                   "fixed_alignment": fixed, "d_fixed_before_deg": d_fixed[0], "d_fixed_after_deg": d_fixed[k],
                                   "tier_dir": r["tier_dir"], "sequence": r["sequence"], "mesh_sha256": s["mesh_sha256"],
                                   "group_shift_m": s["group_shift_m"][k], "kind": "b2"}
        if ep in near_eps and not discrete:
            axes_n = np.asarray(s["principal_axes"], dtype=np.float64)
            cands = [x for x in s["near"] if x["deg"] in NEAR_DEGS and x["shift_m"] <= NEAR_MAX_M]
            d0 = frame0_diff(right_mul(P[:1], np.eye(3), s["com_b"]), R_F, R1)
            scored = [(f"轴{x['axis']}/{x['deg']}°/{x['shift_m'] * 100:.2f}cm", frame0_diff(right_mul(P[:1], rot(axes_n[x["axis"]], x["deg"]), s["com_b"]), R_F, R1), x) for x in cands]
            name, why = near_choose(d0, [(n_, d_) for n_, d_, _ in scored], truth["visible"])
            print(f"B2n ep{ep}：固定对齐 {fixed} 下原差 {d0:.1f}°；候选 {len(scored)} 个：" + ("；".join(f"{n_} → {d_:.1f}°" for n_, d_, _ in scored) or "无") + f"；{'选 ' + name if name else '不改'}（{why}）")
            if name:
                x = next(x_ for n_, _, x_ in scored if n_ == name)
                gn = rot(axes_n[x["axis"]], x["deg"])
                P2 = right_mul(P, gn, s["com_b"])
                _, _, rep2 = al.align_episode(P2, truth, n, human, "auto")
                R_T2, _, _ = al.align_episode(P2, truth, n, human, rep2["choice"])
                b2, why, after = f"b2n {name}", why, (rep2["choice"], frame0_diff(P2, R_T2, R1))
                choice[str(ep)] = {"g": gn.tolist(), "com_b": s["com_b"], "tier1_sha256": truth["file_sha256"], "rule": NEAR_RULE,
                                   "fixed_alignment": fixed, "d_fixed_before_deg": d0, "d_fixed_after_deg": dict((n_, d_) for n_, d_, _ in scored)[name],
                                   "tier_dir": r["tier_dir"], "sequence": r["sequence"], "mesh_sha256": s["mesh_sha256"],
                                   "group_shift_m": x["shift_m"], "kind": "b2n", "axis": x["axis"], "deg": x["deg"],
                                   "candidates": [{"name": n_, "d_fixed_deg": d_} for n_, d_, _ in scored]}
        axes = np.asarray(s["principal_axes"], dtype=np.float64)
        best = min(s["near"], key=lambda x: frame0_diff(right_mul(P[:1], rot(axes[x["axis"]], x["deg"]), s["com_b"]), R_F, R1))
        bd = frame0_diff(right_mul(P[:1], rot(axes[best["axis"]], best["deg"]), s["com_b"]), R_F, R1)
        ad = "-"
        if a.aligned is not None and ep in arows:
            ad = f"{frame0_diff(poses(a.aligned, arows[ep]), np.eye(3), R1):.2f}({aj['episodes'][str(ep)]['choice']})"
        f = lambda x: "-" if x is None else f"{x:.2f}"
        print(f"{ep}\t{r['tier_dir']}\t{rep['choice']}\t{f(rep.get('drot_deg'))}\t{f(rep.get('tilt_deg'))}\t{f(rep.get('yaw_obj_deg'))}\t{f(rep.get('yaw_hum_deg'))}\t"
              + "\t".join(f(diffs[v]) for v in VARIANTS)
              + f"\t{f(auto_d)}\t{fixed}\t{kind}\t{b2}\t{why}\t{after[0]}\t{f(after[1])}"
              + f"\t{best['axis']}/{best['deg']}/{bd:.1f}/{best['shift_m'] * 100:.2f}\t{ad}")
    if a.choice is not None:
        a.choice.write_text(json.dumps({"rule": RULE, "source": str(a.frefs), "sym": str(a.sym), "episodes": choice}, ensure_ascii=False, indent=1))
    print(f"FRAME0_TABLE_OK {len(rows)}（B2 选中 {len(choice)} 集：{', '.join(sorted(choice, key=int)) or '无'}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
