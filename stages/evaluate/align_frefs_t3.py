"""Tier 3 帧参考的第 0 帧对齐（训练机上经 scripts/dev/in-flash-chord 在容器里跑，只用 CPU；要 numpy、pyarrow）。计划 agent-doc/plan-t3align-2026-10-21.md。
输入一份帧参考目录（make_frame_refs_t3.py / merge_frefs_t3.py 的输出：episode_XXXXXX.npz 与 manifest_*.tsv），每集算一个刚体变换 T，写出新帧参考 T ∘ F_t。
- 真值只用 Tier 1 第 0 帧（stages/evaluate/tier1_frame0.py，唯一的 Tier 1 读取口）：物体位姿 P1₀、人体根的平移与朝向；第 0 帧不可见的集保持原帧参考（锚点会变，计划复核第 5 条）。
- 我们这边 F̂：帧参考开头静止的帧（相对第 0 帧位移 ≤ 1 cm、转角 ≤ 2°，最多 30 帧）的位置中位数与朝向弦均值（只用 Tier 3 数据）。
- 竖直方向：这一集 Tier 3 根（清单的 tier_dir → runs/tier3/root[_<名>]）里 ground_plane 的 plane 法向；人体根取同一份 Tier 3 根 parquet 第 0 帧
  （文件 sha256 必须等于清单的 source_parquet_sha256，否则这一集人体不可用）。
- 位姿都是 World_T_object：p′ = R_T p + t_T，q′ = q_T ⊗ q（Hamilton，xyzw）；平移一律取 t_T = p₁ − R_T p̂。
方案（--variant）：obj（R_T = R₁ R̂ᵀ）、yaw_obj（R_T 取它绕竖直方向的 twist）、yaw_hum（人体根“Tier 1 对我们”的旋转的 twist）、pos（R_T = I）、
auto（登记的规则：人体可用且 |yaw_obj − yaw_hum| ≤ 15° 时，倾斜 ≤ 3° 用 obj、否则 yaw_obj；人体可用但不一致用 yaw_hum；人体不可用用 pos）。
--swap-quat：负对照，故意把 Tier 1 四元数按 xyzw 读错（端到端核对必须抓到）。
输出 <新目录>/episode_XXXXXX.npz、manifest_aligned.tsv（每集一行，sha256 换成新文件的）、ALIGN.json，标准输出逐集报表，最后一行 ALIGN_FREFS_OK n 或报错退出。
用法：python align_frefs_t3.py <帧参考目录> <新目录> --tier1 <Tier 1 数据根> --t3 <runs/tier3> [--variant auto] [--swap-quat]
测试：stages/evaluate/test_align_frefs_t3.py。
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

import tier1_frame0

YAW_AGREE_DEG = 15.0
TILT_OBJ_DEG = 3.0
STATIC_M, STATIC_DEG, STATIC_MAX = 0.01, 2.0, 30
MCOLS = ["episode", "run", "step", "tier_dir", "sequence", "frames", "poses_sha256", "source_parquet_sha256", "file", "sha256"]


def q_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


def q_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def q_to_R(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def R_to_q(R):
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, 0.25 * s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s, (R[1, 0] - R[0, 1]) / s]
    q = np.asarray(q)
    q /= np.linalg.norm(q)
    return q if q[3] >= 0 else -q


def rotvec_to_R(v):
    th = np.linalg.norm(v)
    if th < 1e-12:
        return np.eye(3)
    k = v / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(th) * K + (1 - math.cos(th)) * K @ K


def angle_deg(R):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2))))


def swing_twist(R, n):
    """R 绕单位轴 n 的 twist 旋转矩阵、twist 角（度，带符号）与 swing 角（度）。"""
    q = R_to_q(R)
    d = float(np.dot(q[:3], n))
    tw = np.r_[n * d, q[3]]
    if np.linalg.norm(tw) < 1e-12:
        return np.eye(3), 180.0, angle_deg(R)
    tw /= np.linalg.norm(tw)
    yaw = math.degrees(2 * math.atan2(d, q[3]))
    yaw = (yaw + 180) % 360 - 180
    Rt = q_to_R(tw)
    return Rt, yaw, angle_deg(R @ Rt.T)


def chordal_mean(Rs):
    U, _, Vt = np.linalg.svd(np.mean(Rs, axis=0))
    M = U @ Vt
    if np.linalg.det(M) < 0:
        U[:, -1] *= -1
        M = U @ Vt
    return M


def root_dir(t3: Path, tier_dir: str) -> Path:
    return t3 / ("root" if tier_dir == "tier3" else "root_" + tier_dir[len("tier3_"):])


def our_frame0(t3: Path, row: dict):
    """这一集 Tier 3 根：地面法向、第 0 帧人体根（来源 parquet 对得上才给）、第 0 帧可见与否。"""
    root = root_dir(t3, row["tier_dir"])
    ep = int(row["episode"])
    meta = next(m for m in map(json.loads, (root / "meta" / "episodes_metadata.jsonl").read_text().splitlines()) if int(m["episode_index"]) == ep)
    plane = np.asarray(json.loads((root / meta["ground_plane"]).read_text())["plane"], dtype=np.float64)
    n = plane[:3] / np.linalg.norm(plane[:3])
    f = root / "data" / "chunk-000" / f"episode_{ep:06d}.parquet"
    human, vis0 = None, None
    if f.exists() and hashlib.sha256(f.read_bytes()).hexdigest() == row["source_parquet_sha256"]:
        t = pq.read_table(f, columns=["frame_index", "observation.object.visible", "observation.human.translation", "observation.human.pose"],
                          filters=[("frame_index", "==", 0)]).to_pylist()[0]
        v = t["observation.object.visible"]
        vis0 = bool(v[0] if isinstance(v, (list, tuple)) else v)
        human = (np.asarray(t["observation.human.translation"], dtype=np.float64), np.asarray(t["observation.human.pose"][:3], dtype=np.float64))
    return n, human, vis0, meta["sequence_id"]


def static_estimate(P):
    """帧参考开头静止段的稳健位姿：位置中位数、朝向弦均值。P 是 [T,7]（xyz + xyzw）。"""
    R0 = q_to_R(P[0, 3:])
    k = 1
    while k < min(STATIC_MAX, len(P)):
        if np.linalg.norm(P[k, :3] - P[0, :3]) > STATIC_M or angle_deg(q_to_R(P[k, 3:]) @ R0.T) > STATIC_DEG:
            break
        k += 1
    p = np.median(P[:k, :3], axis=0)
    R = chordal_mean(np.stack([q_to_R(x) for x in P[:k, 3:]]))
    return p, R, k


def align_episode(P, truth, n, human, variant):
    """返回 R_T、t_T、选的方案、理由与报表字段。"""
    p_hat, R_hat, k = static_estimate(P)
    R1 = q_to_R(truth["q_xyzw"])
    R_full = R1 @ R_hat.T
    R_yaw_obj, yaw_obj, tilt = swing_twist(R_full, n)
    yaw_hum, R_yaw_hum = None, None
    if human is not None:
        R_h = rotvec_to_R(truth["human_root_rotvec"]) @ rotvec_to_R(human[1]).T
        R_yaw_hum, yaw_hum, _ = swing_twist(R_h, n)
    cand = {"obj": R_full, "yaw_obj": R_yaw_obj, "yaw_hum": R_yaw_hum, "pos": np.eye(3)}
    if variant == "auto":
        if yaw_hum is not None and abs((yaw_obj - yaw_hum + 180) % 360 - 180) <= YAW_AGREE_DEG:
            choice = "obj" if tilt <= TILT_OBJ_DEG else "yaw_obj"
            why = f"物体与人体偏航一致（差 ≤ {YAW_AGREE_DEG}°），倾斜 {tilt:.2f}° {'≤' if choice == 'obj' else '>'} {TILT_OBJ_DEG}°"
        elif yaw_hum is not None:
            choice, why = "yaw_hum", "物体与人体偏航不一致（对称或朝向翻错）"
        else:
            choice, why = "pos", "人体不可用"
    else:
        choice, why = variant, "指定"
        if cand[choice] is None:
            raise SystemExit("yaw_hum 要人体可用")
    R_T = cand[choice]
    t_T = truth["p"] - R_T @ p_hat
    rep = {"choice": choice, "why": why, "static_frames": k, "dp_cm": float(np.linalg.norm(truth["p"] - p_hat) * 100),
           "drot_deg": angle_deg(R_full), "tilt_deg": tilt, "yaw_obj_deg": yaw_obj, "yaw_hum_deg": yaw_hum,
           "f0_vs_fhat_cm": float(np.linalg.norm(P[0, :3] - p_hat) * 100), "f0_vs_fhat_deg": angle_deg(q_to_R(P[0, 3:]) @ R_hat.T)}
    if human is not None:
        rep["human_root_after_cm"] = float(np.linalg.norm(R_T @ human[0] + t_T - truth["human_t"]) * 100)
        rep["human_root_before_cm"] = float(np.linalg.norm(human[0] - truth["human_t"]) * 100)
    return R_T, t_T, rep


def apply(P, R_T, t_T):
    qT = R_to_q(R_T)
    out = P.copy()
    out[:, :3] = P[:, :3] @ R_T.T + t_T
    for i in range(len(P)):
        q = q_mul(qT, P[i, 3:])
        out[i, 3:] = q / np.linalg.norm(q)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("frefs")
    ap.add_argument("out")
    ap.add_argument("--tier1", required=True)
    ap.add_argument("--t3", required=True)
    ap.add_argument("--variant", default="auto", choices=["auto", "obj", "yaw_obj", "yaw_hum", "pos"])
    ap.add_argument("--swap-quat", action="store_true")
    a = ap.parse_args()
    src, out, t3 = Path(a.frefs), Path(a.out), Path(a.t3)
    if out.exists():
        raise SystemExit(f"{out} 已存在")
    rows = {}
    for m in sorted(src.glob("manifest_*.tsv")):
        with open(m, newline="") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                if int(r["episode"]) in rows:
                    raise SystemExit(f"ep{r['episode']} 在清单里不止一行")
                rows[int(r["episode"])] = r
    out.mkdir(parents=True)
    report, mrows = {}, []
    print("ep\t方案\t理由\t静止帧\t第0帧位置差cm\t转角差°\t倾斜°\t偏航(物体)°\t偏航(人体)°\tF0对F̂ cm/°\t人体根 前→后 cm\tTier3第0帧可见")
    for ep in sorted(rows):
        r = rows[ep]
        z = dict(np.load(src / r["file"]))
        P = np.asarray(z["pose_xyzw"], dtype=np.float64)[:, 0, :]
        truth = tier1_frame0.read(Path(a.tier1), ep, swap_quat=a.swap_quat)
        n, human, vis0, seq_root = our_frame0(t3, r)
        if truth["sequence_id"] != r["sequence"] or seq_root != r["sequence"]:
            raise SystemExit(f"ep{ep}：序列对不上（Tier 1 {truth['sequence_id']}、根 {seq_root}、清单 {r['sequence']}）")
        if not truth["visible"]:
            R_T, t_T, rep = np.eye(3), np.zeros(3), {"choice": "none", "why": "Tier 1 第 0 帧物体不可见，保持原帧参考"}
        else:
            R_T, t_T, rep = align_episode(P, truth, n, human, a.variant)
        newP = apply(P, R_T, t_T)
        z["pose_xyzw"] = newP[:, None, :].astype(np.asarray(z["pose_xyzw"]).dtype)
        np.savez(out / r["file"], **z)
        rep.update({"R_T_quat_xyzw": R_to_q(R_T).tolist(), "t_T": t_T.tolist(), "tier1_sha256": truth.get("file_sha256"), "t3_frame0_visible": vis0})
        report[ep] = rep
        mrows.append({**r, "sha256": hashlib.sha256((out / r["file"]).read_bytes()).hexdigest()})
        f = lambda k, fmt="{:.2f}": "-" if rep.get(k) is None else fmt.format(rep[k])
        print(f"{ep}\t{rep['choice']}\t{rep['why']}\t{rep.get('static_frames', '-')}\t{f('dp_cm')}\t{f('drot_deg')}\t{f('tilt_deg')}\t{f('yaw_obj_deg')}\t{f('yaw_hum_deg')}"
              f"\t{f('f0_vs_fhat_cm')}/{f('f0_vs_fhat_deg')}\t{f('human_root_before_cm')}→{f('human_root_after_cm')}\t{vis0}")
    with open(out / "manifest_aligned.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MCOLS, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(mrows)
    (out / "ALIGN.json").write_text(json.dumps({"variant": a.variant, "swap_quat": a.swap_quat, "source": str(src), "rule": {
        "yaw_agree_deg": YAW_AGREE_DEG, "tilt_obj_deg": TILT_OBJ_DEG, "static": [STATIC_M, STATIC_DEG, STATIC_MAX]}, "episodes": report}, ensure_ascii=False, indent=1))
    print(f"ALIGN_FREFS_OK {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
