"""提交前检查两档 CSV（计划 agent-doc/plan-submission-check-2026-10-03.md 第 2 版：判据 C1–C5 中能在本机核的部分，只报 R1–R3）。Mac 上、仓库根执行，只读，
用打包时的 uv 环境（PY = UV_OFFLINE=1 uv run --no-project --python 3.12 --with-requirements <kit>/requirements.txt --with "pandas<3" python）：
  $PY stages/evaluate/submission_check.py --tier 1        Tier 1 的 C1–C4，R1、R2、R3
  $PY stages/evaluate/submission_check.py --tier 2        Tier 2 的 C1–C4，R1、R3；进程开头 forbid_tier1()，读 Tier 1 目录就抛错（净室）
  $PY stages/evaluate/submission_check.py --cross         两档之间：checkpoint 集合不相交、位姿列不逐行相同、代码链接相同且提交在 origin（只读两份 CSV 的列）
  $PY stages/evaluate/submission_check.py --compare A B   C5：两份 CSV 逐字节比；不同时按列报差异
  $PY stages/evaluate/submission_check.py --selftest      在内存里改坏 Tier 1 的 ep3（换 checkpoint 哈希、一行四元数改成非单位、一帧位置挪 1 cm），C1、C3、C4 都必须判不过
读：两份 CSV、runs/submission/final/t{1,2}/ 的导出、stages/evaluate/exports/{mf4090,ml40}.tsv、两台的远端记录（submission_check_remote.sh 的输出，
results/submission/check_remote_mf4090.txt 与 results/ml40/submission/check_remote_ml40.txt）、kit 的样例 CSV、KIT_MANIFEST.json 与三个脚本、本档发布数据的 meta 与参考、
Tier 2 的 v5.4 帧参考。R2（按干净参考的本地分数）只在 --tier 1 里算。每次最后一行 SUBMISSION_CHECK PASS 或 SUBMISSION_CHECK FAIL。
"""
import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KIT = ROOT / "third_party/v2d_submission_kit/update_0f2f9f4b/v2d_submission_kit"
CSV = {t: ROOT / f"runs/submission/csv/track2_tier{t}_full30_1003.csv" for t in (1, 2)}
FINAL = {t: ROOT / f"runs/submission/final/t{t}" for t in (1, 2)}
DATA = {1: ROOT / "data/challenge/track_2/tier_1_multiview_caption", 2: ROOT / "data/challenge/track_2/tier_2_synthetic_noise"}
FRAME_REF_T2 = ROOT / "runs/submission/frame_refs_t2"
EXPORTS = {m: ROOT / f"stages/evaluate/exports/{m}.tsv" for m in ("mf4090", "ml40")}
REMOTE = {"mf4090": ROOT / "results/submission/check_remote_mf4090.txt", "ml40": ROOT / "results/ml40/submission/check_remote_ml40.txt"}
SAMPLE = KIT / "data/track_2_sample_submission.csv"
EVAL_CODE = "b1b3ebe3ae4629d1687a83f420149e3f6fdfc154d70acad55c12fc8fa69811a1"
STEP = "349999104"
POS = ["pos_x", "pos_y", "pos_z"]
QUAT = ["quat_x", "quat_y", "quat_z", "quat_w"]
HASH_COLUMNS = ("eval_script_sha256", "eval_code_sha256", "checkpoint_sha256", "packer_script_sha256", "uploader_script_sha256", "code_commit_url")
ROW = re.compile(r"track_2/e(\d{6})/w(\d{3})/f(\d{6})/o(\d+)$")
QUAT_TOL = 1e-5
C4_POS_TOL, C4_ANG_TOL = 5e-6, 1e-5


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_csv(path: Path):
    import pandas as pd

    df = pd.read_csv(path, dtype={c: str for c in ("row_id", *HASH_COLUMNS)})
    parts = df["row_id"].str.extract(ROW)
    df["episode"], df["world"], df["frame"], df["slot"] = (pd.to_numeric(parts[i]).astype("Int64") for i in range(4))
    return df


def export_map() -> dict:
    out = {}
    for m, path in EXPORTS.items():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                tier, ep, run, step = line.split("\t")[:4]
                out[(int(tier), int(ep))] = (run, step, m)
    return out


def remote_rows() -> dict:
    out = {}
    for m, path in REMOTE.items():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("ROW\t"):
                c = line.split("\t")
                out[(int(c[1]), int(c[2]))] = {"machine": m, "run": c[3], "step": c[4], "policy_sha": c[5], "world_count": c[6], "step_count": c[7],
                                               "non_finite": c[8], "eval_ckpt_sha": c[9], "add_auc": c[10], "chord_sr": c[11], "mp_sr": c[12],
                                               "sp_sr": c[13], "mppe_cm": c[14], "parquet": c[15]}
    return out


def released_meta(tier: int) -> dict:
    meta = {}
    for line in (DATA[tier] / "meta/episodes.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        meta[int(d["episode_index"])] = {"length": int(d["length"])}
    for line in (DATA[tier] / "meta/episodes_metadata.jsonl").read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        meta[int(d["episode_index"])]["sequence_id"] = d["sequence_id"]
    return meta


def export_files(tier: int, ep: int):
    import pyarrow.parquet as pq

    j = json.loads((FINAL[tier] / f"episode_{ep:06d}.json").read_text(encoding="utf-8"))
    t = pq.read_table(FINAL[tier] / f"episode_{ep:06d}.parquet")
    md = {k.decode(): v.decode() for k, v in (t.schema.metadata or {}).items()}
    return j, t.to_pandas(), md


def released_reference(tier: int, ep: int):
    """本档发布数据的参考（kit 的读法：丢掉物体不可见的帧）。"""
    from v2dlb.trajectory import episode_parquet, read_episode

    return read_episode(episode_parquet(DATA[tier], ep), ep)


# ---------------------------------------------------------------- 判据
def check_c1(tier, df, emap, remote, meta, kit_hashes) -> list:
    bad = []
    for col, want in (("eval_code_sha256", EVAL_CODE), ("eval_script_sha256", kit_hashes["eval_script"]),
                      ("packer_script_sha256", kit_hashes["packer"]), ("uploader_script_sha256", kit_hashes["uploader"])):
        vals = set(df[col].unique())
        if vals != {want}:
            bad.append(f"Tier {tier} {col} 有 {len(vals)} 个值 {sorted(v[:12] for v in vals)}，应只有 {want[:12]}")
    if df["code_commit_url"].nunique() != 1:
        bad.append(f"Tier {tier} code_commit_url 有 {df['code_commit_url'].nunique()} 个值")
    for ep in range(30):
        block = df[df["episode"] == ep]
        run, step, _machine = emap.get((tier, ep), (None, None, None))
        if run is None:
            bad.append(f"Tier {tier} ep{ep}：导出清单里没有")
            continue
        j, _, md = export_files(tier, ep)
        r = remote.get((tier, ep), {})
        ck = set(block["checkpoint_sha256"].unique())
        ck_path = Path(j["checkpoint"])
        checks = {
            "CSV 里只有一个 checkpoint_sha256": len(ck) == 1,
            "CSV 的 checkpoint 与导出 JSON 相同": ck == {j["checkpoint_sha256"]},
            "导出 parquet 元数据的 checkpoint 相同": md.get("flash_chord.checkpoint_sha256") == j["checkpoint_sha256"],
            "服务器上 policy 文件的 sha256 相同": r.get("policy_sha") == j["checkpoint_sha256"],
            "4096 世界评测记的 checkpoint 相同": r.get("eval_ckpt_sha") == j["checkpoint_sha256"],
            "JSON 的 run 与步数等于导出清单": ck_path.parent.name == run and ck_path.name == f"policy_{step}.safetensors" and step == STEP,
            "远端记录的 run 与清单一致": r.get("run") == run,
            "导出 parquet 的评测代码哈希": md.get("flash_chord.eval_code_sha256") == EVAL_CODE,
            "导出 parquet 的评测脚本哈希": md.get("flash_chord.eval_script_sha256") == kit_hashes["eval_script"],
            "训练参考在本档目录": f"/runs/motion/tier{tier}/" in j["parquet"] and f"/runs/motion/tier{tier}/" in md.get("flash_chord.reference_parquet", ""),
            "训练参考与 run 记录相同": r.get("parquet") == j["parquet"] == md.get("flash_chord.reference_parquet"),
            "sequence_id 与本档发布数据一致": f"sequence_id={meta[ep]['sequence_id']}/" in j["parquet"],
            "导出 JSON 没有非有限世界": not j.get("non_finite_worlds"),
            "4096 世界评测：4096 个世界、没有非有限世界": r.get("world_count") == "4096" and r.get("non_finite") == "0",
        }
        for name, ok in checks.items():
            if not ok:
                bad.append(f"Tier {tier} ep{ep}（{run}）：{name} 不成立")
    return bad


def check_c2(tier, df, sample_ids, meta, sample_sha_ok) -> tuple:
    bad, notes = [], []
    if not sample_sha_ok:
        bad.append("本地样例 CSV 的 sha256 与 KIT_MANIFEST.json 不符")
    ids = df["row_id"].tolist()
    if ids != sample_ids:
        extra, missing = set(ids) - set(sample_ids), set(sample_ids) - set(ids)
        bad.append(f"Tier {tier} row_id 与样例不同：多 {len(extra)} 个、少 {len(missing)} 个、集合{'相同' if not extra and not missing else '不同'}")
    if df["episode"].isna().any():
        bad.append(f"Tier {tier} 有 {int(df['episode'].isna().sum())} 个 row_id 解析不了")
    if set(df["world"].dropna().unique()) != {0} or set(df["slot"].dropna().unique()) != {0}:
        bad.append(f"Tier {tier} 出现了 w000、o0 以外的世界或物体")
    if sorted(df["episode"].dropna().unique().tolist()) != list(range(30)):
        bad.append(f"Tier {tier} episode 不是 0–29")
    for ep in range(30):
        frames = set(df.loc[df["episode"] == ep, "frame"].astype(int).tolist())
        visible = {int(f) for f in released_reference(tier, ep).frame_index}
        dropped = sorted(set(range(meta[ep]["length"])) - frames)
        invisible = sorted(set(range(meta[ep]["length"])) - visible)
        if frames != visible:
            bad.append(f"Tier {tier} ep{ep}：CSV 的帧与本档发布数据的可见帧不同（CSV 缺 {dropped[:5]}，不可见 {invisible[:5]}）")
        elif dropped:
            notes.append(f"Tier {tier} ep{ep}：发布数据 {meta[ep]['length']} 帧，CSV {len(frames)} 行；少的帧 {dropped} 正是本档发布 parquet 里物体不可见的帧")
    return bad, notes


def check_c3(tier, df) -> list:
    import numpy as np

    bad = []
    vals = df[POS + QUAT].to_numpy(dtype=float)
    if not np.isfinite(vals).all():
        bad.append(f"Tier {tier} 有 {int((~np.isfinite(vals)).any(axis=1).sum())} 行含非有限值")
    off = np.abs(np.linalg.norm(df[QUAT].to_numpy(dtype=float), axis=1) - 1.0)
    if (off > QUAT_TOL).any():
        k = int(np.argmax(off))
        bad.append(f"Tier {tier} 有 {int((off > QUAT_TOL).sum())} 行四元数模长偏离 1 超过 {QUAT_TOL}（最大 {off.max():.3g}，{df['row_id'].iloc[k]}）")
    return bad


def conversion_reference(tier, ep, frames):
    """打包器换算用的参考（Tier 1：发布数据；Tier 2：v5.4 帧参考），取 CSV 的帧。"""
    import numpy as np

    if tier == 1:
        ref = released_reference(1, ep)
        idx, poses = ref.frame_index, ref.pose_xyzw
    else:
        with np.load(FRAME_REF_T2 / f"episode_{ep:06d}.npz", allow_pickle=False) as d:
            assert str(d["coordinate_frame"].item()) == "released_track2_world" and int(d["episode_index"].item()) == ep
            idx, poses = np.asarray(d["frame_index"]), np.asarray(d["pose_xyzw"], dtype=float)
    lookup = {int(f): i for i, f in enumerate(idx)}
    return poses[[lookup[int(f)] for f in frames], 0, :]


def quat_mul(a, b):
    import numpy as np

    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], axis=-1)


def rotmat(q):
    import numpy as np

    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def unit(q):
    import numpy as np

    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def check_c4(tier, df) -> tuple:
    import numpy as np

    bad, worst = [], []
    for ep in range(30):
        block = df[df["episode"] == ep].sort_values("frame")
        frames = block["frame"].astype(int).to_numpy()
        _, ex, _ = export_files(tier, ep)
        ex = ex.set_index("reference_frame")
        missing = [int(f) for f in frames if f not in ex.index]
        if missing:
            bad.append(f"Tier {tier} ep{ep}：导出里缺 CSV 的帧 {missing[:5]}")
            continue
        rows = ex.loc[frames]
        ach_p = rows[["achieved_position_x", "achieved_position_y", "achieved_position_z"]].to_numpy(float)
        ach_q = unit(rows[["achieved_quaternion_x", "achieved_quaternion_y", "achieved_quaternion_z", "achieved_quaternion_w"]].to_numpy(float))
        ref_p = rows[["reference_position_x", "reference_position_y", "reference_position_z"]].to_numpy(float)
        ref_q = unit(rows[["reference_quaternion_x", "reference_quaternion_y", "reference_quaternion_z", "reference_quaternion_w"]].to_numpy(float))
        conv = conversion_reference(tier, ep, frames)
        rel_p, rel_q = conv[:, :3], unit(conv[:, 3:])
        rq = unit(quat_mul(rel_q[0], ref_q[0] * np.array([-1.0, -1.0, -1.0, 1.0])))
        R = rotmat(rq)
        off = rel_p[0] - R @ ref_p[0]
        pred_p = ach_p @ R.T + off
        pred_q = quat_mul(np.broadcast_to(rq, ach_q.shape), ach_q)
        got_p = block[POS].to_numpy(float)
        got_q = unit(block[QUAT].to_numpy(float))
        dp = float(np.linalg.norm(pred_p - got_p, axis=1).max())
        da = float((2 * np.arccos(np.clip(np.abs(np.sum(pred_q * got_q, axis=1)), 0, 1))).max())
        angle = float(np.degrees(2 * np.arccos(min(1.0, abs(float(rq[3]))))))
        worst.append((ep, dp, da, float(np.linalg.norm(off)), angle))
        if dp > C4_POS_TOL or da > C4_ANG_TOL:
            bad.append(f"Tier {tier} ep{ep}：按打包器的构造复算 CSV，位置最大差 {dp:.3g} m、朝向最大差 {da:.3g} rad，超过 {C4_POS_TOL} m / {C4_ANG_TOL} rad")
    return bad, worst


# ---------------------------------------------------------------- 只报
def kit_line(rel: str, pattern: str) -> str:
    lines = (KIT / rel).read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines, 1):
        if re.search(pattern, line):
            return f"{rel}:{i}: {line.strip()}"
    return f"{rel}: 没找到 {pattern}"


def report_r1(tier, remote) -> list:
    out = [f"  背景：{kit_line('config/tracks.json', 'resets to the same frame and runs a deterministic policy')}",
           f"  背景：{kit_line('flash_chord/src/flash_chord/configs/evaluation/flash_sac.yaml', '^deterministic')}"]
    flags = []
    for ep in range(30):
        j, _, _ = export_files(tier, ep)
        m = j["metrics"]
        r = remote[(tier, ep)]
        one_mp = m.get("maniptrans_sr")
        mp4096 = float(r["mp_sr"])
        out.append(f"  ep{ep:<2} {r['run']:<36} 1 世界：chord {m.get('chord_sr'):.0f} MP {one_mp:.0f} SP {m.get('spider_sr_uncentered'):.0f} "
                   f"ADD-AUC {m['add_auc']:.4f} MPPE {m['mppe_cm']:.2f} cm ｜ 4096 世界：chord {float(r['chord_sr']):.3f} MP {mp4096:.3f} "
                   f"SP {float(r['sp_sr']):.3f} ADD-AUC {float(r['add_auc']):.4f} MPPE {float(r['mppe_cm']):.2f} cm")
        if mp4096 >= 0.5 and one_mp == 0:
            flags.append(f"ep{ep}（4096 世界 MP-SR {mp4096:.3f}，这 1 个世界失败）")
        if mp4096 < 0.05 and one_mp == 1:
            flags.append(f"ep{ep}（4096 世界 MP-SR {mp4096:.3f}，这 1 个世界成功）")
    ones = [float(export_files(tier, ep)[0]["metrics"]["maniptrans_sr"]) for ep in range(30)]
    means = [float(remote[(tier, ep)]["mp_sr"]) for ep in range(30)]
    out.append(f"  1 个世界的 MP-SR 成功 {int(sum(ones))} / 30；4096 世界 MP-SR 的 30 条平均 {sum(means) / 30:.3f}")
    out.append(f"  标出的 episode：{flags or '无'}")
    return out


def report_r2_tier1(df) -> list:
    """Tier 1：按 kit 打包器同一套算法、干净参考与网格顶点，逐条算本地分数（同 pack_submission.py 的 crosscheck）。"""
    import numpy as np
    import pandas as pd
    from v2dlb import object_pose as op
    from v2dlb.vertices import mesh_path_for, sample_vertices

    spec = json.loads((KIT / "config/tracks.json").read_text(encoding="utf-8"))["tracks"]["track_2_tier1"]
    rows = []
    for ep in range(30):
        ref = released_reference(1, ep)
        for step, frame in enumerate(ref.frame_index):
            for slot, name in enumerate(ref.object_names):
                rows.append((f"track_2/e{ep:06d}/w000/f{int(frame):06d}/o{slot}", ep, 0, int(frame), slot, name, *ref.pose_xyzw[step, slot]))
    sol = pd.DataFrame(rows, columns=["row_id", "episode_index", "world_index", "frame_index", "object_slot", "object_name", *op._REF_COLUMNS])
    merged = sol.merge(df[["row_id", *POS, *QUAT]], on="row_id", how="left")
    verts = {n: sample_vertices(mesh_path_for(DATA[1] / "mesh", str(n))) for n in sorted(merged["object_name"].unique())}
    per = {m: {} for m in spec["metrics"]}
    for ep, g in merged.groupby("episode_index", sort=True):
        ach, ref_t, ids, nm = op._episode_tensors(g)
        ach = op._normalise_quaternions(ach, "submission", ValueError)
        ref_t = op._normalise_quaternions(ref_t, "reference", ValueError)
        v = np.stack([verts[str(n)] for n in nm])
        for metric in spec["metrics"]:
            val = op.episode_metric(metric, ach, ref_t, ids, v, align_initial_pose=bool(spec.get("align_initial_pose", False)))
            if not np.isnan(val):
                per[metric][int(ep)] = val
    out = [f"  口径：{kit_line('v2dlb/vertices.py', 'own sampler')}",
           f"        {kit_line('v2dlb/vertices.py', 'treat any small')}",
           f"        {kit_line('v2dlb/vertices.py', 'ADD_VERTEX_COUNT = 500')}",
           f"        {kit_line('metric_code/track_2/AUC.py', 'auc_vertex_samples_b64')}",
           f"  指标 {spec['metrics']}；align_initial_pose={bool(spec.get('align_initial_pose', False))}"]
    out += [f"  ep{ep:<2} " + "  ".join(f"{m} {per[m].get(ep, float('nan')):.4f}" for m in spec["metrics"]) for ep in range(30)]
    out.append("  汇总（30 条平均）：" + "  ".join(f"{m} {np.mean(list(per[m].values())):.4f}" for m in spec["metrics"]))
    return out


def report_r3(tier, df) -> list:
    import numpy as np

    out, flagged = [], []
    for ep in range(30):
        block = df[df["episode"] == ep].sort_values("frame")
        ref = released_reference(tier, ep)
        lookup = {int(f): i for i, f in enumerate(ref.frame_index)}
        fr = block["frame"].astype(int).tolist()
        keep = [i for i, f in enumerate(fr) if f in lookup]
        inp = ref.pose_xyzw[[lookup[fr[i]] for i in keep], 0, :3]
        d = np.linalg.norm(block[POS].to_numpy(float)[keep] - inp, axis=1)
        out.append(f"  ep{ep:<2} 与本档发布输入的位置差：中位 {np.median(d):.3f} m，最大 {d.max():.3f} m")
        if d.max() > 1.0:
            flagged.append(f"ep{ep}（最大 {d.max():.2f} m）")
    return out + [f"  最大值超过 1 m 的：{flagged or '无'}"]


# ---------------------------------------------------------------- 各模式
def kit_hashes() -> dict:
    return {"eval_script": sha256(KIT / "flash_chord/scripts/evaluate_policy.py"), "packer": sha256(KIT / "tools/pack_submission.py"),
            "uploader": sha256(KIT / "submit/v2d_submit.py")}


def run_tier_checks(tier, df, emap, remote, meta, sample_ids, sample_sha_ok, hashes) -> tuple:
    fails, lines = [], []
    c1 = check_c1(tier, df, emap, remote, meta, hashes)
    c2, c2notes = check_c2(tier, df, sample_ids, meta, sample_sha_ok)
    c3 = check_c3(tier, df)
    c4, worst = check_c4(tier, df)
    for name, bad in (("C1", c1), ("C2", c2), ("C3", c3), ("C4", c4)):
        lines.append(f"Tier {tier} {name}：{'过' if not bad else '不过'}")
        lines += [f"  {b}" for b in bad[:12]]
        fails += [f"{name} {b}" for b in bad]
    lines += [f"  C2 附注：{n}" for n in c2notes]
    if worst:
        p = max(worst, key=lambda x: x[1])
        a = max(worst, key=lambda x: x[2])
        lines.append(f"  C4 最大差：位置 {p[1]:.3g} m（ep{p[0]}），朝向 {a[2]:.3g} rad（ep{a[0]}）；评测器系到发布系的变换：平移 "
                     f"{min(w[3] for w in worst):.2f}–{max(w[3] for w in worst):.2f} m，旋转 {min(w[4] for w in worst):.1f}–{max(w[4] for w in worst):.1f}°")
    return fails, lines


def mode_tier(tier: int) -> int:
    if tier == 2:
        sys.path.insert(0, str(ROOT / "libs/v2d_infra"))
        from tier_guard import forbid_tier1

        forbid_tier1()  # 净室：此后本进程读 Tier 1 目录就抛错
        print("净室：已调 forbid_tier1()")
    sys.path.insert(0, str(KIT))
    import pandas as pd

    df = load_csv(CSV[tier])
    emap, remote, meta = export_map(), remote_rows(), released_meta(tier)
    sample_ids = pd.read_csv(SAMPLE, usecols=["row_id"], dtype=str)["row_id"].tolist()
    manifest = json.loads((KIT / "KIT_MANIFEST.json").read_text(encoding="utf-8"))["files"]
    sample_sha_ok = sha256(SAMPLE) == manifest["data/track_2_sample_submission.csv"]
    hashes = kit_hashes()
    print(f"Tier {tier} CSV：{len(df)} 行，sha256 {sha256(CSV[tier])}；样例 {len(sample_ids)} 行（sha256 与 KIT_MANIFEST {'相同' if sample_sha_ok else '不同'}）；"
          f"导出清单 {len(emap)} 行；远端记录 {len(remote)} 行")
    fails, lines = run_tier_checks(tier, df, emap, remote, meta, sample_ids, sample_sha_ok, hashes)
    print("\n".join(lines))
    print(f"\n== R1 Tier {tier} 每条 episode：1 个世界的导出对 4096 世界评测（两边都按本档评测器参考算）")
    print("\n".join(report_r1(tier, remote)))
    if tier == 1:
        print("\n== R2 Tier 1 按 kit 口径（干净参考、发布网格的顶点）逐条本地分数；排行榜以 Kaggle 为准")
        print("\n".join(report_r2_tier1(df)))
    print(f"\n== R3 Tier {tier} 与本档发布输入的位置差（Tier 1 的发布输入是干净参考，Tier 2 是带噪输入）")
    print("\n".join(report_r3(tier, df)))
    print(f"\n判据不过的：{len(fails)} 处")
    for f in fails[:20]:
        print(f"  {f}")
    print(f"SUBMISSION_CHECK {'PASS' if not fails else 'FAIL'}")
    return 0 if not fails else 1


def mode_cross() -> int:
    import pandas as pd

    cols = ["row_id", *POS, *QUAT, "checkpoint_sha256", "code_commit_url"]
    a, b = (pd.read_csv(CSV[t], usecols=cols, dtype={"row_id": str, "checkpoint_sha256": str, "code_commit_url": str}) for t in (1, 2))
    fails = []
    ck1, ck2 = set(a["checkpoint_sha256"]), set(b["checkpoint_sha256"])
    print(f"checkpoint：Tier 1 {len(ck1)} 个、Tier 2 {len(ck2)} 个，交集 {len(ck1 & ck2)} 个")
    if ck1 & ck2:
        fails.append("两档的 checkpoint 有交集")
    same_rows = bool((a["row_id"] == b["row_id"]).all())
    identical = int((a[POS + QUAT].round(6) == b[POS + QUAT].round(6)).all(axis=1).sum()) if same_rows else -1
    print(f"两档 row_id 逐行相同：{same_rows}；位姿 7 列逐行相同的行数 {identical}")
    if identical == len(a):
        fails.append("两档的位姿列逐行相同")
    urls = set(a["code_commit_url"]) | set(b["code_commit_url"])
    print(f"code_commit_url：{sorted(urls)}")
    if len(urls) != 1:
        fails.append("两档的代码链接不同")
    commit = next(iter(urls)).rsplit("/", 1)[-1]
    branches = subprocess.run(["git", "branch", "-r", "--contains", commit], capture_output=True, text=True, cwd=ROOT, check=False).stdout.split()
    print(f"提交 {commit} 所在的远端分支：{branches or '无'}")
    if not branches:
        fails.append("代码提交不在 origin 上")
    for f in fails:
        print(f"  {f}")
    print(f"SUBMISSION_CHECK {'PASS' if not fails else 'FAIL'}")
    return 0 if not fails else 1


def mode_compare(path_a: Path, path_b: Path) -> int:
    import pandas as pd

    sa, sb = sha256(path_a), sha256(path_b)
    print(f"{path_a.name} sha256 {sa}\n{path_b.name} sha256 {sb}")
    if sa == sb:
        print("逐字节相同")
        print("SUBMISSION_CHECK PASS")
        return 0
    a, b = (pd.read_csv(p, dtype=str) for p in (path_a, path_b))
    print(f"不同：行数 {len(a)} 对 {len(b)}，列 {list(a.columns) == list(b.columns)}")
    if len(a) == len(b) and list(a.columns) == list(b.columns):
        for c in a.columns:
            n = int((a[c] != b[c]).sum())
            if n:
                k = int((a[c] != b[c]).to_numpy().argmax())
                print(f"  列 {c}：{n} 行不同，例：{a.loc[k, 'row_id']} {a.loc[k, c]} 对 {b.loc[k, c]}")
    print("SUBMISSION_CHECK FAIL")
    return 1


def mode_selftest() -> int:
    sys.path.insert(0, str(KIT))
    import pandas as pd

    df = load_csv(CSV[1])
    emap, remote, meta = export_map(), remote_rows(), released_meta(1)
    sample_ids = pd.read_csv(SAMPLE, usecols=["row_id"], dtype=str)["row_id"].tolist()
    d = df.copy()
    rows = d.index[d["episode"] == 3]
    d.loc[rows, "checkpoint_sha256"] = "0" * 64
    d.loc[rows[5], QUAT] = [0.0, 0.0, 0.0, 0.5]
    d.loc[rows[9], "pos_x"] = d.loc[rows[9], "pos_x"] + 0.01
    fails, _ = run_tier_checks(1, d, emap, remote, meta, sample_ids, True, kit_hashes())
    got = {f.split()[0] for f in fails}
    assert {"C1", "C3", "C4"} <= got, got
    print(f"自测：改坏 Tier 1 ep3 之后判不过的判据 {sorted(got)}（应含 C1、C3、C4）；条数 {len(fails)}")
    print("SELFTEST_OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--tier", type=int, choices=(1, 2))
    g.add_argument("--cross", action="store_true")
    g.add_argument("--compare", nargs=2, type=Path)
    g.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.tier:
        return mode_tier(a.tier)
    if a.cross:
        return mode_cross()
    if a.compare:
        return mode_compare(*a.compare)
    return mode_selftest()


if __name__ == "__main__":
    raise SystemExit(main())
