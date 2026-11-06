"""fail_onset.py 的测试（计划 agent-doc/plan-frame-onset-2026-10-14.md 第 3 节第 0 步、第 4 节）。要 numpy、pyarrow 与 pytest。

1. 起点：只有骨盆、只有物体、两者都有（取早的）、都没有（no_cross）的失败世界各自计数对；冻结期内的物体越线不算；最集中的窗口对；三类占比对；
2. 复现门：p 居中、差 0.02 时 fixed 门不过、noise 门过；差太大时 noise 也不过；
3. 参考帧数与 step_count 不符：门不过、不出计数；
4. 接触三段：摔倒世界的 t_pel 落在第一次接触之前、接触中、接触之后的不接触段，各自占比对；
5. 缺 dump 的 run 记门不过，同一次给的其余 run 照常处理，退出码 0，结尾计数对；
6. policy 里的参考路径以 /repo/ 开头时，换成 --root 下的路径；
7. 新评 JSON 的 start_frame 不是 0：门不过；
8. G4：三条里一条的骨盆阈值与另两条不同，那一条门不过、删去计数；两条各一种时不论顺序都门不过；混档拒绝；复现门的 z 值对；
9. 失败世界里冻结期内的越线不计；参考接触有 0、1 以外的值、npz 截断：这一条门不过，其余照常；Tier 3 的 run 读到 Tier 1 目录：守卫拦下、调用失败；
   起点分位用 np.percentile。
"""

import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
NAMES = ["pelvis_position_m", "pelvis_orientation_rad", "palm_position_m", "palm_orientation_rad", "object_position_m", "object_orientation_rad"]
T = 300


def make(root: Path, run: str, pattern, n_fail=50, n_ok=10, ori=0.6, mp_shift=0.0, ref_t=T, contact=None, repo_prefix=False, start_frame=0,
         pel_thr=0.5):
    """pattern：每个失败世界的 (t_obj, t_pel)，None 表示不越线；循环使用。"""
    w = n_ok + n_fail
    te = np.zeros((T, w, 6), dtype=np.float32)
    succ = np.ones(w, dtype=bool)
    for i, wi in enumerate(range(n_ok, w)):
        succ[wi] = False
        t_obj, t_pel = pattern[i % len(pattern)]
        if t_obj is not None:
            te[t_obj:, wi, 4] = 0.3
        if t_pel is not None:
            te[t_pel:, wi, 0] = 1.0
    te[10:40, 0, 4] = 0.5  # 冻结期内的物体越线，不计
    if n_fail > 4:
        te[10:40, n_ok + 4, 4] = 0.5   # 失败世界里冻结期内的越线也不计（代码审查第 6 条；pattern 第 5 项是 (None, None) 时它应当记 no_cross）
    o = root / "runs/flash_chord" / run
    (o / "eval_dump").mkdir(parents=True)
    np.savez_compressed(o / "eval_dump/eval_1_w9.npz", tracking_error=te.astype(np.float16), error_names=np.array(NAMES),
                        criteria_error_names=np.array(NAMES), thresholds=np.array([pel_thr, 1.0, 0.3, 1.0, 0.1, ori]),
                        active_after_step=np.array([-1, -1, 50, 50, 50, 50]), non_finite_worlds=np.array([], dtype=np.int64),
                        mp_success=succ, mp_position_m=np.where(succ, 0.01, 0.3), control_fps=np.float64(30.0))
    mp = float(succ.mean())
    new = {"step_count": T, "world_count": w, "checkpoint_sha256": "abc", "start_frame": start_frame,
           "reference_tracking": {"reference_end_fraction": 1.0},
           "metrics": {"maniptrans_sr": mp, "mppe_cm": 5.0, "object_position_error_std_m": 0.1}}
    old = {"step_count": T, "world_count": w, "checkpoint_sha256": "abc",
           "metrics": {"maniptrans_sr": mp + mp_shift, "mppe_cm": 5.0, "object_position_error_std_m": 0.1}}
    (o / "eval_dump/eval_1_w9.json").write_text(json.dumps(new))
    (o / "eval_1_w9.json").write_text(json.dumps(old))
    rel = f"runs/motion/{run}/data.parquet"
    (root / rel).parent.mkdir(parents=True)
    c = np.zeros((2, ref_t)) if contact is None else contact
    pq.write_table(pa.table({"hand_sides": [["left", "right"]], "hand_contact_active": [c.tolist()]}), root / rel)
    cfg = {"task": {"parquet": "'" + ("/repo/" + rel if repo_prefix else str(root / rel)) + "'"}}
    head = json.dumps({"__metadata__": {"user.resolved_config_json": json.dumps(cfg)}}).encode()
    (o / "policy_1.safetensors").write_bytes(struct.pack("<Q", len(head)) + head)


def call(root: Path, *runs, repro="noise"):
    out = root / "out"
    p = subprocess.run([sys.executable, str(HERE / "fail_onset.py"), str(out), *runs, "--steps", "1", "--worlds", "9", "--root", str(root),
                        "--repro", repro], capture_output=True, text=True)
    return p, {r: json.loads((out / f"onset_{r}.json").read_text()) for r in runs if (out / f"onset_{r}.json").exists()}


def test_onset_counts(tmp_path):
    r = "t3-ep7-w2048-cap10m-r3a-hm-ml40"
    make(tmp_path, r, [(60, 120), (150, 70), (None, 90), (200, None), (None, None)])
    p, d = call(tmp_path, r)
    assert p.returncode == 0, p.stdout + p.stderr
    d = d[r]
    assert d["gate_ok"], d["problems"]
    assert (d["n_failed"], d["n_onset"], d["n_no_cross"], d["n_fallen"]) == (50, 40, 10, 30)
    assert [d["n_on"][t] for t in (60, 70, 90, 200)] == [10, 10, 10, 10] and sum(d["n_on"]) == 40
    assert d["n_on"][20] == 0 and sum(d["n_obj"]) == 30 and d["n_obj"][150] == 10
    assert d["mode_window"] == [55, 70, 20]
    assert d["q_on"][2] == 80.0   # np.percentile 的中位：60、70、90、200 各 10 个
    of, pf, si = d["classes"]["25"]
    assert abs(of - 1 / 3) < 1e-9 and abs(pf - 2 / 3) < 1e-9 and si == 0
    assert "FAIL_ONSET_DONE（门过 1、门不过 0）" in p.stdout


def test_repro_gate(tmp_path):
    r = "t3-ep21-w2048-cap4m-r3a-hm-mf4090"
    make(tmp_path, r, [(60, 120)], n_fail=10, n_ok=10, mp_shift=0.02)
    _, d = call(tmp_path, r, repro="fixed")
    assert not d[r]["gate_ok"] and any("MP-SR 差" in x for x in d[r]["problems"])
    _, d = call(tmp_path, r, repro="noise")
    assert d[r]["gate_ok"], d[r]["problems"]
    assert abs(d[r]["z"] - (-0.02) / (2 * 0.52 * 0.48 / 20) ** 0.5) < 1e-6   # p 取原评 0.52，N 取新评 20 个世界
    big = "t3-ep6-w2048-cap10m-r3a-hm-ml40"
    make(tmp_path, big, [(60, 120)], n_fail=10, n_ok=10, mp_shift=0.49)
    _, d = call(tmp_path, big, repro="noise")
    assert not d[big]["gate_ok"]


def test_ref_frames(tmp_path):
    r = "t3-ep4-w2048-cap10m-r3a-hm-ml40"
    make(tmp_path, r, [(60, 120)], ref_t=T - 1)
    p, d = call(tmp_path, r)
    assert p.returncode == 0
    assert not d[r]["gate_ok"] and any("参考帧数" in x for x in d[r]["problems"]) and "n_on" not in d[r]
    assert "门不过，不出计数" in p.stdout


def test_contact_phase(tmp_path):
    r = "t1-ep25-w2048-cap10m-ml40"
    c = np.zeros((2, T))
    c[1, 80:130] = 1
    c[1, 180:250] = 1
    make(tmp_path, r, [(None, 70), (None, 100), (None, 150)], n_fail=30, ori=1.5, contact=c)
    _, d = call(tmp_path, r, repro="fixed")
    d = d[r]
    assert d["gate_ok"], d["problems"]
    assert d["first_contact"] == 80
    ph = d["pel_phase"]
    assert all(abs(ph[k] - 1 / 3) < 1e-9 for k in ("before", "during", "gap")), ph


def test_missing_dump(tmp_path):
    good = "t1-ep4-w2048-cap10m-ml40"
    make(tmp_path, good, [(None, 70)], ori=1.5)
    (tmp_path / "runs/flash_chord/t1-ep9-w2048-cap10m-ml40").mkdir(parents=True)
    p, d = call(tmp_path, "t1-ep9-w2048-cap10m-ml40", good, repro="fixed")
    assert p.returncode == 0, p.stderr
    assert not d["t1-ep9-w2048-cap10m-ml40"]["gate_ok"] and d[good]["gate_ok"]
    assert p.stdout.rstrip().endswith("FAIL_ONSET_DONE（门过 1、门不过 1）")


def test_mixed_tiers_refused(tmp_path):
    make(tmp_path, "t1-ep4-w2048-cap10m-ml40", [(None, 70)], ori=1.5)
    make(tmp_path, "t3-ep4-w2048-cap10m-r3a-hm-ml40", [(60, 120)])
    p, d = call(tmp_path, "t1-ep4-w2048-cap10m-ml40", "t3-ep4-w2048-cap10m-r3a-hm-ml40")
    assert p.returncode == 2 and d == {} and "一次只收同一档" in p.stderr


def test_frame_gate(tmp_path):
    r = "t3-ep13-w2048-cap4m-r3a-hm-mf4090"
    make(tmp_path, r, [(60, 120)], start_frame=5)
    p, d = call(tmp_path, r)
    assert not d[r]["gate_ok"] and any("start_frame 5" in x for x in d[r]["problems"]), d[r]["problems"]


def test_g4(tmp_path):
    runs = [f"t3-ep{e}-w2048-cap4m-r3a-hm-mf4090" for e in (1, 2, 3)]
    make(tmp_path, runs[0], [(60, 120)])
    make(tmp_path, runs[1], [(60, 120)])
    make(tmp_path, runs[2], [(60, 120)], pel_thr=0.6)
    p, d = call(tmp_path, *runs)
    assert d[runs[0]]["gate_ok"] and d[runs[1]]["gate_ok"], (d[runs[0]]["problems"], d[runs[1]]["problems"])
    assert not d[runs[2]]["gate_ok"] and any(x.startswith("G4") for x in d[runs[2]]["problems"])
    assert "n_on" not in d[runs[2]] and "n_on" in d[runs[0]]   # 门不过的删去计数
    assert p.stdout.rstrip().endswith("FAIL_ONSET_DONE（门过 2、门不过 1）")


def test_g4_tie(tmp_path):
    a, b = "t3-ep1-w2048-cap4m-r3a-hm-mf4090", "t3-ep2-w2048-cap4m-r3a-hm-mf4090"
    make(tmp_path, a, [(60, 120)])
    make(tmp_path, b, [(60, 120)], pel_thr=0.6)
    for order in ((a, b), (b, a)):   # 两条各一种：不论顺序，两条都门不过
        _, d = call(tmp_path, *order)
        assert not d[a]["gate_ok"] and not d[b]["gate_ok"], (order, d[a]["problems"], d[b]["problems"])


def test_contact_values(tmp_path):
    r = "t3-ep14-w2048-cap4m-r3a-hm-mf4090"
    c = np.zeros((2, T))
    c[1, 50:60] = 0.5
    make(tmp_path, r, [(60, 120)], contact=c)
    _, d = call(tmp_path, r)
    assert not d[r]["gate_ok"] and any("0、1 以外" in x for x in d[r]["problems"]), d[r]["problems"]


def test_corrupt_npz(tmp_path):
    bad, good = "t3-ep16-w2048-cap4m-r3a-hm-mf4090", "t3-ep17-w2048-cap4m-r3a-hm-mf4090"
    make(tmp_path, bad, [(60, 120)])
    make(tmp_path, good, [(60, 120)])
    f = tmp_path / f"runs/flash_chord/{bad}/eval_dump/eval_1_w9.npz"
    f.write_bytes(f.read_bytes()[:200])   # 截断
    p, d = call(tmp_path, bad, good)
    assert p.returncode == 0 and not d[bad]["gate_ok"] and d[good]["gate_ok"], (p.stderr, d.get(bad, {}).get("problems"))


def test_tier_guard(tmp_path):
    root = tmp_path / "tier_1_multiview_caption"   # Tier 3 的 run 读到带 Tier 1 目录名的路径：守卫拦下，整次调用失败
    r = "t3-ep19-w2048-cap4m-r3a-hm-mf4090"
    make(root, r, [(60, 120)])
    p = subprocess.run([sys.executable, str(HERE / "fail_onset.py"), str(tmp_path / "out"), r, "--steps", "1", "--worlds", "9", "--root", str(root),
                        "--repro", "noise"], capture_output=True, text=True)
    assert p.returncode != 0 and "Tier1AccessError" in p.stderr, (p.returncode, p.stderr[-500:])


def test_repo_prefix(tmp_path):
    r = "t3-ep8-w2048-cap10m-r3a-hm-ml40"
    make(tmp_path, r, [(60, 120)], repo_prefix=True)
    _, d = call(tmp_path, r)
    assert d[r]["gate_ok"], d[r]["problems"]
    assert d[r]["parquet"] == str(tmp_path / f"runs/motion/{r}/data.parquet")
