"""diag_report.py 与 diag_triggers.py 的合成数据测试（只用 numpy；暂无契约 ID：诊断工具不在交付链上）。

参考绕某轴在第 209–420 帧匀速转 300°（默认竖直轴；另测倾斜 20° 的轴）。11 个世界：
0–3 跟到 90%（成功）；4–5 只转 40°；6–7 不转但每帧 ±0.5° 来回抖动，世界 6 另有一对 ±40° 跳变；8–9 先转 60° 再转回 0°；
10 第 0 帧就偏 5°、之后始终落后参考 85.5°（成功，但第 336 帧只转了约 90°：固定 94° 的旧规则会误判，逐世界下界不会）。
随机翻转四元数正负号。环境自带的朝向误差按位姿算，应当对得上。
检查：各道闸通过与被拦（G0、G1、G2、G3、G4、G5、G6 各改坏一处）、G6 无成功世界时“不适用”、分类计数、主轴夹角、跳变计数、
Q0 各段贡献相加等于全程、summary_json、SVG 标签配对、diag_triggers 能读报告并给出四项判定、--frame-scale 只改 Q1、Q2 按关键帧列数的两行、Q2b 落后量与 --phases、summary_json 的新字段。
用法：python stages/evaluate/test_diag_report.py
"""
import json
import re
import subprocess
import sys
import tempfile
from html.parser import HTMLParser
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
T, W, S0, S1 = 553, 11, 209, 420
THR = np.arange(0.01, 0.10, 0.01)
PF = 336  # 合成参考在这一帧离初始朝向最接近 180°


def auc(x):
    x = np.asarray(x, np.float64).ravel()
    return float(np.trapezoid([np.mean(x < t) for t in THR], x=np.linspace(0, 1, THR.size)))


def axis_angle_quat(axis, deg):
    h = np.radians(deg) / 2
    s = np.sin(h)[..., None] * np.asarray(axis)[None, :]
    return np.concatenate([s, np.cos(h)[..., None]], -1)  # xyzw


def ramp(total, a=S0, b=S1):
    t = np.arange(T)
    return np.clip((t - a) / (b - a), 0, 1) * total


def geodesic_deg(qa, qb):
    return np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(qa * qb, -1)), 0, 1)))


def build(tmp: Path, tilt_deg=0.0, **broken):
    rng = np.random.default_rng(0)
    axis = np.array([0.0, np.sin(np.radians(tilt_deg)), np.cos(np.radians(tilt_deg))])
    ref_deg = ramp(300.0)
    qr = axis_angle_quat(axis, ref_deg)
    ach = np.zeros((T, W, 1, 7))
    ach[:, :, 0, :3] = 0.02
    for w in range(W):
        if w < 4:
            deg = 0.9 * ref_deg
        elif w < 6:
            deg = ramp(40.0)
        elif w < 8:
            deg = np.cumsum(np.where(np.arange(T) % 2 == 0, 0.5, -0.5))
            if w == 6:
                deg = deg + np.where(np.arange(T) == 100, 40.0, 0.0)
        elif w < 10:
            deg = np.where(np.arange(T) <= 300, ramp(60.0, S0, 300), 60.0 - ramp(60.0, 300, S1))
        else:
            deg = np.maximum(5.0, ref_deg - 85.5)
        q = axis_angle_quat(axis, deg)
        q *= np.where(rng.random(T) < 0.5, -1.0, 1.0)[:, None]
        ach[:, w, 0, 3:7] = q
    ref = np.zeros((T, 1, 7))
    ref[:, 0, 3:7] = qr
    names = ["pelvis_position_m", "pelvis_orientation_rad", "palm_position_m", "palm_orientation_rad",
             "object_position_m", "object_orientation_rad"]
    err = np.full((T, W, 6), 0.01)
    ori = geodesic_deg(ach[:, :, 0, 3:7], qr[:, None])
    err[..., 5] = np.radians(ori) + (np.radians(1.0) if broken.get("env_ori") else 0.0)
    err[300:, 6:8, 2] = 0.2  # 抖动世界 palm 越线
    if broken.get("fake_success"):  # 让不转的世界 6 在误差上“成功”：与位姿矛盾，G3 与 G6 都应拦下
        err[:, 6, 2] = 0.01
        err[:, 6, 5] = 0.01
    if broken.get("all_fail"):
        err[300:, :, 2] = 0.2
    criteria = {"error_names": names, "thresholds": [0.25, 0.8, 0.15, 1.5, 0.1, 1.5],
                "active_after_step": [-1, -1, 10, 10, 10, 10],
                "causes": [["pelvis", [0, 1]], ["palm", [2, 3]], ["object", [4, 5]]]}
    thr, act = np.array(criteria["thresholds"]), np.array(criteria["active_after_step"])
    exceeded = ((np.arange(T)[:, None, None] > act[None, None, :]) & (err > thr[None, None, :])).any((0, 2))
    success_rate = float((~exceeded).mean())
    add = rng.uniform(0, 0.12, size=(T, W))
    add_static = rng.uniform(0, 0.12, size=T)
    contact = np.zeros((T, 2), bool)
    contact[S0:S1, 1] = True
    np.savez(tmp / "r.npz", achieved_pose_w=ach, reference_pose_w=ref, tracking_error=err, add=add, add_static=add_static,
             tracking_error_names=np.array(names), ref_palm_names=np.array(["left_hand_palm_link", "right_hand_palm_link"]),
             ref_palm_mesh_dist=np.full((T, 2), 0.05), contact_sides=np.array(["left", "right"]), contact_active=contact,
             completion_truncated=np.ones(W, bool), completion_progress=np.ones(W), non_finite_worlds=np.array([], np.int64),
             criteria_json=np.array(json.dumps(criteria)),
             reported_success_rate=np.float64(0.9 if broken.get("reported") else success_rate),
             policy=np.array("synthetic"), upstream_md5=np.array("{}"), wrapper_calls=np.array('{"record": 1, "score": 1}'),
             trimesh_matches_evaluator=np.bool_(True), palm_dist_vertex_source=np.array("trimesh_all_vertices"))
    metrics = {"add_auc": auc(add) + (0.01 if broken.get("add_auc") else 0.0), "chord_sr": success_rate,
               "maniptrans_sr": 0.0, "mppe_cm": 3.0}
    diag = {"metrics": metrics, "checkpoint_sha256": "x", "world_count": W, "step_count": T + (1 if broken.get("shape") else 0)}
    same = json.loads(json.dumps(diag))
    if broken.get("same_card"):  # 世界少时 chord_sr 的 z 检验很弱，另把 ADD-AUC 改掉 0.2（> 11 个世界时的容差 0.005 × √(4096/11) ≈ 0.096）
        same["metrics"]["chord_sr"] = 0.0 if success_rate > 0.5 else 1.0
        same["metrics"]["add_auc"] += 0.2
    if broken.get("same_card_small"):  # 只差 0.05：在 11 个世界的容差以内，不该拦
        same["metrics"]["add_auc"] += 0.05
    (tmp / "diag.json").write_text(json.dumps(diag))
    (tmp / "same.json").write_text(json.dumps(same))
    return success_rate


def run(tmp: Path, expect=(210, 420, 270), pf=PF, svg=False, extra=()):
    cmd = [sys.executable, str(HERE / "diag_report.py"), str(tmp / "r.npz"), "--diag-json", str(tmp / "diag.json"),
           "--same-card-json", str(tmp / "same.json"), "--segments", str(S0), str(S1), "--expect-ref", *map(str, expect),
           "--predict-frame", str(pf)]
    if svg:
        cmd += ["--svg", str(tmp / "c.svg")]
    cmd += list(extra)
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def svg_balanced(text: str) -> bool:
    stack, bad = [], []

    class P(HTMLParser):
        def handle_starttag(self, tag, attrs):
            stack.append(tag)

        def handle_startendtag(self, tag, attrs):
            pass

        def handle_endtag(self, tag):
            if stack and stack[-1] == tag:
                stack.pop()
            else:
                bad.append(tag)

    P().feed(text)
    return text.lstrip().startswith("<svg") and not stack and not bad


def main() -> int:
    fails = 0

    def check(name, ok, detail=""):
        nonlocal fails
        print(f"  {'PASS' if ok else 'FAIL'}  {name}" + ("" if ok else f"  | {detail[:500]}"))
        fails += not ok

    def line(out, key):
        return next((ln for ln in out.splitlines() if key in ln), "")

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sr = build(tmp)
        p = run(tmp, svg=True)
        out = p.stdout
        check("成功率 5/11（世界 0–3 与 10）", abs(sr - 5 / 11) < 1e-12, str(sr))
        check("全部闸通过、退出码 0", p.returncode == 0 and "VALIDITY PASS" in out and out.count("  FAIL  ") == 0, out[:1200] + p.stderr[-400:])
        check("G1 判为逐位相同", "包装前后逐位相同" in out, out[:600])
        g6 = line(out, "G6 几何自检")
        m = re.search(r"转离初始朝向最小 ([0-9.]+)°", g6)
        check("G6：贴着下界的世界放行（最小转角 < 94° 仍通过）", "PASS" in g6 and bool(m) and float(m.group(1)) < 94, g6)
        cls = line(out, "分类（事先登记）")
        check("四类计数 5/4/2/0", all(s in cls for s in ("净转过半 5", "转过但没保持或没跟上 4", "基本没转 2", "其他 0")), cls)
        check("主轴竖直", "竖直方向夹角 0.0°" in out)
        check("跳变 2 次", "跳变 2 次" in out, line(out, "跳变"))
        contrib = [float(x) for x in re.findall(r"贡献 ([0-9.]+) / ", out)]
        full = float(re.search(r"全程：policy ([0-9.]+)", out).group(1))
        check("Q0 三段贡献相加等于全程", len(contrib) == 3 and abs(sum(contrib) - full) < 3e-4, f"{contrib} vs {full}")
        sj = line(out, "summary_json ")
        s = json.loads(sj.strip()[len("summary_json "):]) if sj else {}
        ok_sj = s.get("valid") is True and s.get("success") == 5 and "right" in s.get("contact_palm_mesh_dist_cm", {})
        check("summary_json 可解析且内容对", ok_sj, sj)
        check("SVG 标签配对完整", svg_balanced((tmp / "c.svg").read_text()))
        q2b = line(out, "世界中位：第 209 帧")
        check("Q2b 落后量：第 209 帧 0、第 420 帧中位 260（11 个世界排序后的第 6 个）", "第 209 帧 +0" in q2b and "第 420 帧 +260.0" in q2b, q2b)
        check("Q2b 净转过半的 5 个世界（0–3 落后 30°、10 落后 85.5°）中位 30", "净转过半的 5 个世界：第 420 帧落后量中位 +30.0" in out,
              line(out, "净转过半的"))
        ok_new = (s.get("frames") == T and s.get("segments") == [S0, S1] and abs(s.get("ref_net_deg", 0) - 300) < 0.5
                  and abs(s.get("window_cross_share", 0) - 6 / 11) < 1e-9 and s.get("phase_lag_deg") == [] and s.get("half_rotated") == 5
                  and "first_fail_causes" in s)
        check("summary_json 新字段：帧数、关注段、参考净转 300°、段内越线 6/11（世界 4–9）", ok_new, sj)
        pp = run(tmp, extra=("--phases", "209", "300", "420"))
        sp = json.loads(line(pp.stdout, "summary_json ").strip()[len("summary_json "):] or "{}")
        check("--phases 209 300 420：两段各一行、summary 里两段落后量", pp.returncode == 0 and pp.stdout.count("  阶段 ") == 2
              and len(sp.get("phase_lag_deg", [])) == 2 and sp.get("phase_bounds") == [209, 300, 420], pp.stdout[-900:])
        pb = run(tmp, extra=("--phases", "300", "209"))
        check("--phases 不是升序时报错退出", pb.returncode not in (0, 3) and "升序" in pb.stderr, pb.stderr[-300:])
        q1 = line(out, "世界中位：")
        check("不给 --frame-scale：Q1 仍列 ep20 原帧 100 / 269 / 359", all(f"  {f}:" in q1 for f in (100, 269, 359)), q1)
        ps = run(tmp, svg=True, extra=("--frame-scale", "1.6666667"))   # 同样带 --svg，才能逐行比较其余输出
        q1s = line(ps.stdout, "世界中位：")
        keyed = ("世界中位：", "转动进度（°")   # 这两行按关键帧列数，--frame-scale 只该改它们
        same_rest = [ln for ln in ps.stdout.splitlines() if not ln.strip().startswith(keyed)] == \
            [ln for ln in out.splitlines() if not ln.strip().startswith(keyed)]
        q2s = line(ps.stdout, "转动进度（°")
        check("--frame-scale 5/3：Q1、Q2 的关键帧改列 167 / 448 / 500、不再列 100，其余输出逐行不变",
              ps.returncode == 0 and all(f"  {f}:" in q for q in (q1s, q2s) for f in (167, 448, 500))
              and "  100:" not in q1s + q2s and same_rest, q1s + " | " + q2s)
        (tmp / "main.txt").write_text(out)
        (tmp / "control.txt").write_text(out)
        tr = subprocess.run([sys.executable, str(HERE / "diag_triggers.py"), "--main", str(tmp / "main.txt"),
                             "--control", str(tmp / "control.txt")], capture_output=True, text=True, check=False)
        check("diag_triggers 给出四项判定", tr.returncode == 0 and len(re.findall(r"^(触发|未触发)  T[1-4]", tr.stdout, re.MULTILINE)) == 4,
              tr.stdout + tr.stderr)
        for key, gate in (("shape", "G0"), ("same_card", "G1"), ("add_auc", "G2"), ("env_ori", "G3"), ("reported", "G5")):
            build(tmp, **{key: True})
            p = run(tmp)
            check(f"{gate} 改坏后被拦（退出码 3、不给分类）", p.returncode == 3 and f"FAIL  {gate}" in p.stdout and "分类：闸没过" in p.stdout,
                  p.stdout[:800])
        build(tmp, same_card_small=True)
        p = run(tmp)
        check("G1 容差随世界数缩放：11 个世界时 ADD-AUC 差 0.05 放行", p.returncode == 0 and "PASS  G1" in p.stdout and "0.0965" in p.stdout,
              line(p.stdout, "G1"))
        build(tmp)
        p = run(tmp, expect=(209, 420, 269))
        check("G4 已知答案不符时被拦", p.returncode == 3 and "FAIL  G4" in p.stdout, p.stdout[:600])
        build(tmp, fake_success=True)
        p = run(tmp)
        check("G6 成功世界没转够时被拦", p.returncode == 3 and "FAIL  G6" in p.stdout, p.stdout[:800])
        build(tmp, all_fail=True)
        p = run(tmp)
        check("没有成功世界：G6 不适用、仍通过", p.returncode == 0 and "没有成功的世界，不适用" in p.stdout, p.stdout[:800])
        build(tmp, tilt_deg=20.0)
        p = run(tmp)
        cls = line(p.stdout, "分类（事先登记）")
        check("主轴倾斜 20°：夹角算对、分类不变", "竖直方向夹角 20.0°" in p.stdout and "净转过半 5" in cls and p.returncode == 0,
              p.stdout[:800])
    print(f"结果：{'全部通过' if fails == 0 else f'{fails} 项失败'}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
