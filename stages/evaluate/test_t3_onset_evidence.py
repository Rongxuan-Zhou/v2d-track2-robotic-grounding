"""t3_onset_evidence.py 的测试（计划 agent-doc/plan-frame-onset-2026-10-14.md 第 3 节第 0 步、第 4 节）。要 numpy 与 pytest；脚本在子进程里跑
（它会装档位守卫的 audit hook，装上就去不掉）。

1. 起点集中在旗标之后的 15 帧内：A = 1、B 按窗口算对、E ≥ 2；4 条都这样、机会水平 p 很小时读“集中”；
2. 进入汇总的不到 4 条：读“看不清”；旗标处处都有（B > 0.5）记“不可比”；起点中位 < 100 帧记“不可比”；
3. 门：证据表行数与 step_count 不符、布尔列有别的取值（G5）；onset 的 gate_ok 为假（fail_onset）；阈值与其他 run 不同（G4）；两条各一种阈值时都不过；
4. 按文件名筛：不是 t3 best_hm 的不读、不收；best_hm 的集少了或重复：退出码 2；文件名与 JSON 里的 run 不符：门不过；
5. 接触标签变化记 F_sw；窗口边界：旗标在起点前第 15 帧算、前第 16 帧不算；
6. ep04 已确认的头尾颠倒帧之后 30 帧内的起点数；
7. 命中够 2/3 但机会水平 p > 0.05 时不读“集中”；
8. 每个用到的列单独置位时只让对应的旗标命中（代码审查第 8 条）；
9. 证据目录在 Tier 1 目录名下：守卫拦下、调用失败。
"""

import csv
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
T = 300
THR = [0.5, 1.0, 0.3, 1.0, 0.1, 0.6]
ACT = [-1, -1, 50, 50, 50, 50]
COLS = ["episode", "frame", "state", "label_left", "label_right", "limb_d_m", "viol_object_any", "viol_lin_held", "viol_lin_airborne",
        "viol_lin_supported", "viol_rot_torque", "viol_rot_jump", "viol_rot_jump_symmetric", "viol_rest_drift", "viol_rot_drift",
        "viol_root_acc", "turn_prev_deg", "dev_local_deg", "orient_outlier", "mask_px", "iou", "evidence"]
FLAGS = ("F_flip", "F_jump", "F_vis", "F_obj", "F_rot", "F_root", "F_sw")
COL0 = 5   # 输出行里第一个旗标的列号
F_FLIP, F_SW = COL0, COL0 + 6


def table(path: Path, ep: int, rows=T, flip=(), label_switch=(), everywhere=False, weird=(), set_at=None):
    """set_at = (列名, 值, 帧集合)：只在这些帧把这一列设成这个值。"""
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        lab = 0
        for t in range(rows):
            if t in label_switch:
                lab = 1 - lab
            r = {c: "0" for c in COLS}
            r.update(episode=ep, frame=t, state="held", label_left=lab, label_right=0, limb_d_m="0.1", mask_px=1000, iou="", evidence="ok")
            if t in flip or everywhere:
                r["orient_outlier"] = "1"
            if t in weird:
                r["orient_outlier"] = "nan"
            if set_at and t in set_at[2]:
                r[set_at[0]] = set_at[1]
            w.writerow([r[c] for c in COLS])


def onset(d: Path, run: str, at: int, n=60, gate=True, steps=T, thr=THR, name=None) -> Path:
    n_on = [0] * steps
    n_on[at] = n
    j = {"run": run, "gate_ok": gate, "problems": [] if gate else ["MP-SR 差 +0.0500 超过 0.033"], "step_count": steps,
         "act": ACT, "thr": thr, "n_on": n_on, "n_onset": n, "mode_window": [at - 15, at, n]}
    p = d / (name or f"onset_{run}.json")
    p.write_text(json.dumps(j))
    return p


def hm(ep: int) -> str:
    return f"t3-ep{ep}-w2048-cap4m-r3a-hm-mf4090"


def call(evdir: Path, *files, eps=None):
    extra = ["--expect-eps", ",".join(map(str, eps))] if eps else []
    return subprocess.run([sys.executable, str(HERE / "t3_onset_evidence.py"), str(evdir), *map(str, files), *extra],
                          capture_output=True, text=True)


def rows(p) -> dict:
    return {x.split("\t")[0]: x.split("\t") for x in p.stdout.splitlines() if x.startswith("t3-ep")}


def test_concentrated(tmp_path):
    files = []
    for ep in (1, 2, 3, 4):
        table(tmp_path / f"ep{ep:02d}.csv", ep, flip={95})
        files.append(onset(tmp_path, hm(ep), 100))
    p = call(tmp_path, *files, eps=(1, 2, 3, 4))
    assert p.returncode == 0, p.stderr
    r = rows(p)[hm(1)]
    assert r[F_FLIP] == "0.05/1.00/18.75", r   # B = 16/300（第 95–110 帧当起点时窗口里有第 95 帧），A = 1
    assert r[4] == "100%（单窗口）" and r[3].startswith("100–100（中位 100"), r
    assert "F_flip（短翻转 ≤15 帧）：集中（4/4 条 E ≥ 2.0，机会水平 p = 0.000）" in p.stdout, p.stdout
    assert "成段翻转（整段朝向错）这张表测不了" in p.stdout
    assert p.stdout.rstrip().endswith("T3_ONSET_EVIDENCE_DONE")


def test_few_runs_not_comparable_and_early(tmp_path):
    table(tmp_path / "ep01.csv", 1, flip={95})
    table(tmp_path / "ep02.csv", 2, everywhere=True)
    table(tmp_path / "ep03.csv", 3, flip={55})
    files = [onset(tmp_path, hm(1), 100), onset(tmp_path, hm(2), 100), onset(tmp_path, hm(3), 60)]   # ep3 起点中位 60 < 100
    p = call(tmp_path, *files, eps=(1, 2, 3))
    assert "F_flip（短翻转 ≤15 帧）：看不清（进入汇总的只有 1 条）" in p.stdout, p.stdout
    r = rows(p)
    assert r[hm(2)][F_FLIP] == "1.00/1.00/1.00（不可比）"
    assert r[hm(3)][F_FLIP].endswith("（不可比）") and "太早" in r[hm(3)][3]


def test_gates(tmp_path):
    table(tmp_path / "ep01.csv", 1, rows=T - 1)
    table(tmp_path / "ep02.csv", 2)
    table(tmp_path / "ep05.csv", 5, weird={10, 11})
    table(tmp_path / "ep06.csv", 6)
    table(tmp_path / "ep07.csv", 7)
    files = [onset(tmp_path, hm(1), 100), onset(tmp_path, hm(2), 100, gate=False), onset(tmp_path, hm(5), 100),
             onset(tmp_path, hm(6), 100, thr=[0.6] + THR[1:]), onset(tmp_path, hm(7), 100)]
    p = call(tmp_path, *files, eps=(1, 2, 5, 6, 7))
    out = p.stdout
    assert "门不过（G5）：证据表 ep01.csv 有 299 行" in out and "门不过（fail_onset）：MP-SR 差" in out, out
    assert "门不过（G5）：证据表 ep05.csv 的布尔列有别的取值：{'orient_outlier': ['nan']}" in out, out
    assert f"{hm(6)}\tep6\t—\t门不过（G4）" in out, out


def test_g4_tie(tmp_path):
    table(tmp_path / "ep01.csv", 1)
    table(tmp_path / "ep02.csv", 2)
    files = [onset(tmp_path, hm(1), 100), onset(tmp_path, hm(2), 100, thr=[0.6] + THR[1:])]
    out = call(tmp_path, *files, eps=(1, 2)).stdout
    assert f"{hm(1)}\tep1\t—\t门不过（G4）" in out and f"{hm(2)}\tep2\t—\t门不过（G4）" in out, out


def test_selection(tmp_path):
    table(tmp_path / "ep01.csv", 1)
    a = onset(tmp_path, hm(1), 100)
    dyn = onset(tmp_path, "t3-ep15-w2048-cap4m-r3a-dyn4-mf4090", 100)
    t1 = onset(tmp_path, "t1-ep25-w2048-cap10m-ml40", 100)
    p = call(tmp_path, a, dyn, t1, eps=(1,))
    assert p.returncode == 0, p.stderr
    assert "onset_t3-ep15-w2048-cap4m-r3a-dyn4-mf4090.json\t—\t—\t不收" in p.stdout and "onset_t1-ep25-w2048-cap10m-ml40.json\t—\t—\t不收" in p.stdout
    assert call(tmp_path, a, eps=(1, 2)).returncode == 2                  # 少了 ep2
    dup = onset(tmp_path, hm(1), 100, name=f"x/onset_{hm(1)}.json") if (tmp_path / "x").mkdir() is None else None
    assert call(tmp_path, a, dup, eps=(1,)).returncode == 2               # ep1 重复
    wrong = onset(tmp_path, hm(2), 100, name=f"onset_{hm(1)}.json")       # 文件名说 ep1、JSON 里是 ep2
    p = call(tmp_path, wrong, eps=(1,))
    assert "门不过：文件名" in p.stdout, p.stdout


def test_window_edge_and_switch(tmp_path):
    table(tmp_path / "ep01.csv", 1, flip={85})          # 起点 100：窗口 [85, 100] 含第 85 帧
    table(tmp_path / "ep02.csv", 2, flip={84})          # 第 84 帧在窗口外
    table(tmp_path / "ep03.csv", 3, label_switch={90})  # 第 90 帧接触标签变化
    files = [onset(tmp_path, hm(ep), 100) for ep in (1, 2, 3)]
    r = rows(call(tmp_path, *files, eps=(1, 2, 3)))
    assert r[hm(1)][F_FLIP].split("/")[1] == "1.00"
    assert r[hm(2)][F_FLIP].split("/")[1] == "0.00"
    assert r[hm(3)][F_SW].split("/")[1] == "1.00"


def test_known_flip(tmp_path):
    table(tmp_path / "ep04.csv", 4, rows=700)
    a = onset(tmp_path, "t3-ep4-w2048-cap10m-r3a-hm-ml40", 320, steps=700)
    p = call(tmp_path, a, eps=(4,))
    assert "ep4 已确认的头尾颠倒帧 313 之后 30 帧内的起点 60 个，占有起点的 100.0%；同样宽度的机会占比 4.4%" in p.stdout, p.stdout
    assert "头尾颠倒帧 509 之后 30 帧内的起点 0 个" in p.stdout


def test_chance_level_blocks(tmp_path):
    files = []
    for ep in (1, 2, 3, 4):   # 旗标在第 1、34、…、298 帧：B = 146/300 ≈ 0.487，E ≈ 2.05；4 条全中的机会 0.487⁴ ≈ 0.056 > 0.05，不读“集中”
        table(tmp_path / f"ep{ep:02d}.csv", ep, flip=set(range(1, 300, 33)))
        files.append(onset(tmp_path, hm(ep), 100))
    p = call(tmp_path, *files, eps=(1, 2, 3, 4))
    line = next(x for x in p.stdout.splitlines() if x.startswith("  F_flip"))
    assert line.startswith("  F_flip（短翻转 ≤15 帧）：看不清（E ≥ 2.0 的 4 条、机会水平 p = 0.056"), line


def test_each_column(tmp_path):
    cases = [("orient_outlier", "1", {"F_flip"}), ("viol_rot_jump", "1", {"F_jump", "F_rot"}), ("evidence", "off_image", {"F_vis"}),
             ("evidence", "no_mask+outlier", {"F_vis"}), ("viol_object_any", "1", {"F_obj"}), ("viol_rot_torque", "1", {"F_rot"}),
             ("viol_rot_jump_symmetric", "1", {"F_rot"}), ("viol_rot_drift", "1", {"F_rot"}), ("viol_rest_drift", "1", set()),
             ("viol_root_acc", "1", {"F_root"}), ("label_right", "1", {"F_sw"})]
    for i, (col, val, want) in enumerate(cases):
        d = tmp_path / f"c{i}"
        d.mkdir()
        table(d / "ep01.csv", 1, set_at=(col, val, {95}))
        r = rows(call(d, onset(d, hm(1), 100), eps=(1,)))[hm(1)]
        hit = {f for k, f in enumerate(FLAGS) if r[COL0 + k].split("/")[1] == "1.00"}
        assert hit == want, (col, val, hit, r)


def test_tier_guard(tmp_path):
    ev = tmp_path / "tier_1_multiview_caption"
    ev.mkdir()
    table(ev / "ep01.csv", 1)
    p = call(ev, onset(tmp_path, hm(1), 100), eps=(1,))
    assert p.returncode != 0 and "Tier1AccessError" in p.stderr, (p.returncode, p.stderr[-400:])
