"""fall_order_one.py 的测试：单条 run 的驱动只换入口，门与判法沿用 fall_order.py（计划 agent-doc/plan-first-place-2026-10-14.md 第 2 节选项丙）。

1. 变体 run（名字带 -r3a-、朝向阈值 0.6）与官方 run（1.5）各自过门、出描述，退出码 0；
2. 名字带 -r3a- 却是 1.5 的阈值：门不过、不分类，退出码 1；--ori 1.5 改核 1.5 时过；
3. 两条一起给、其中一条门不过：好的那条照常描述，退出码 1；
4. MP-SR 与原评测差超过 0.01：门不过；
5. 缺 dump 文件：这条按门不过记、不抛异常，同一次给的其余 run 照常处理；
6. --ori 0.6 用在名字不带 -r3a- 的 run 上按 0.6 核，且一次给多条时每条都按它核。
7. 不同档的 run 一起给：退出码 2；
8. t2 的 run 装 Tier 1 禁读：root 在 tier_1_multiview_caption 目录下时读文件就被拦（档位隔离），t1 的 run 不装、照常读。
9. 名字带 -r3o- 的 run（r3o，变体 A 加物体朝向项）同样按 0.6 核：0.6 过门，1.5 门不过（fall_order.is_variant，r3o 计划代码审查第 5 条）。
用 test_fall_order.py 的 make_run 造数据。要 numpy 与 pytest。
"""

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from test_fall_order import make_run  # noqa: E402


def call(root, *runs, ori=None):
    cmd = [sys.executable, str(HERE / "fall_order_one.py"), *runs, "--steps", "1", "--worlds", "9", "--root", str(root)]
    if ori:
        cmd += ["--ori", ori]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_variant_and_base_pass(tmp_path):
    make_run(tmp_path, "t1-ep2-w2048-cap10m-r3a-ml40", 0.6, [(150, 70)] * 4 + [(None, 90)])
    make_run(tmp_path, "t1-ep25-w2048-cap10m-ml40", 1.5, [(60, 120)])
    p = call(tmp_path, "t1-ep2-w2048-cap10m-r3a-ml40", "t1-ep25-w2048-cap10m-ml40")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "== t1-ep2-w2048-cap10m-r3a-ml40（物体朝向阈值按 0.6 核）" in p.stdout
    assert "== t1-ep25-w2048-cap10m-ml40（物体朝向阈值按 1.5 核）" in p.stdout
    assert "[t1-ep2-w2048-cap10m-r3a-ml40] 失败世界 50" in p.stdout and "骨盆先" in p.stdout
    assert p.stdout.rstrip().endswith("FALL_ORDER_ONE_DONE")


def test_threshold_gate_and_override(tmp_path):
    make_run(tmp_path, "t2-ep4-w2048-cap10m-r3a-ml40", 1.5, [(60, 120)])
    p = call(tmp_path, "t2-ep4-w2048-cap10m-r3a-ml40")
    assert p.returncode == 1 and "门不过，不分类" in p.stdout and "不是 0.6" in p.stdout, p.stdout
    p = call(tmp_path, "t2-ep4-w2048-cap10m-r3a-ml40", ori="1.5")
    assert p.returncode == 0 and "FALL_ORDER_ONE_DONE" in p.stdout, p.stdout + p.stderr


def test_one_bad_one_good(tmp_path):
    make_run(tmp_path, "t1-ep9-w2048-cap10m-ml40", 1.5, [(60, 120)])
    make_run(tmp_path, "t1-ep11-w2048-cap4m-mf4090", 1.5, [(60, 120)], sha="other")
    p = call(tmp_path, "t1-ep9-w2048-cap10m-ml40", "t1-ep11-w2048-cap4m-mf4090")
    assert p.returncode == 1 and "[t1-ep9-w2048-cap10m-ml40] 失败世界" in p.stdout, p.stdout
    assert "sha256" in p.stdout and "FALL_ORDER_ONE_GATE_FAILED 1 条" in p.stdout


def test_mp_repro_gate(tmp_path):
    make_run(tmp_path, "t1-ep1-w2048-cap4m-p185-mf4090", 1.5, [(60, 120)], mp_shift=0.05)
    p = call(tmp_path, "t1-ep1-w2048-cap4m-p185-mf4090")
    assert p.returncode == 1 and "超过 0.01" in p.stdout, p.stdout


def test_missing_dump_does_not_stop_others(tmp_path):
    make_run(tmp_path, "t1-ep9-w2048-cap10m-ml40", 1.5, [(60, 120)])
    make_run(tmp_path, "t1-ep4-w2048-cap10m-ml40", 1.5, [(60, 120)])
    (tmp_path / "runs/flash_chord/t1-ep4-w2048-cap10m-ml40/eval_dump/eval_1_w9.npz").unlink()
    p = call(tmp_path, "t1-ep4-w2048-cap10m-ml40", "t1-ep9-w2048-cap10m-ml40")
    assert p.returncode == 1 and "Traceback" not in p.stderr, p.stderr
    assert "== t1-ep4-w2048-cap10m-ml40：门不过，不分类：读不了评测文件或缺键" in p.stdout, p.stdout
    assert "[t1-ep9-w2048-cap10m-ml40] 失败世界" in p.stdout and "FALL_ORDER_ONE_GATE_FAILED 1 条" in p.stdout


def test_ori_override_applies_to_every_run(tmp_path):
    make_run(tmp_path, "t1-ep25-w2048-cap10m-ml40", 0.6, [(60, 120)])
    make_run(tmp_path, "t1-ep28-w2048-cap4m-p185-mf4090", 0.6, [(60, 120)])
    p = call(tmp_path, "t1-ep25-w2048-cap10m-ml40", "t1-ep28-w2048-cap4m-p185-mf4090")
    assert p.returncode == 1 and p.stdout.count("门不过，不分类") == 2, p.stdout
    p = call(tmp_path, "t1-ep25-w2048-cap10m-ml40", "t1-ep28-w2048-cap4m-p185-mf4090", ori="0.6")
    assert p.returncode == 0 and p.stdout.count("（物体朝向阈值按 0.6 核）") == 2, p.stdout + p.stderr


def test_mixed_tiers_refused(tmp_path):
    make_run(tmp_path, "t1-ep9-w2048-cap10m-ml40", 1.5, [(60, 120)])
    make_run(tmp_path, "t2-ep9-w2048-cap4m-mf4090", 1.5, [(60, 120)])
    p = call(tmp_path, "t1-ep9-w2048-cap10m-ml40", "t2-ep9-w2048-cap4m-mf4090")
    assert p.returncode == 2 and "一次只收同一档" in p.stderr, p.stdout + p.stderr


def test_tier_guard(tmp_path):
    root = tmp_path / "tier_1_multiview_caption"
    make_run(root, "t2-ep9-w2048-cap4m-mf4090", 1.5, [(60, 120)])
    make_run(root, "t1-ep9-w2048-cap10m-ml40", 1.5, [(60, 120)])
    p = call(root, "t2-ep9-w2048-cap4m-mf4090")
    assert p.returncode != 0 and "档位隔离" in p.stderr, p.stdout + p.stderr
    p = call(root, "t1-ep9-w2048-cap10m-ml40")
    assert p.returncode == 0 and "FALL_ORDER_ONE_DONE" in p.stdout, p.stdout + p.stderr


def test_r3o_name_is_variant(tmp_path):
    make_run(tmp_path, "t1-ep2-w2048-cap4m-r3o-mf4090", 0.6, [(150, 70)] * 4 + [(None, 90)])
    make_run(tmp_path, "t1-ep9-w2048-cap4m-r3o-mf4090", 1.5, [(60, 120)])
    p = call(tmp_path, "t1-ep2-w2048-cap4m-r3o-mf4090")
    assert p.returncode == 0 and "（物体朝向阈值按 0.6 核）" in p.stdout, p.stdout + p.stderr
    p = call(tmp_path, "t1-ep9-w2048-cap4m-r3o-mf4090")
    assert p.returncode == 1 and "不是 0.6" in p.stdout, p.stdout

