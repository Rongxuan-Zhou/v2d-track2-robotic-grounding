"""r3o_gate.py 的测试（计划 agent-doc/plan-r3o-2026-10-15.md 第 3 节的段间门；只用标准库）。
临时目录里放假的评测 JSON 与一份两段的规格，核：
  1. 对照过、跟丢集都没用 → CONTINUE；对照不过、都没用 → STOP（退出码 1）；
  2. 对照不过，但一集的失败_物体比基线低 0.2 以上、或 MP-SR ≥ 0.1、或修好 → CONTINUE；
  3. 基线 SP-SR 已 ≥ 0.3 的集，SP-SR ≥ 0.3 不算有起色；基线 < 0.3 的集算；
  4. 评测文件缺、世界数不是 4096、有非有限世界 → ERROR（退出码 2）；
  5. Tier 2 的段装了 Tier 1 禁读：root 在 tier_1_multiview_caption 下就读不了（档位隔离），退出码非零。
用法：python3 stages/policy/test_r3o_gate.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = {
    "a1": {"tier": 1, "control": {"run": "t1-ep20-c", "floor": 0.45},
           "lost": {"t1-ep25-x": {"baseline_run": "b25", "baseline_obj_fail": 1.0, "baseline_sp": 0.0},
                    "t1-ep2-x": {"baseline_run": "b2", "baseline_obj_fail": 0.9, "baseline_sp": 0.4}}},
    "a2": {"tier": 2, "control": {"run": "t2-ep12-c", "floor": 0.5},
           "lost": {"t2-ep25-x": {"baseline_run": "b25", "baseline_obj_fail": 1.0, "baseline_sp": 0.0}}},
}


def put(root: Path, run: str, mp: float, sp: float = 0.0, obj: float = 1.0, worlds: int = 4096, nonfinite=()):
    d = root / "runs/flash_chord" / run
    d.mkdir(parents=True, exist_ok=True)
    (d / "eval_349999104_w4096.json").write_text(json.dumps({
        "world_count": worlds, "non_finite_worlds": list(nonfinite),
        "metrics": {"maniptrans_sr": mp, "spider_sr_uncentered": sp},
        "reference_tracking": {"failure_cause_fraction": {"object": obj, "palm": 1.0, "pelvis": 1.0}}}))


def gate(root: Path, part: str):
    (root / "spec.json").write_text(json.dumps(SPEC))
    return subprocess.run([sys.executable, str(HERE / "r3o_gate.py"), part, "--root", str(root), "--spec", str(root / "spec.json")],
                          capture_output=True, text=True)


def case(ctl_mp, lost25, lost2, want_rc, want_line):
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        put(root, "t1-ep20-c", ctl_mp)
        put(root, "t1-ep25-x", *lost25)
        put(root, "t1-ep2-x", *lost2)
        p = gate(root, "a1")
        assert p.returncode == want_rc and p.stdout.rstrip().endswith(want_line), (ctl_mp, lost25, lost2, p.stdout, p.stderr)
        return p.stdout


def test_continue_stop():
    case(0.80, (0.0, 0.0, 1.0), (0.0, 0.1, 0.95), 0, "R3O_GATE_CONTINUE")
    out = case(0.30, (0.0, 0.0, 1.0), (0.0, 0.1, 0.95), 1, "R3O_GATE_STOP")
    assert "→ 没用" in out


def test_gain_keeps_going():
    out = case(0.30, (0.0, 0.0, 0.79), (0.0, 0.0, 0.95), 0, "R3O_GATE_CONTINUE")   # 失败_物体 1.0 → 0.79
    assert "t1-ep25-x：" in out and "→ 有起色" in out
    case(0.30, (0.12, 0.0, 1.0), (0.0, 0.0, 0.95), 0, "R3O_GATE_CONTINUE")        # MP-SR ≥ 0.1
    out = case(0.30, (0.35, 0.4, 0.5), (0.0, 0.0, 0.95), 0, "R3O_GATE_CONTINUE")  # 修好
    assert "→ 修好" in out


def test_sp_needs_low_baseline():
    case(0.30, (0.0, 0.0, 1.0), (0.0, 0.5, 0.95), 1, "R3O_GATE_STOP")      # ep2 的基线 SP 0.4：SP 0.5 不算
    case(0.30, (0.0, 0.35, 1.0), (0.0, 0.0, 0.95), 0, "R3O_GATE_CONTINUE")  # ep25 的基线 SP 0：SP 0.35 算


def test_errors():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        put(root, "t1-ep20-c", 0.8)
        put(root, "t1-ep25-x", 0.0)
        p = gate(root, "a1")   # t1-ep2-x 的评测缺
        assert p.returncode == 2 and p.stdout.rstrip().endswith("R3O_GATE_ERROR"), p.stdout
        put(root, "t1-ep2-x", 0.0, worlds=2048)
        p = gate(root, "a1")
        assert p.returncode == 2 and "世界数 2048" in p.stdout, p.stdout
        put(root, "t1-ep2-x", 0.0, nonfinite=[3])
        p = gate(root, "a1")
        assert p.returncode == 2 and "非有限世界 1 个" in p.stdout, p.stdout
        p = gate(root, "nope")
        assert p.returncode == 2 and "规格里没有段" in p.stdout, p.stdout


def test_tier2_guard():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t) / "tier_1_multiview_caption"
        root.mkdir()
        put(root, "t2-ep12-c", 0.9)
        put(root, "t2-ep25-x", 0.0)
        p = gate(root, "a2")
        assert p.returncode != 0 and "档位隔离" in (p.stdout + p.stderr), p.stdout + p.stderr
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        put(root, "t2-ep12-c", 0.9)
        put(root, "t2-ep25-x", 0.0)
        p = gate(root, "a2")
        assert p.returncode == 0 and p.stdout.rstrip().endswith("R3O_GATE_CONTINUE"), p.stdout + p.stderr


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("ALL_OK")
