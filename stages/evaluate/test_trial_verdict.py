"""trial_verdict.py 的测试（守计划 agent-doc/plan-step3-recipe-trial-2026-10-04.md 第 2 版第 4 节的判据；运行规范未成文，暂无契约 ID）。
只用标准库，Mac 与训练机都能跑。每个用例在临时目录里造一个小仓库：policy 文件、基线 seed 42 的重评（export_8c778f9）与评测链的结果、
评测日志里 in-flash-chord 的挂载行、DONE；判定脚本在子进程里跑（Tier 2 会装上卸不掉的 audit hook）。
  1. 中途检查：平均差 > 0 → CONTINUE；平均 ≤ 0 且最大 < +0.10 → STOP；平均 ≤ 0 但有一条 ≥ +0.10 → CONTINUE；
     评测日志挂的不是 8c778f9、或 checkpoint sha256 对不上 → 退出码 2
  2. 全判定：全过 → PASS；增益集中在两条（S2 不过）、一条 ADD-AUC 退 0.06（S3 不过）、对照退 0.2（S6 不过）、
     变体训练日志里有 NONFINITE_GUARD_STOP（S5 不过）→ FAIL；基线 seed 42 的重评缺失 → INCOMPLETE（3）；
     BC 段一条都没开跑 → STOPPED_AT_MIDCHECK（4）；基线 seed 43 缺评测 → B 只用 seed 42 并注明
  3. Tier 2 清单先调 forbid_tier1()
用法：python3 stages/evaluate/test_trial_verdict.py
"""
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEPS, WORLDS = 1000, 4096
EPS, CAN = [1, 2, 3, 4, 5], 9
MOUNT_OK = "in-flash-chord: /workspace ← /x/fc_src/fc_8c778f9（flash_chord 8c778f905c 副本）\n"
MOUNT_OLD = "in-flash-chord: /workspace ← /x/fc_src/v040（flash_chord v0.4.0 33129dd）\n"


def names(ep):
    return {k: f"t1-ep{ep}-{k}" for k in ("base42", "var42", "base43", "var43")}


def spec(tier=1):
    return {"tier": tier, "machine": "mf4090", "steps": STEPS, "worlds": WORLDS,
            "episodes": [dict(ep=ep, **names(ep)) for ep in EPS],
            "canary": {"ep": CAN, "base42": f"t1-ep{CAN}-base42", "var42": f"t1-ep{CAN}-var42"}}


def write_run(root, run, ep, kind, mp, auc=0.5, sp=0.9, mppe=4.0, mount=MOUNT_OK, sha_ok=True, done=True, log_extra=""):
    o = root / "runs" / "flash_chord" / run
    o.mkdir(parents=True, exist_ok=True)
    pol = o / f"policy_{STEPS}.safetensors"
    pol.write_bytes(run.encode() * 7)
    import hashlib
    sha = hashlib.sha256(pol.read_bytes()).hexdigest() if sha_ok else "0" * 64
    d = {"checkpoint_sha256": sha, "environment_steps": STEPS, "world_count": WORLDS, "non_finite_worlds": [],
         "metrics": {"maniptrans_sr": mp, "spider_sr_uncentered": sp, "add_auc": auc, "mppe_cm": mppe},
         "reference_tracking": {"success_rate": 0.9}}
    logs = root / "runs" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    if kind == "export":
        j = o / "export_8c778f9" / f"s{STEPS}_w{WORLDS}" / f"episode_{ep:06d}.json"
        log = logs / f"{run}.export_8c778f9_s{STEPS}_w{WORLDS}.log"
    else:
        j = o / f"eval_{STEPS}_w{WORLDS}.json"
        log = logs / f"{run}.eval_{STEPS}_w{WORLDS}.log"
    j.parent.mkdir(parents=True, exist_ok=True)
    j.write_text(json.dumps(d))
    log.write_text("=== 评测\n" + mount + "report: ok\n")
    if done:
        (o / "DONE").write_text("ok\n")
    if log_extra:
        (logs / f"{run}.s1.log").write_text(log_extra)


def build(root, base42, var42, base43=None, var43=None, can=(0.9, 0.9), auc=None, skip=(), opts=None):
    """base42 等是每条 episode 的 MP-SR 列表；auc 给 {(run 类型, ep): 值} 覆盖默认 0.5；skip 里的 run 不写；
    opts 给 {run 类型: write_run 的额外参数}。"""
    auc = auc or {}
    for i, ep in enumerate(EPS):
        n = names(ep)
        for k, vals in (("base42", base42), ("var42", var42), ("base43", base43), ("var43", var43)):
            if vals is None or n[k] in skip:
                continue
            write_run(root, n[k], ep, "export" if k == "base42" else "eval", vals[i], auc=auc.get((k, ep), 0.5),
                      **(opts or {}).get(k, {}))
    write_run(root, f"t1-ep{CAN}-base42", CAN, "export", can[0])
    write_run(root, f"t1-ep{CAN}-var42", CAN, "eval", can[1])


def run(root, *args, sp=None):
    s = root / "spec.json"
    s.write_text(json.dumps(sp or spec()))
    r = subprocess.run([sys.executable, str(HERE / "trial_verdict.py"), str(s), "--root", str(root), *args],
                       capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def case(fn):
    with tempfile.TemporaryDirectory() as t:
        return fn(Path(t))


B = [0.30, 0.20, 0.10, 0.40, 0.25]


def test_midcheck_continue():
    def f(root):
        build(root, B, [b + 0.05 for b in B])
        rc, out = run(root, "--midcheck")
        assert rc == 0 and "MIDCHECK CONTINUE" in out, out
    case(f)


def test_midcheck_stop():
    def f(root):
        build(root, B, [b - 0.02 for b in B[:4]] + [B[4] + 0.05])
        rc, out = run(root, "--midcheck")
        assert rc == 1 and "MIDCHECK STOP" in out, out
    case(f)


def test_midcheck_one_big_gain_continues():
    def f(root):
        build(root, B, [B[0] + 0.12] + [b - 0.05 for b in B[1:]])
        rc, out = run(root, "--midcheck")
        assert rc == 0 and "MIDCHECK CONTINUE" in out and "平均差 -0.0160" in out, out
    case(f)


def test_midcheck_wrong_evaluator():
    def f(root):
        build(root, B, [b + 0.3 for b in B], opts={"var42": {"mount": MOUNT_OLD}})
        rc, out = run(root, "--midcheck")
        assert rc == 2 and "不是 8c778f9 副本" in out, out
    case(f)


def test_midcheck_sha_mismatch():
    def f(root):
        build(root, B, [b + 0.3 for b in B], opts={"base42": {"sha_ok": False}})
        rc, out = run(root, "--midcheck")
        assert rc == 2 and "checkpoint_sha256" in out, out
    case(f)


def test_full_pass():
    def f(root):
        build(root, B, [b + 0.25 for b in B], [b + 0.02 for b in B], [b + 0.20 for b in B])
        rc, out = run(root)
        assert rc == 0 and out.rstrip().endswith("TRIAL_VERDICT PASS"), out
        assert "只报 敏感性" in out and "只报 chord_sr" in out, out
    case(f)


def test_full_fail_s2_concentrated_gain():
    def f(root):
        g = [0.50, 0.40, 0.04, 0.04, 0.04]   # 平均 +0.204，但 ≥ +0.05 的只有 2 条
        build(root, B, [b + x for b, x in zip(B, g)], B, [b + x for b, x in zip(B, g)])
        rc, out = run(root)
        assert rc == 1 and "S1 过" in out and "S2 不过" in out and "FAIL（S2 不过）" in out, out
    case(f)


def test_full_fail_s3_auc_drop():
    def f(root):
        build(root, B, [b + 0.3 for b in B], B, [b + 0.3 for b in B], auc={("var42", 3): 0.43, ("var43", 3): 0.43})
        rc, out = run(root)
        assert rc == 1 and "S3 不过" in out and "ep3 ADD-AUC -0.070" in out, out
    case(f)


def test_full_fail_s6_canary():
    def f(root):
        build(root, B, [b + 0.3 for b in B], B, [b + 0.3 for b in B], can=(0.9, 0.7))
        rc, out = run(root)
        assert rc == 1 and "S6 不过" in out, out
    case(f)


def test_full_fail_s5_guard_stop():
    def f(root):
        build(root, B, [b + 0.3 for b in B], B, [b + 0.3 for b in B],
              opts={"var43": {"log_extra": "x\nNONFINITE_GUARD_STOP environment_steps=1 reason=total\n"}})
        rc, out = run(root)
        assert rc == 1 and "S5 不过" in out and "NONFINITE_GUARD_STOP" in out, out
    case(f)


def test_incomplete_without_baseline_reeval():
    def f(root):
        build(root, B, [b + 0.3 for b in B], B, [b + 0.3 for b in B], skip={"t1-ep2-base42"})
        rc, out = run(root)
        assert rc == 3 and "INCOMPLETE" in out, out
    case(f)


def test_stopped_at_midcheck():
    def f(root):
        build(root, B, [b - 0.02 for b in B])
        rc, out = run(root)
        assert rc == 4 and "STOPPED_AT_MIDCHECK" in out, out
    case(f)


def test_missing_base43_uses_seed42():
    def f(root):
        build(root, B, [b + 0.25 for b in B], [b + 0.02 for b in B], [b + 0.20 for b in B], skip={"t1-ep4-base43"})
        (root / "runs" / "flash_chord" / "t1-ep4-base43").mkdir(parents=True)   # 开跑了但没评完（例如训练失败）
        rc, out = run(root)
        assert "ep4 基线 seed 43 缺失，B 只用 seed 42" in out and "TRIAL_VERDICT PASS" in out, out
    case(f)


def test_tier2_calls_guard():
    def f(root):
        (root / "libs" / "v2d_infra").mkdir(parents=True)
        shutil.copy(HERE.parents[1] / "libs" / "v2d_infra" / "tier_guard.py", root / "libs" / "v2d_infra")
        build(root, B, [b + 0.05 for b in B])
        rc, out = run(root, "--midcheck", sp=spec(tier=2))
        assert rc == 0 and "净室：Tier 2 清单，已调 forbid_tier1()" in out, out
    case(f)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("ALL_OK")
