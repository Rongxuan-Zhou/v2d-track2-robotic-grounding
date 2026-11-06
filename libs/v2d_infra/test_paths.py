"""paths.py 里 V2D_T3_SET 的测试（守计划 agent-doc/plan-t3-pilot-v2-2026-10-10.md 第 3 节：校正版的目录不覆盖第一轮；运行规范未成文，
暂无契约 ID）。只用标准库，每种取值在子进程里导入一次（paths 在导入时读环境变量）。
  1. 不设或设成空：TIER3 是 runs/tier3/root，tier_dirname(1..3) 是 tier1、tier2、tier3
  2. V2D_T3_SET=v2：TIER3 是 runs/tier3/root_v2，tier_dirname(3) 是 tier3_v2，Tier 1、2 不变
  3. 不合规的值（大写、下划线、斜杠、..、空格）：导入就报错，报错里有 V2D_T3_SET
  4. tier_dirname(4) 报错
用法：python3 libs/v2d_infra/test_paths.py
"""
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = ("import sys; sys.path.insert(0, sys.argv[1]); import paths as p; "
        "print(p.TIER3.relative_to(p.REPO).as_posix(), *(p.tier_dirname(n) for n in (1, 2, 3)))")


def probe(value, code=CODE):
    env = {k: v for k, v in os.environ.items() if k != "V2D_T3_SET"}
    if value is not None:
        env["V2D_T3_SET"] = value
    return subprocess.run([sys.executable, "-c", code, str(HERE)], env=env, capture_output=True, text=True)


def test_unset_and_empty():
    for value in (None, ""):
        r = probe(value)
        assert r.returncode == 0 and r.stdout.split() == ["runs/tier3/root", "tier1", "tier2", "tier3"], (value, r.stdout, r.stderr)


def test_set():
    r = probe("v2")
    assert r.returncode == 0 and r.stdout.split() == ["runs/tier3/root_v2", "tier1", "tier2", "tier3_v2"], (r.stdout, r.stderr)


def test_bad_values_refuse():
    for value in ("V2", "v_2", "a/b", "..", "v2 "):
        r = probe(value)
        assert r.returncode != 0 and "V2D_T3_SET" in r.stderr, (value, r.stdout, r.stderr)


def test_no_tier4():
    r = probe(None, "import sys; sys.path.insert(0, sys.argv[1]); import paths as p; p.tier_dirname(4)")
    assert r.returncode != 0 and "没有第 4 档" in r.stderr, (r.stdout, r.stderr)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("ALL_OK")
