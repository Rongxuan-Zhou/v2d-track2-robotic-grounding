"""tier_guard 的测试（子进程里跑，因为 audit hook 装上就卸不掉）。

守：档位隔离——Tier 2/3 提交管线读不到 Tier 1 的任何文件，且 Tier 2 照常可读。
契约表尚未落到 docs/contracts/（仓库处于 P0），暂无契约 ID，落表后补上。
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# 探针在子进程里跑（hook 卸不掉）。
PROBE = r'''
import sys, glob, json; sys.path.insert(0, %r)
import numpy as np, pyarrow.parquet as pq
from paths import TIER1, TIER2
from tier_guard import forbid_tier1, Tier1AccessError
forbid_tier1()
t2 = sorted((TIER2 / "data" / "chunk-000").glob("*.parquet"))[0]
pq.read_table(t2); json.load(open(TIER2 / "meta" / "info.json"))
print("TIER2_OK")
t1 = str(TIER1 / "data" / "chunk-000" / t2.name)
for name, fn in [("open", lambda: open(TIER1 / "meta" / "info.json")),
                 ("pq.read_table", lambda: pq.read_table(t1)),
                 ("pq.ParquetFile", lambda: pq.ParquetFile(t1)),
                 ("glob", lambda: glob.glob(str(TIER1 / "data" / "chunk-000" / "*"))),
                 ("np.load", lambda: np.load(str(TIER1 / "x.npz")))]:
    try:
        fn(); print("LEAK", name)
    except Tier1AccessError:
        print("BLOCKED", name)
''' % str(HERE)


def test_guard():
    out = subprocess.run([sys.executable, "-c", PROBE], capture_output=True, text=True).stdout
    print(out)
    assert "TIER2_OK" in out, "守卫误伤了 Tier 2 读取"
    assert "LEAK" not in out, "有读取路径没拦住"
    assert out.count("BLOCKED") == 5


# 没装 pyarrow 的解释器（ml40 宿主的 python3）：forbid_tier1() 不报错，open 仍被拦。只用标准库，Mac 上也能跑。
PROBE_NO_PYARROW = r'''
import sys; sys.path.insert(0, %r)
sys.modules["pyarrow"] = None; sys.modules["pyarrow.parquet"] = None   # 让 import pyarrow 抛 ImportError
from paths import TIER1
from tier_guard import forbid_tier1, Tier1AccessError
forbid_tier1()
print("GUARD_ON")
try:
    open(TIER1 / "meta" / "info.json"); print("LEAK open")
except Tier1AccessError:
    print("BLOCKED open")
''' % str(HERE)


def test_guard_without_pyarrow():
    r = subprocess.run([sys.executable, "-c", PROBE_NO_PYARROW], capture_output=True, text=True)
    print(r.stdout, r.stderr)
    assert "GUARD_ON" in r.stdout, "没装 pyarrow 时 forbid_tier1() 报错了"
    assert "BLOCKED open" in r.stdout and "LEAK" not in r.stdout, "没装 pyarrow 时 open 没拦住"


if __name__ == "__main__":
    test_guard_without_pyarrow()
    try:
        import numpy, pyarrow  # noqa: F401
    except ImportError:
        print("跳过 test_guard：这个解释器没有 numpy / pyarrow")
    else:
        test_guard()
    print("通过")
