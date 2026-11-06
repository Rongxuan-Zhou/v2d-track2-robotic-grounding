"""soma_fk 缓存的测试。在装了 py-soma-x 的机器上跑（mf4090 宿主机）。

守：FK 唯一入口的两条约定——缓存结果与直接计算逐位相同；flex 开关不同不共用缓存。
契约表尚未落到 docs/contracts/（仓库处于 P0），暂无契约 ID，落表后补上。
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pyarrow.parquet as pq
import soma_fk
from paths import TIER1


def main():
    t = pq.read_table(sorted((TIER1 / "data" / "chunk-000").glob("*.parquet"))[0])
    for flex in (True, False):
        p = soma_fk.CACHE_DIR / f"{soma_fk._cache_key(t, flex)}.npy"
        p.unlink(missing_ok=True)                       # 从冷缓存开始
        ref = soma_fk._fk_compute(t, flex=flex)
        t0 = time.time(); a = soma_fk.fk(t, flex=flex); cold = time.time() - t0
        assert p.exists(), "第一次调用没有写缓存"
        t0 = time.time(); b = soma_fk.fk(t, flex=flex); warm = time.time() - t0
        assert np.array_equal(a, ref) and np.array_equal(b, ref), "缓存结果与直接计算不一致"
        print(f"  flex={flex}: 冷 {cold:.2f}s  热 {warm:.3f}s  逐位一致 ✓")
    assert soma_fk._cache_key(t, True) != soma_fk._cache_key(t, False), "flex 开关共用了缓存键"
    print("  flex 开关缓存键不同 ✓\n全部通过")


if __name__ == "__main__":
    main()
