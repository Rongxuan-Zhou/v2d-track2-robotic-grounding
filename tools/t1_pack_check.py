"""核对 Tier 1 打包用的结果目录与选择清单逐集一致（Mac 上、仓库根执行）。

用法：python3 tools/t1_pack_check.py <选择清单.tsv> <结果目录>
选择清单每行“档位 episode run 步数 机器”。每集 episode_XXXXXX.json 的 checkpoint 要指向清单里的 run 与步数、world_count 为 1、
non_finite_worlds 为空，parquet 存在。全过打印 T1_PACK_CHECK_OK <集数>，否则 T1_PACK_CHECK_FAIL 并退出码 1。
"""

import json
import sys
from pathlib import Path


def main():
    sel, res = sys.argv[1], Path(sys.argv[2])
    bad, n = [], 0
    for line in open(sel):
        if not line.strip():
            continue
        _, ep, run, step, mach = line.split()[:5]
        n += 1
        e6 = f"{int(ep):06d}"
        js, pq = res / f"episode_{e6}.json", res / f"episode_{e6}.parquet"
        if not js.is_file() or not pq.is_file():
            bad.append(f"ep{ep}：缺 json 或 parquet")
            continue
        d = json.loads(js.read_text())
        want = f"/runs/flash_chord/{run}/policy_{step}.safetensors"
        if not d["checkpoint"].endswith(want):
            bad.append(f"ep{ep}：checkpoint {d['checkpoint']}，清单 {run} {step}")
        if d["world_count"] != 1 or d["non_finite_worlds"]:
            bad.append(f"ep{ep}：world_count {d['world_count']}，non_finite_worlds {d['non_finite_worlds']}")
        print(f"ep{ep:<3} {run:<42} {mach:<7} {d['checkpoint_sha256'][:12]}")
    extra = sorted({p.name for p in res.glob("episode_*")} - {f"episode_{int(l.split()[1]):06d}.{x}" for l in open(sel) if l.strip() for x in ("json", "parquet")})
    if extra:
        bad.append(f"结果目录里有清单以外的文件：{extra}")
    for b in bad:
        print("  不符：" + b)
    print(f"T1_PACK_CHECK_OK {n}" if not bad else f"T1_PACK_CHECK_FAIL {len(bad)}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
