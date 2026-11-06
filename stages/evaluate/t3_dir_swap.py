"""拼 Tier 3 的提交轨迹目录：把基础目录（上一次交的 30 集）复制到新目录，指定的集换成新导出的文件，再逐个核（Mac 上、仓库根执行，只用标准库）。
计划 agent-doc/plan-t3-gb3-switch-2026-10-17.md 第 4 节（同 kfix1017 的轨迹目录做法）。

核：基础目录恰好 30 集、每集一个 episode_XXXXXX.json 与 .parquet；新导出目录里有要换的每一集；拼好的目录里与基础目录不同的文件恰好是
要换的那几集的 json 与 parquet，其余逐字相同；换进去的文件与新导出目录里的逐字相同。打印换掉的文件的 sha256（前后各一）。
用法：python3 stages/evaluate/t3_dir_swap.py <基础目录> <新导出目录> <输出目录> <集,集,…>   输出目录已存在就拒绝。最后一行 T3_DIR_SWAP_OK。
"""

import hashlib
import shutil
import sys
from pathlib import Path


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    if len(sys.argv) != 5:
        sys.exit("用法：t3_dir_swap.py <基础目录> <新导出目录> <输出目录> <集,集,…>")
    base, new, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    eps = sorted({int(x) for x in sys.argv[4].split(",") if x.strip()})
    if out.exists():
        sys.exit(f"{out} 已存在，拒绝覆盖")
    names = sorted(p.name for p in base.iterdir())
    want = sorted(f"episode_{e:06d}.{k}" for e in range(30) for k in ("json", "parquet"))
    if names != want:
        sys.exit(f"基础目录 {base} 不是恰好 30 集的 json 与 parquet（{len(names)} 个文件）")
    swap = {f"episode_{e:06d}.{k}" for e in eps for k in ("json", "parquet")}
    for n in sorted(swap):
        if not (new / n).is_file():
            sys.exit(f"新导出目录里没有 {n}")
    shutil.copytree(base, out)
    for n in sorted(swap):
        old = sha(out / n)
        shutil.copy2(new / n, out / n)
        print(f"换 {n}：{old[:16]}… → {sha(out / n)[:16]}…")
    diff = sorted(n for n in want if sha(out / n) != sha(base / n))
    same_new = all(sha(out / n) == sha(new / n) for n in swap)
    print(f"与基础目录不同的文件：{diff}")
    print(f"恰好是 ep{'、'.join(map(str, eps))} 的 json 与 parquet：{diff == sorted(swap)}；换进去的与新导出逐字相同：{same_new}；其余 {len(want) - len(diff)} 个逐字相同")
    if diff != sorted(swap) or not same_new:
        sys.exit("核对不过")
    print("T3_DIR_SWAP_OK")


if __name__ == "__main__":
    main()
