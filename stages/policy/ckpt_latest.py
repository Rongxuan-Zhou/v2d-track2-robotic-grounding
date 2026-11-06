"""打印 <run 目录> 下最新一份完整可用的 FlashCHORD training state（state_<步数>.safetensors）的步数。

"完整"指：safetensors 文件头能解析、文件大小等于头里声明的数据末尾，且同步数的 policy 文件也完整。
训练进程在写 checkpoint 时被杀，最新一份可能只写了一半；这时退回上一份。没有可用的就不输出。
只用标准库，宿主机上直接跑。
"""
import json
import re
import struct
import sys
from pathlib import Path


def complete(path: Path) -> bool:
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                return False
            (n,) = struct.unpack("<Q", raw)
            if n <= 0 or 8 + n > size:
                return False
            header = json.loads(f.read(n))
        end = max((v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"), default=0)
        return size == 8 + n + end
    except (OSError, ValueError, KeyError, TypeError):
        return False


def main(run_dir: str) -> None:
    d = Path(run_dir)
    steps = sorted(
        (int(m.group(1)) for p in d.glob("state_*.safetensors") if (m := re.fullmatch(r"state_(\d+)\.safetensors", p.name))),
        reverse=True,
    )
    for s in steps:
        if complete(d / f"state_{s}.safetensors") and complete(d / f"policy_{s}.safetensors"):
            print(s)
            return


if __name__ == "__main__":
    main(sys.argv[1])
