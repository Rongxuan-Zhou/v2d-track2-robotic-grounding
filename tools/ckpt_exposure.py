"""归档清单里的 checkpoint 公开之后会暴露什么（只读；训练机上、仓库根执行，经 scripts/dev/pipe-py，只用标准库）。

用户 10/11 回复 HF 归档“不需要private”。改公开之前查两件事：
  1. 最终 state（kind final）里是不是含着同步数的 policy：逐个张量比原始字节的 sha256，policy 的张量都能在 state 里找到，
     就说明拿到这份 state 就拿到了这条 run 的最终策略（同步数的 policy 本身留在训练机上，不在仓库里）。
  2. safetensors 文件头的 __metadata__ 里有没有凭据或个人信息：token、api key、secret、password、hf_ 开头的 token、
     邮箱、/home/<用户> 路径、W&B 的 entity 与 project、网址。像凭据的几项只报个数，不打印内容（以免写进记录）；
     W&B entity、project 打印取值（它们不是凭据）。
只读文件头和张量字节，不改任何文件。

用法：python3 tools/ckpt_exposure.py <清单.json> [<清单.json> ...]
"""
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

RUNS = Path("runs/flash_chord")
SECRET = {"token": r"token", "api key": r"api[_-]?key", "secret": r"secret", "password": r"passw", "hf_ token": r"hf_[A-Za-z0-9]{20}"}
INFO = {"邮箱": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "/home 路径": r"/home/[A-Za-z0-9_.-]+", "网址": r"https?://[^\s\"\\,]+"}
WANDB = re.compile(r'\\?"wandb_(entity|project)\\?"\s*:\s*\\?"([^"\\]*)')


def header(p: Path):
    with p.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        h = json.loads(f.read(n))
    return h, 8 + n


def tensor_hashes(p: Path) -> set:
    h, base = header(p)
    h.pop("__metadata__", None)
    out = set()
    with p.open("rb") as f:
        for v in h.values():
            a, b = v["data_offsets"]
            f.seek(base + a)
            out.add(hashlib.sha256(f.read(b - a)).hexdigest())
    return out


def main() -> int:
    if len(sys.argv) < 2:
        sys.exit("用法：python3 tools/ckpt_exposure.py <清单.json> [...]")
    for m in sys.argv[1:]:
        rows = json.loads(Path(m).read_text())
        print(f"## {m}：{len(rows)} 个文件，{len({r['run'] for r in rows})} 条 run")
        secret = {k: 0 for k in SECRET}
        info = {k: set() for k in INFO}
        wandb, missing, fin_ok, fin_n = set(), 0, 0, 0
        for r in rows:
            p = RUNS / r["run"] / r["file"]
            if not p.exists():
                missing += 1
                continue
            blob = json.dumps(header(p)[0].get("__metadata__") or {})
            for k, pat in SECRET.items():
                secret[k] += len(re.findall(pat, blob, re.I))
            for k, pat in INFO.items():
                info[k] |= set(re.findall(pat, blob))
            wandb |= {f"{a}={b}" for a, b in WANDB.findall(blob)}
            if r["kind"] == "final":
                fin_n += 1
                pol = p.with_name(f"policy_{r['step']}.safetensors")
                ph = tensor_hashes(pol) if pol.exists() else set()
                inside = len(ph & tensor_hashes(p)) if ph else 0
                ok = bool(ph) and inside == len(ph)
                fin_ok += ok
                print(f"  final {r['run']}/{r['file']}：同步数 policy {len(ph)} 个张量，{inside} 个在 state 里")
        print(f"不在本地的文件 {missing} 个")
        print(f"最终 state {fin_n} 份，其中 {fin_ok} 份含有同步数 policy 的全部张量")
        print("文件头 metadata 里像凭据的匹配次数（只报个数）：" + "，".join(f"{k} {v}" for k, v in secret.items()))
        print("文件头 metadata 里的个人信息：" + "；".join(f"{k} {len(v)} 种" for k, v in info.items())
              + (f"；/home 路径 {sorted(info['/home 路径'])}" if info["/home 路径"] else ""))
        print(f"W&B：{sorted(wandb) if wandb else '无'}")
    print("CKPT_EXPOSURE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
