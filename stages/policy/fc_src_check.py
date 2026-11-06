"""核对一份 flash_chord 源码副本（V2D_FC_SRC 挂成 /workspace 的那种）是不是它声称的版本（训练机上、仓库根执行，只用标准库）。

计划 agent-doc/plan-box6-and-eval-runtime-2026-10-01.md 第 3 节。对照的是仓库 third_party/video_to_data 里的 v0.4.0 flash_chord：
  1. 对照本身可信：third_party/video_to_data 的 HEAD 是 33129dd，已跟踪的文件没有改动；
  2. 副本里有 SOURCE_V2D.txt（scripts/dev/in-flash-chord 只认带它的副本）；
  3. 逐文件比内容：副本与 v0.4.0 不同（含只在一边有）的文件，正好是清单里列的那些；
  4. 清单里每个文件在副本里的 sha256 与清单相同。
忽略任意层级名为 outputs、logs 的目录与文件（生成副本时 rsync 就排除了它们）、__pycache__/、*.pyc（容器里 import 会往挂载目录写 .pyc）、
.cache/（FlashCHORD 运行时写的凸包缓存，10/1 ml40 实测：副本与对照各自生成，同名文件字节也不同）、根下的 SOURCE_V2D.txt，
以及根下评测器用过之后可能留下的 wandb/、.hydra/（只报个数）。符号链接比链接目标。
清单格式同 shasum：每行“sha256  相对 flash_chord 目录的路径”。现有两份：
  third_party/patches/flash_chord-8c778f9.sha256（上游 8c778f9，11 个文件，tools/make_fc_patch.sh 生成）
  third_party/patches/flash_chord-p185.sha256（v0.4.0 + #185 补丁副本，只有 scene/support.py）
用法：python3 stages/policy/fc_src_check.py <副本目录> <清单>；全过退出码 0、最后一行 FC_SRC_OK，否则退出码 1、最后一行 FC_SRC_FAIL n。
"""
import filecmp
import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "third_party/video_to_data"
BASE = UPSTREAM / "robotic_grounding/flash_chord"
SKIP_NAMES = {"outputs", "logs"}   # 任意层级、目录或文件，同 make_*_fc_src.sh 里 rsync 不带 / 的 --exclude
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".cache"}   # .cache：FlashCHORD 建场景时写的凸包缓存（assets/.cache/convex/*.npz），npz 带时间戳，重算一次字节就变
RUNTIME_TOP = {"wandb", ".hydra"}   # 挂成 /workspace 用过之后，评测器可能在根下留下的运行产物：不比，单独报个数
WANT_HEAD = os.environ.get("V2D_FC_SRC_BASE_HEAD", "33129dd")   # 只给测试换


def tree(top: Path) -> dict:
    """相对路径 → 绝对路径；按上面的规则跳过目录与文件；不跟随目录符号链接。"""
    out = {}
    for d, dirs, files in os.walk(top):
        rel_d = Path(d).relative_to(top)
        dirs[:] = [x for x in dirs if x not in SKIP_NAMES | SKIP_DIRS and not (rel_d == Path(".") and x in RUNTIME_TOP)]
        for x in [x for x in dirs if (Path(d) / x).is_symlink()]:   # 指向目录的符号链接：不进去，当一个条目比链接目标
            dirs.remove(x)
            files.append(x)
        for f in files:
            if f.endswith(".pyc") or f in SKIP_NAMES or (rel_d == Path(".") and f == "SOURCE_V2D.txt"):
                continue
            out[str(rel_d / f) if rel_d != Path(".") else f] = Path(d) / f
    return out


def same(a: Path, b: Path) -> bool:
    if a.is_symlink() or b.is_symlink():
        return a.is_symlink() and b.is_symlink() and os.readlink(a) == os.readlink(b)
    return filecmp.cmp(a, b, shallow=False)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    copy, man = Path(sys.argv[1]), Path(sys.argv[2])
    bad = 0

    def item(ok: bool, msg: str) -> None:
        nonlocal bad
        print(f"  {'ok  ' if ok else 'FAIL'} {msg}")
        bad += not ok

    print(f"== 副本 {copy.resolve()}，清单 {man}")
    git = lambda *a: subprocess.run(["git", "-C", str(UPSTREAM), *a], capture_output=True, text=True).stdout.strip()
    head = git("rev-parse", "--short=7", "HEAD")
    item(head == WANT_HEAD, f"对照 third_party/video_to_data 的 HEAD {head or '（读不出）'} = {WANT_HEAD}（v0.4.0）")
    dirty = git("status", "--porcelain", "--untracked-files=no")
    item(not dirty, "对照的已跟踪文件没有改动" + ("" if not dirty else f"：{dirty.splitlines()[:3]}"))
    src = copy / "SOURCE_V2D.txt"
    item(src.is_file(), f"SOURCE_V2D.txt：{src.read_text().splitlines()[0] if src.is_file() else '没有'}")
    want = {}
    for line in man.read_text().splitlines():
        if line.strip() and not line.startswith("#"):
            h, rel = line.split(None, 1)
            want[rel.strip()] = h
    left = [x for x in sorted(RUNTIME_TOP) if (copy / x).is_dir()]
    if left:
        print(f"  info 副本根下有运行产生的目录 {left}（不比；挂载用过之后才会有）")
    a, b = tree(BASE), tree(copy)
    differ = sorted(r for r in set(a) | set(b) if r not in a or r not in b or not same(a[r], b[r]))
    print(f"  info 对照 {len(a)} 个文件，副本 {len(b)} 个，不同 {len(differ)} 个")
    for r in differ:
        print(f"       {'只在副本' if r not in a else '只在对照' if r not in b else '内容不同'} {r}")
    item(set(differ) == set(want), f"不同的文件正好是清单里的 {len(want)} 个"
         + ("" if set(differ) == set(want) else f"（多 {sorted(set(differ) - set(want))}，少 {sorted(set(want) - set(differ))}）"))
    for rel, h in sorted(want.items()):
        got = sha256(copy / rel) if (copy / rel).is_file() else "（没有）"
        item(got == h, f"{rel} sha256 {got[:16]}…")
    print("FC_SRC_OK" if bad == 0 else f"FC_SRC_FAIL {bad}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
