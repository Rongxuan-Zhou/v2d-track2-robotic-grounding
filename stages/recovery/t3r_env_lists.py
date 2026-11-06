"""t3takeover 第 1b 级：读一个 venv 的发行版清单，并给每个发行版的 RECORD 算摘要（只读；只用标准库，系统 python3 就能跑）。
计划 agent-doc/plan-t3takeover-1b-2026-10-21.md 第 1 节第 1、3 条，判据 E0 与复核改动第 5 条。
- 每个 *.dist-info：名字（PEP 503 规范化）、版本、INSTALLER、有没有 direct_url.json；
- 摘要：RECORD 里带 sha256 的（路径，哈希）去掉 ../ 开头（bin 下的脚本与链接）、.pyc、__pycache__、INSTALLER、REQUESTED、direct_url.json、RECORD 本身，
  dist-info 目录名统一成 <dist-info>，排序后整体再算 sha256 取前 16 位；同名同版本而构建不同（文件内容不同）时摘要就不同。
用法：
  python3 t3r_env_lists.py <venv>                              每个发行版一行：名字 版本 INSTALLER direct_url 文件数 摘要
  python3 t3r_env_lists.py <venv> --req <文件> [--skip-direct-url]   另写一份“名字==版本”清单（版本照 METADATA 原样）
  python3 t3r_env_lists.py <我们的 venv> --compare <参照 venv> [--skip-direct-url]   逐个发行版比版本与摘要，有不同退出码 1
  python3 t3r_env_lists.py <我们的 venv> --compare-list <参照清单> [--skip-direct-url]    参照是本工具第一种用法存下的输出（记录文件，# 开头的行与末行不算），
                                                                                  比的时候不用再读参照的 venv（1b 建环境时她的 venv 被盖住）
测试：stages/recovery/test_t3r_env_lists.py。
"""
import argparse
import csv
import hashlib
import re
import sys
from pathlib import Path

SKIP_NAMES = {"INSTALLER", "REQUESTED", "direct_url.json", "RECORD"}


def norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def site_packages(venv: Path) -> Path:
    found = sorted(venv.glob("lib/python3.*/site-packages"))
    if len(found) != 1:
        raise SystemExit(f"{venv}：site-packages 不是恰好一个（{len(found)} 个）")
    return found[0]


def _meta(dist: Path) -> tuple[str, str]:
    name = version = None
    for line in (dist / "METADATA").read_text(encoding="utf-8", errors="replace").splitlines():
        if name is None and line.startswith("Name:"):
            name = line.split(":", 1)[1].strip()
        elif version is None and line.startswith("Version:"):
            version = line.split(":", 1)[1].strip()
        if not line.strip():
            break
    if name is None or version is None:
        raise SystemExit(f"{dist}：METADATA 里没有 Name 或 Version")
    return name, version


def record_digest(dist: Path) -> tuple[int, str]:
    rows = []
    with open(dist / "RECORD", newline="", encoding="utf-8") as fh:
        for row in csv.reader(fh):
            if len(row) < 2 or not row[1]:
                continue
            path, h = row[0], row[1]
            base = path.rsplit("/", 1)[-1]
            if path.startswith("../") or path.endswith(".pyc") or "__pycache__/" in path:
                continue
            if ".dist-info/" in path and base in SKIP_NAMES:
                continue
            first, _, rest = path.partition("/")
            if first.endswith(".dist-info"):  # 目录名的大小写与分隔符随装法可能不同，内容照样比
                path = "<dist-info>/" + rest
            rows.append(f"{path}\t{h}")
    rows.sort()
    return len(rows), hashlib.sha256("\n".join(rows).encode()).hexdigest()[:16]


def scan(venv: Path) -> dict[str, dict]:
    out = {}
    for dist in sorted(site_packages(venv).glob("*.dist-info")):
        name, version = _meta(dist)
        n, digest = record_digest(dist)
        installer = (dist / "INSTALLER").read_text().strip() if (dist / "INSTALLER").exists() else "-"
        key = norm(name)
        if key in out:
            raise SystemExit(f"{venv}：{name} 有两份 dist-info")
        out[key] = {"name": name, "version": version, "installer": installer,
                    "direct_url": (dist / "direct_url.json").exists(), "files": n, "digest": digest}
    return out


def load_list(path: Path) -> dict[str, dict]:
    """读第一种用法的输出（scripts/dev/record 的记录也行：跳过 # 开头的行与 T3R_ 开头的末行）。"""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#") or line.startswith("T3R_"):
            continue
        k, version, installer, du, files, digest = line.split("\t")
        out[k] = {"name": k, "version": version, "installer": installer, "direct_url": du == "direct_url",
                  "files": int(files), "digest": digest}
    if not out:
        raise SystemExit(f"{path}：一个发行版都没读到")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("venv")
    ap.add_argument("--req")
    ap.add_argument("--compare")
    ap.add_argument("--compare-list")
    ap.add_argument("--skip-direct-url", action="store_true")
    a = ap.parse_args()
    ours = scan(Path(a.venv))
    if a.skip_direct_url:
        ours = {k: v for k, v in ours.items() if not v["direct_url"]}
    if a.req:
        Path(a.req).write_text("".join(f"{v['name']}=={v['version']}\n" for _, v in sorted(ours.items())))
    if not a.compare and not a.compare_list:
        for k, v in sorted(ours.items()):
            print(f"{k}\t{v['version']}\t{v['installer']}\t{'direct_url' if v['direct_url'] else '-'}\t{v['files']}\t{v['digest']}")
        print(f"T3R_ENV_LIST_DONE {len(ours)} 个发行版")
        return 0
    ref = scan(Path(a.compare)) if a.compare else load_list(Path(a.compare_list))
    if a.skip_direct_url:
        ref = {k: v for k, v in ref.items() if not v["direct_url"]}
    bad = 0
    for k in sorted(set(ours) | set(ref)):
        o, r = ours.get(k), ref.get(k)
        if o is None or r is None:
            print(f"缺\t{k}\t{'我们没有' if o is None else '参照没有'}")
            bad += 1
        elif o["version"] != r["version"]:
            print(f"版本不同\t{k}\t{o['version']} 对 {r['version']}")
            bad += 1
        elif o["digest"] != r["digest"]:
            print(f"内容不同\t{k}\t{o['version']}\t文件 {o['files']} 对 {r['files']}\t摘要 {o['digest']} 对 {r['digest']}")
            bad += 1
    print(f"T3R_ENV_COMPARE_DONE 发行版 {len(ref)} 个（参照），不同 {bad} 个")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
