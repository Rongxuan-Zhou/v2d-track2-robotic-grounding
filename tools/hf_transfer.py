"""经 Hugging Face 私有仓库在训练机之间转目录（训练机上、仓库根执行；凭据只经环境变量 HF_TOKEN，代理用环境里的 https_proxy）。
计划 agent-doc/plan-d-t3hm-train-2026-10-12.md 第 8 节（第 3 版：用户 10/13 定“需要传数据还是得HF”，经 Mac 的管道太慢）。

  push --repo <用户/仓库> --tag <标签> --dirs <目录>...    发送端，要 huggingface_hub（mf4090 上有）。
      1. 环境：HF_TOKEN 要有；HF_ENDPOINT 要么不设、要么是 https://huggingface.co（服务器 .env 把它指向下载镜像，带 token 的请求不能经它转发）。
      2. 仓库必须已存在、私有、根目录有 README.md（先用 hf_archive.py readme 建）。
      3. 各目录下只许有普通文件与目录；逐个算 sha256，清单 <标签>/MANIFEST.sha256（sha256sum 的格式，路径相对仓库根，按字节序排）。
      4. 仓库里 <标签>/ 下已有文件时：与清单完全一致就不重传（同一个标签可以重下）；是清单的一部分且内容都对就接着传；
         有清单外的文件或内容不对就停。核对内容：LFS 文件比 LFS sha256，其余文件比 git blob 的 sha1。
      5. 硬链接进暂存目录 runs/hf_xfer/<标签>/（已存在就停），upload_large_folder 只传 <标签>/**（私有，2 个并发）；出错时删掉暂存的硬链接再报错。
      6. 列仓库 <标签>/ 下的文件，路径集合、大小、内容都要与清单加 MANIFEST.sha256 一致；删暂存目录；最后一行 HF_PUSH_OK <文件数>。
  pull --repo <用户/仓库> --tag <标签> --dest <目录>       接收端，只用标准库（ml40 的主机 Python 没有 huggingface_hub）。
      1. dest 必须已存在且是空目录；下载 <标签>/MANIFEST.sha256，每行的 sha256 是 64 位十六进制，路径是相对路径、不含 .. 与 .、不重复，
         落点必须在 dest 里面。
      2. 按清单逐个下载 <端点>/<仓库>/resolve/main/<标签>/<路径> 到 <dest>/<路径>.part：只对与起点同协议、同主机的地址带 Authorization，
         重定向到别处（LFS/xet 的 CDN）时不带；边下边算 sha256，与清单相同才改名；不同或出错就隔几秒重下，最多 3 次，还不对就停。
      3. 最后一行 HF_PULL_OK <文件数>，或 HF_PULL_FAILED <原因>。
token 只从环境变量读，不打印；--endpoint 只给测试用（默认 https://huggingface.co）。测试：tools/test_hf_transfer.py。
"""
import argparse
import hashlib
import os
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

MANIFEST = "MANIFEST.sha256"
STAGE = Path("runs/hf_xfer")
HUB = "https://huggingface.co"
RETRY_SLEEP_S = float(os.environ.get("V2D_XFER_RETRY_SLEEP_S", "5"))


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git_blob_sha1(p: Path) -> str:
    h = hashlib.sha1(b"blob %d\0" % p.stat().st_size)
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def safe_rel(p: str) -> bool:
    q = Path(p)
    return (bool(p) and not q.is_absolute() and bool(q.parts) and all(x not in ("..", ".") for x in q.parts)
            and "\\" not in p and not p.startswith("./") and "//" not in p and not p.endswith("/"))


def _api():  # 测试里换成假的
    from huggingface_hub import HfApi
    return HfApi()


def _tree(api, repo: str, tag: str) -> dict:
    """仓库里 <标签>/ 下每个文件的 (大小, LFS sha256 或 None, blob id)。标签目录还不存在时 HF 回 404，当作空（10/13 冒烟发现）。"""
    got = {}
    try:
        for f in api.list_repo_tree(repo, repo_type="model", path_in_repo=tag, recursive=True, expand=True):
            if getattr(f, "size", None) is not None:  # 目录没有 size
                got[f.path] = (f.size, getattr(getattr(f, "lfs", None), "sha256", None), getattr(f, "blob_id", None))
    except Exception as e:  # noqa: BLE001
        if type(e).__name__ in ("RemoteEntryNotFoundError", "EntryNotFoundError") and not got:
            return {}
        raise
    return got


def _diff(want: dict, got: dict) -> tuple:
    """want：路径 → (大小, sha256, git blob sha1)。返回 (缺的, 内容不对的, 多出的)。"""
    missing = [p for p in want if p not in got]
    bad = []
    for p in want:
        if p not in got:
            continue
        (size, lfs, blob), (wsize, wsha, wblob) = got[p], want[p]
        if size != wsize or (lfs is not None and lfs != wsha) or (lfs is None and blob != wblob):
            bad.append(p)
    extra = sorted(set(got) - set(want))
    return missing, bad, extra


def cmd_push(a) -> int:
    if not os.environ.get("HF_TOKEN"):
        sys.exit("缺环境变量 HF_TOKEN（经 stdin 注入，不要写在命令行上）")
    ep = os.environ.get("HF_ENDPOINT", "")
    if ep and ep.rstrip("/") != HUB:
        sys.exit("HF_ENDPOINT 不是 https://huggingface.co，带 token 的请求不能经它转发，停下")
    api = _api()
    tag = a.tag
    if not api.repo_exists(a.repo, repo_type="model"):
        sys.exit(f"仓库 {a.repo} 不存在（先用 hf_archive.py readme 建私有仓库）")
    if not bool(api.repo_info(a.repo, repo_type="model").private):
        sys.exit(f"仓库 {a.repo} 不是私有的，不往里传")
    if not api.file_exists(a.repo, "README.md", repo_type="model"):
        sys.exit(f"仓库 {a.repo} 根目录没有 README.md")
    rows = []
    for d in a.dirs:
        if not safe_rel(d) or not Path(d).is_dir():
            sys.exit(f"目录不对：{d}")
        for p in sorted(Path(d).rglob("*")):
            if p.is_symlink() or not (p.is_file() or p.is_dir()):
                sys.exit(f"有非普通文件：{p}")
            if p.is_file():
                rows.append((sha256(p), p.as_posix(), p.stat().st_size, git_blob_sha1(p)))
    rows.sort(key=lambda r: r[1].encode())
    if not rows or len({r[1] for r in rows}) != len(rows):
        sys.exit("文件列表为空或有重复")
    man = "".join(f"{h}  {rel}\n" for h, rel, _, _ in rows).encode("utf-8")
    want = {f"{tag}/{rel}": (size, h, b) for h, rel, size, b in rows}
    want[f"{tag}/{MANIFEST}"] = (len(man), hashlib.sha256(man).hexdigest(), hashlib.sha1(b"blob %d\0" % len(man) + man).hexdigest())
    missing, bad, extra = _diff(want, _tree(api, a.repo, tag))
    if bad or extra:
        sys.exit(f"仓库里 {tag}/ 下已有与清单不符的文件（不符 {len(bad)}、多出 {len(extra)}，例如 {(bad + extra)[:3]}），换一个标签")
    if not missing:
        print(f"仓库里 {tag}/ 已与清单一致（{len(want)} 个文件），不重传", flush=True)
        print(f"HF_PUSH_OK {len(rows)}")
        return 0
    root = STAGE / tag
    if root.exists():
        sys.exit(f"暂存目录 {root} 已存在")
    print(f"清单：{len(rows)} 个文件，{sum(r[2] for r in rows) / 1e6:.1f} MB；仓库里已有 {len(want) - len(missing)} 个；暂存 {root}；开始上传", flush=True)
    try:
        root.mkdir(parents=True)
        for _, rel, _, _ in rows:
            (root / rel).parent.mkdir(parents=True, exist_ok=True)
            os.link(rel, root / rel)
        (root / MANIFEST).write_bytes(man)
        if hasattr(api, "upload_large_folder"):
            api.upload_large_folder(repo_id=a.repo, folder_path=str(STAGE), repo_type="model", allow_patterns=[f"{tag}/**"],
                                    private=True, num_workers=2)
        else:  # 11/04 plan-t3gb6：ml40 gate 环境的 huggingface_hub 没有 upload_large_folder；仓库已存在且私有（上面核过），一次提交传 <标签>/
            api.upload_folder(repo_id=a.repo, folder_path=str(STAGE), repo_type="model", allow_patterns=[f"{tag}/**"],
                              commit_message=f"transfer {tag}")
    except BaseException:
        shutil.rmtree(root, ignore_errors=True)  # 只是硬链接，源文件不动
        shutil.rmtree(STAGE / ".cache", ignore_errors=True)
        raise
    missing, bad, extra = _diff(want, _tree(api, a.repo, tag))
    print(f"仓库里 {tag}/：清单 {len(want)} 个（含 {MANIFEST}）；缺 {len(missing)}、不符 {len(bad)}、多出 {len(extra)}"
          + (f"；例如 {(missing + bad + extra)[:3]}" if missing or bad or extra else ""), flush=True)
    if missing or bad or extra:
        sys.exit("HF_PUSH_FAILED 仓库里的文件与清单不符（暂存目录留着查）")
    shutil.rmtree(root)
    shutil.rmtree(STAGE / ".cache", ignore_errors=True)
    if STAGE.exists() and not any(STAGE.iterdir()):
        STAGE.rmdir()
    print(f"HF_PUSH_OK {len(rows)}")
    return 0


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def _origin(url: str) -> tuple:
    u = urllib.parse.urlsplit(url)
    return u.scheme, u.netloc


def _fetch(opener, url: str, token: str, out: Path) -> str:
    """下载到 out，返回 sha256。只对与起点同协议、同主机的地址带 token；重定向自己跟，最多 6 次。"""
    home = _origin(url)
    for _ in range(6):
        req = urllib.request.Request(url, headers={"User-Agent": "v2d-hf-transfer"})
        if token and _origin(url) == home:
            req.add_header("Authorization", f"Bearer {token}")
        try:
            resp = opener.open(req, timeout=120)
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308) and e.headers.get("Location"):
                url = urllib.parse.urljoin(url, e.headers["Location"])
                e.close()
                continue
            raise
        h = hashlib.sha256()
        with resp, open(out, "wb") as f:
            for b in iter(lambda: resp.read(1 << 20), b""):
                f.write(b)
                h.update(b)
        return h.hexdigest()
    raise RuntimeError("重定向太多")


def cmd_pull(a) -> int:
    dest = Path(a.dest)
    if not dest.is_dir() or any(dest.iterdir()):
        print(f"HF_PULL_FAILED {dest} 不存在或不是空目录")
        return 1
    token = os.environ.get("HF_TOKEN", "")
    if not token:
        print("HF_PULL_FAILED 缺环境变量 HF_TOKEN")
        return 1
    droot = dest.resolve()
    opener = urllib.request.build_opener(_NoRedirect)
    base = f"{a.endpoint.rstrip('/')}/{a.repo}/resolve/main/"
    q = lambda rel: base + urllib.parse.quote(f"{a.tag}/{rel}")  # noqa: E731
    man = dest / f".{MANIFEST}.part"
    try:
        _fetch(opener, q(MANIFEST), token, man)
        text = man.read_text(encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        print(f"HF_PULL_FAILED 下载 {MANIFEST} 失败：{type(e).__name__} {getattr(e, 'code', '')}")
        return 1
    finally:
        man.unlink(missing_ok=True)
    rows = []
    for line in text.splitlines():
        h, _, rel = line.partition("  ")
        ok = len(h) == 64 and all(c in "0123456789abcdef" for c in h) and safe_rel(rel)
        if ok:
            t = (droot / rel).resolve()
            ok = t != droot and droot in t.parents
        if not ok:
            print(f"HF_PULL_FAILED 清单里有读不懂或不安全的行：{line[:120]}")
            return 1
        rows.append((h, rel))
    if not rows or len({r for _, r in rows}) != len(rows):
        print("HF_PULL_FAILED 清单为空或有重复路径")
        return 1
    print(f"清单：{len(rows)} 个文件；开始下载到 {dest}", flush=True)
    total = 0
    for i, (h, rel) in enumerate(rows, 1):
        out = dest / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        part = out.with_name(out.name + ".part")
        for attempt in (1, 2, 3):
            try:
                got = _fetch(opener, q(rel), token, part)
            except Exception as e:  # noqa: BLE001
                got = f"出错 {type(e).__name__} {getattr(e, 'code', '')}"
            if got == h:
                total += part.stat().st_size
                part.rename(out)
                break
            part.unlink(missing_ok=True)
            if attempt == 3:
                print(f"HF_PULL_FAILED {rel}：三次都不对（{got[:40]}）")
                return 1
            time.sleep(RETRY_SLEEP_S * attempt)
        if i % 50 == 0 or i == len(rows):
            print(f"  {i} / {len(rows)}，{total / 1e6:.1f} MB", flush=True)
    print(f"HF_PULL_OK {len(rows)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("push")
    p.add_argument("--repo", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--dirs", nargs="+", required=True)
    g = sub.add_parser("pull")
    g.add_argument("--repo", required=True)
    g.add_argument("--tag", required=True)
    g.add_argument("--dest", required=True)
    g.add_argument("--endpoint", default=HUB)
    a = ap.parse_args()
    if not a.tag.replace("_", "").replace("-", "").isalnum() or not a.tag.isascii():
        sys.exit(f"标签只许字母、数字、下划线、连字符：{a.tag}")
    return cmd_push(a) if a.cmd == "push" else cmd_pull(a)


if __name__ == "__main__":
    sys.exit(main())
