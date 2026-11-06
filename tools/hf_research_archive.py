"""赛后研究归档：把一台机器上的 Track 2 数据按规则分成公开、私有两份，扫描凭据后上传 HF，再核对、抽查（只传不删）。
计划 agent-doc/plan-hf-research-archive-2026-11-04.md（第 3 版，按独立审查 NO GO 的 14 条修改）。训练机与 Mac 都能跑。

子命令，各自落盘，任何一步没过就停在那一步：
  plan     --machine M --work W --src 源=仓库内前缀[:auto|public|private] ... [--pack 目录=仓库内 tar 路径[:auto|public|private]] ...
           源可以是目录或单个文件；源之间不能嵌套。逐个文件归类：符号链接、排除（含 .git/.cache 段、官方代码与数据）、
           按文件名判定的凭据文件（直接挡下）、代码（私有）、其余（公开）。--pack 的目录打成 tar，成员同样过这些规则
           （auto：非代码一个 tar、代码另一个 .code.tar）。写 W/manifest.jsonl、W/symlinks.tsv、W/excluded.tsv、W/held.tsv。
  scan     --work W [--members a,b] [--max-private-gb 3]   已知凭据原值从 stdin 读（每行一个，至少 1 个，否则停），
           加通用样式扫清单里每个文件（safetensors 头是 JSON 时只扫头）。命中的移出清单记进 W/held.tsv（路径与规则名，
           不记命中的值）；公开文件里出现成员用户名的改成私有，记 W/member_ref.tsv。私有合计超过上限就停。
  stage    --work W [--min-free-gb 3] [--max-private-gb 3]   先清掉这台机器旧的暂存，再把清单里的文件硬链接到
           W/stage/<public|private>/<仓库内路径>；索引写进 W/stage/<vis>/INDEX/<机器>/。公开索引里出现成员名的行只进私有索引，
           索引里出现凭据就停。已知凭据原值同样从 stdin 读。
  readme   --repo R --vis V --file README.md   仓库不存在就按 V 新建（dataset），传 README.md，下载回来比 sha256。
  upload   --work W --repo R --vis V [--workers 4] [--min-free-gb 3]   仓库要已存在、可见性等于 V、根目录有 README.md。
  verify   --work W --repo R --vis V   比仓库 <机器>/ 与 INDEX/<机器>/ 下的文件与暂存：缺的、多的、大小或哈希不同的都算不一致。
  spotcheck --work W --repo R --vis V [--n 5] [--min-safetensors 2]   随机下载 n 个回来重算 sha256，比完删掉。
只用标准库加 huggingface_hub（要有 upload_large_folder，即 <2）；HF_ENDPOINT 必须没设或等于 https://huggingface.co。
"""
import argparse
import fnmatch
import hashlib
import json
import os
import random
import re
import shutil
import struct
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

VIS = ("public", "private")
CODE_SUFFIX = (".py", ".pyx", ".pxd", ".sh", ".bash", ".zsh", ".ipynb", ".patch", ".diff", ".bundle", ".c", ".cc", ".cpp",
               ".cxx", ".h", ".hpp", ".cu", ".cuh", ".js", ".ts", ".go", ".rs", ".java", ".cmake", ".mk")
CODE_NAME = re.compile(r"^(Dockerfile.*|Makefile|CMakeLists\.txt|requirements.*\.txt|pyproject\.toml|setup\.cfg|uv\.lock)$")
CODE_DIR = re.compile(r"(^|/)(\.github|code|code_tier\d+)(/|$)")  # 对源内相对路径匹配，不碰 <机器>/repo/ 这样的前缀
SECRET_NAME = re.compile(r"(^|/)(\.env(?!\.example$)[^/]*|\.netrc|_netrc|kaggle\.json|access_token|[^/]*\.pem|[^/]*\.key|"
                         r"id_(rsa|ed25519|ecdsa|dsa)[^/]*|[^/]*_history|\.git-credentials|credentials[^/]*|[^/]*\.token|token|stored_tokens|hosts\.yml|\.pypirc|\.npmrc|"
                         r"[^/]*\.p12|[^/]*\.pfx)$"
                         r"|(^|/)(\.ssh|\.docker|\.kaggle|\.aws|\.gnupg)/|(^|/)wandb/settings$")
OFFICIAL = "官方数据，README 写明下载来源；sha256 另记"
HASH_EXCLUDED = "sha256 另记"
# 排除：(glob, 原因)；对仓库内路径（<机器>/<前缀>/...）匹配，fnmatch 的 * 也匹配 /
EXCLUDE = [
    ("*/data/challenge/*", OFFICIAL),
    ("*/third_party/video_to_data/*", "官方工具包（上游 nvidia-isaac/video_to_data）的副本；sha256 另记"),
    ("*/t3recon/w/sam3d_body/*", "第三方模型权重（SAM 3D Body），从官方下载；sha256 另记"),
    ("*/container/c1104/hf/*.tar.zst", "提交容器，已在 maniphys-lab-v2d-track2-submission；sha256 另记"),
    ("*/container/c1104/fc_8c778f9/*", "与同机 fc_src/fc_8c778f9 相同（文件 sha256 列表的哈希一致），见私有仓库"),
    ("*/__pycache__/*", "字节码缓存"), ("*.pyc", "字节码缓存"),
    ("*/runs/hf_stage/*", "旧归档的硬链接暂存"), ("*/runs/hf_spotcheck/*", "旧归档的抽查临时目录"),
    ("*/runs/hf_readme_check/*", "旧归档的临时目录"), ("*/runs/cache/*", "缓存"),
]
SEGMENT_EXCLUDE = {".git": "git 元数据（代码另以 git bundle 保存）", ".cache": "缓存目录（HF 不接受含 .cache 的路径）",
                   ".uvcache": "uv 缓存"}
GENERIC = [
    ("hf_token", rb"hf_[A-Za-z0-9]{30,}"),
    ("private_key", rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    ("aws_key", rb"AKIA[0-9A-Z]{16}"),
    ("url_password", rb"://[^/\s:@'\"]+:[^/\s@'\"]+@"),
    ("wandb_key", rb"WANDB_API_KEY\s*[=:]\s*['\"]?[0-9a-f]{40}"),
    ("netrc_password", rb"\bpassword\s+[0-9a-f]{40}\b"),
    ("api_key_assign", rb"[Aa][Pp][Ii][_-]?[Kk][Ee][Yy]['\"]?\s*[=:]\s*['\"]?[A-Za-z0-9_\-]{32,}"),  # 不用内联忽略大小写标志：合并成一个表达式后 3.11+ 报错
    ("bearer", rb"[Aa]uthorization:\s*[Bb]earer\s+[A-Za-z0-9._\-]{20,}"),
    ("kaggle_token", rb"KGAT_[0-9a-f]{20,}"),
    ("github_token", rb"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"),
    ("sk_key", rb"\bsk-[A-Za-z0-9_\-]{32,}"),
    ("slack_token", rb"xox[abprs]-[A-Za-z0-9\-]{10,}"),
    ("json_private_key", rb"\"private_key\"\s*:"),
]
CHUNK = 1 << 24
OVERLAP = 1024
MAX_SECRET = 400


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(p: Path) -> str:
    h = hashlib.sha1(b"blob %d\0" % p.stat().st_size)
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def is_code(rel: str) -> bool:
    name = rel.rsplit("/", 1)[-1]
    return rel.endswith(CODE_SUFFIX) or bool(CODE_NAME.match(name)) or bool(CODE_DIR.search(rel))


def excluded(dest: str):
    for seg, why in SEGMENT_EXCLUDE.items():
        if seg in dest.split("/"):
            return why
    for pat, why in EXCLUDE:
        if fnmatch.fnmatch(dest, pat):
            return why
    return None


def classify(rel: str, dest: str, mode: str):
    """→ ('excluded', 原因) | ('held', 规则名) | ('public'|'private', None)。"""
    why = excluded(dest)
    if why:
        return "excluded", why
    if SECRET_NAME.search(rel) or SECRET_NAME.search(dest):
        return "held", "secret_filename"
    if mode != "auto":
        return mode, None
    return ("private" if is_code(rel) else "public"), None


def spec(s: str):
    """'左=右[:模式]' → (左, 右, 模式)。"""
    left, right = s.split("=", 1)
    mode = "auto"
    if ":" in right and right.rsplit(":", 1)[1] in ("auto", "public", "private"):
        right, mode = right.rsplit(":", 1)
    return Path(left), right.strip("/"), mode


def read_jsonl(p: Path) -> list:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def write_jsonl(p: Path, rows: list) -> None:
    p.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in rows))


def tsv(rows) -> str:
    return "".join("\t".join(str(x) for x in r) + "\n" for r in rows)


def cmd_plan(a) -> int:
    work = Path(a.work).resolve()
    (work / "tar").mkdir(parents=True, exist_ok=True)
    srcs = [spec(s) for s in a.src]
    roots = [s.resolve() for s, _, _ in srcs]
    for i, x in enumerate(roots):
        for j, y in enumerate(roots):
            if i != j and (x == y or y in x.parents):
                sys.exit(f"源嵌套：{x} 在 {y} 里，停下")
        if x == work or x in work.parents:
            sys.exit(f"源 {x} 包含工作目录 {work}，停下")
    packs = {p.resolve(): (dest, mode) for p, dest, mode in (spec(s) for s in a.pack)}
    st = {"rows": [], "links": [], "excl": [], "held": []}
    for (src, prefix, mode), root in zip(srcs, roots):
        base = f"{a.machine}/{prefix}"
        if root.is_file():
            add_file(st, root, root.name, base, mode, single=True)
            continue
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            d = Path(dirpath)
            if d.resolve() in packs:
                dest, pmode = packs.pop(d.resolve())
                pack_dir(st, d, f"{a.machine}/{dest}", pmode, work)
                dirnames[:] = []
                continue
            for name in sorted(dirnames):
                p = d / name
                if p.is_symlink():
                    st["links"].append((f"{base}/{p.relative_to(root).as_posix()}", os.readlink(p)))
                    dirnames.remove(name)
            for name in sorted(filenames):
                p = d / name
                add_file(st, p, p.relative_to(root).as_posix(), base, mode)
    if packs:
        sys.exit(f"--pack 的目录不在任何源里：{[str(x) for x in packs]}")
    rows = st["rows"]
    # --reuse：同一源路径、同样大小、不是本次新打的 tar，就沿用旧清单的 sha256（封榜后源文件不再变；若变了，
    # 上传时客户端按真实内容算哈希，verify 与清单比对会报不一致）
    old = {}
    if a.reuse and Path(a.reuse).exists():
        old = {(r["src"], r["size"]): r["sha256"] for r in read_jsonl(Path(a.reuse)) if "packed_from" not in r}
    todo = [r for r in rows if "packed_from" in r or (r["src"], r["size"]) not in old]
    for r in rows:
        if r not in todo:
            r["sha256"] = old[(r["src"], r["size"])]
    with ThreadPoolExecutor(a.threads) as ex:
        for r, h in zip(todo, ex.map(lambda r: sha256(Path(r["src"])), todo)):
            r["sha256"] = h
    print(f"sha256 沿用 {len(rows) - len(todo)} 个、新算 {len(todo)} 个", flush=True)
    dests = [r["dest"] for r in rows]
    if len(dests) != len(set(dests)):
        sys.exit("仓库内路径重复，停下")
    write_jsonl(work / "manifest.jsonl", rows)
    (work / "symlinks.tsv").write_text(tsv(st["links"]))
    (work / "excluded.tsv").write_text(tsv(st["excl"]))
    (work / "held.tsv").write_text(tsv(st["held"]))
    (work / "member_ref.tsv").unlink(missing_ok=True)  # 重跑 plan 后必须重跑 scan，stage 才认
    pub = [r for r in rows if r["vis"] == "public"]
    pri = [r for r in rows if r["vis"] == "private"]
    print(f"{a.machine}：公开 {len(pub)} 个 {sum(r['size'] for r in pub) / 1e9:.2f} GB；私有 {len(pri)} 个 "
          f"{sum(r['size'] for r in pri) / 1e9:.3f} GB；符号链接 {len(st['links'])}；排除 {len(st['excl'])} 个 "
          f"{sum(x[1] for x in st['excl']) / 1e9:.2f} GB；按文件名挡下 {len(st['held'])} 个；PLAN_DONE")
    return 0


def add_file(st, p: Path, rel: str, base: str, mode: str, single: bool = False) -> None:
    dest = base if single else f"{base}/{rel}"
    if p.is_symlink():
        st["links"].append((dest, os.readlink(p)))
        return
    if not p.is_file():
        st["excl"].append((dest, 0, "", "不是普通文件"))
        return
    kind, why = classify(rel, dest, mode)
    size = p.stat().st_size
    if kind == "public" and dest.endswith(ARCHIVE_SUFFIX):  # 别人打好的包：里面有 .git 就排除，有代码就进私有
        k = archive_kind(p)
        if k == "git":
            kind, why = "excluded", "含 git 元数据的打包（代码另以 git bundle 保存）；sha256 另记"
        elif k == "code":
            kind = "private"
    if kind == "public" and p.stat().st_mode & 0o077 == 0:  # 只有属主可读写的文件多半是凭据，不进公开仓库
        kind, why = "held", "owner_only_mode"
    if kind == "excluded":
        st["excl"].append((dest, size, sha256(p) if why.endswith(HASH_EXCLUDED) else "", why))
    elif kind == "held":
        st["held"].append((dest, size, why))
    else:
        st["rows"].append({"src": str(p), "dest": dest, "vis": kind, "size": size})


ARCHIVE_SUFFIX = (".tar", ".tar.gz", ".tgz", ".tar.bz2", ".tar.xz", ".zip", ".whl")


def archive_kind(p: Path) -> str:
    """'git'：成员路径里有 .git 段；'code'：有代码成员；'data'：都没有；打不开的按 'code' 处理（宁可进私有）。"""
    import zipfile
    try:
        if p.name.endswith((".zip", ".whl")):
            names = zipfile.ZipFile(p).namelist()
        else:
            with tarfile.open(p) as t:
                names = t.getnames()
    except Exception:
        return "code"
    if any(".git" in n.split("/") for n in names):
        return "git"
    return "code" if any(is_code(n) for n in names) else "data"


def pack_dir(st, d: Path, dest: str, mode: str, work: Path) -> None:
    groups = {"public": [], "private": []}
    base = dest[:-len(".tar")] if dest.endswith(".tar") else dest
    for dirpath, dirnames, filenames in os.walk(d, followlinks=False):
        for name in sorted(dirnames + filenames):
            p = Path(dirpath) / name
            rel = p.relative_to(d).as_posix()
            if p.is_dir() and not p.is_symlink():
                continue
            if p.is_symlink():
                groups["private" if mode == "private" else "public"].append(rel)  # 符号链接本身进 tar（tar 能存）
                continue
            kind, why = classify(rel, f"{base}/{rel}", mode)
            if kind == "excluded":
                st["excl"].append((f"{base}/{rel}", p.stat().st_size, "", why))
            elif kind == "held":
                st["held"].append((f"{base}/{rel}", p.stat().st_size, why))
            else:
                groups[kind].append(rel)
    for vis, mem in groups.items():
        if not mem:
            continue
        name = f"{base}.tar" if (mode != "auto" or vis == "public") else f"{base}.code.tar"
        tp = work / "tar" / name.replace("/", "__")
        with tarfile.open(tp, "w", format=tarfile.PAX_FORMAT) as t:
            for rel in mem:
                t.add(d / rel, arcname=rel, recursive=False)
        st["rows"].append({"src": str(tp), "dest": name, "vis": vis, "size": tp.stat().st_size,
                           "packed_from": base, "members": len(mem)})


def secrets_from_stdin() -> list:
    secrets = [x.strip() for x in sys.stdin.read().splitlines() if x.strip()]
    if not secrets:
        sys.exit("stdin 上没有已知凭据原值，停下（扫描必须带原值）")
    if any(len(s) < 8 or len(s) > MAX_SECRET for s in secrets):
        sys.exit(f"有已知凭据原值短于 8 或长于 {MAX_SECRET} 个字符，停下")
    return secrets


def build_rx(secrets: list, members: list):
    parts = [f"(?P<{n}>{p.decode()})".encode() for n, p in GENERIC]
    parts += [b"(?P<known_%d>%s)" % (i, re.escape(s.encode())) for i, s in enumerate(secrets)]
    mrx = re.compile(b"|".join(re.escape(m.encode()) for m in members)) if members else None
    return re.compile(b"|".join(parts)), mrx


def scan_bytes(chunks, rx, mrx):
    hits, member, tail = set(), False, b""
    for chunk in chunks:
        buf = tail + chunk
        for m in rx.finditer(buf):
            hits.add(m.lastgroup)
        if mrx is not None and not member and mrx.search(buf):
            member = True
        tail = buf[-OVERLAP:]
    return hits, member


def scan_file(p: Path, rx, mrx):
    """返回 (命中的凭据规则名集合, 是否出现成员用户名)。safetensors 的头是 JSON 时只扫头，否则全量扫。"""
    with p.open("rb") as f:
        if p.suffix == ".safetensors" and p.stat().st_size >= 9:
            n = struct.unpack("<Q", f.read(8))[0]
            head = f.read(1)
            if head == b"{" and n <= 100 << 20:
                return scan_bytes([head + f.read(n - 1)], rx, mrx)
            f.seek(0)
        return scan_bytes(iter(lambda: f.read(CHUNK), b""), rx, mrx)


# 快扫：GNU grep 多进程先筛出可能命中的文件，再由 scan_file 精确判规则（结果与慢扫相同：ERE 与上面的 Python 样式等价，
# 已知原值与成员名按固定串匹配）。safetensors 不进 grep，照旧只扫 JSON 头。
GENERIC_ERE = [
    r"hf_[A-Za-z0-9]{30,}",
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    r"AKIA[0-9A-Z]{16}",
    r"://[^/[:space:]:@'\"]+:[^/[:space:]@'\"]+@",
    r"WANDB_API_KEY[[:space:]]*[=:][[:space:]]*['\"]?[0-9a-f]{40}",
    r"\bpassword[[:space:]]+[0-9a-f]{40}\b",
    r"[Aa][Pp][Ii][_-]?[Kk][Ee][Yy]['\"]?[[:space:]]*[=:][[:space:]]*['\"]?[A-Za-z0-9_-]{32,}",
    r"[Aa]uthorization:[[:space:]]*[Bb]earer[[:space:]]+[A-Za-z0-9._-]{20,}",
    r"KGAT_[0-9a-f]{20,}",
    r"gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}",
    r"\bsk-[A-Za-z0-9_-]{32,}",
    r"xox[abprs]-[A-Za-z0-9-]{10,}",
    r"\"private_key\"[[:space:]]*:",
]


def gnu_grep() -> bool:
    try:
        import subprocess
        return "GNU grep" in subprocess.run(["grep", "--version"], capture_output=True, text=True).stdout
    except OSError:
        return False


def grep_hits(files: list, patterns: str, fixed: bool, threads: int) -> set:
    """返回 files 里被 grep 命中的文件集合；pattern 经 stdin 交给 grep（-f -），不进命令行。grep 出错（退出码 2）就停。"""
    import subprocess
    def run(chunk):
        r = subprocess.run(["grep", "-a", "-l", "-s", "-F" if fixed else "-E", "-f", "-", "--", *chunk],
                           input=patterns.encode(), capture_output=True)
        if r.returncode not in (0, 1):
            sys.exit(f"grep 出错（{r.returncode}）：{r.stderr.decode(errors='replace')[:200]}")
        return set(r.stdout.decode(errors="surrogateescape").splitlines())
    chunks = [files[i:i + 200] for i in range(0, len(files), 200)]
    out = set()
    with ThreadPoolExecutor(threads) as ex:
        for x in ex.map(run, chunks):
            out |= x
    return out


def fast_scan(rows: list, secrets: list, members: list, rx, mrx, threads: int) -> list:
    if len(GENERIC_ERE) != len(GENERIC):
        sys.exit("GENERIC_ERE 与 GENERIC 条数不同，停下")
    plain = [r["src"] for r in rows if not r["src"].endswith(".safetensors")]
    cand = grep_hits(plain, "\n".join(GENERIC_ERE) + "\n", False, threads)
    cand |= grep_hits(plain, "\n".join(secrets) + "\n", True, threads)
    mcand = grep_hits(plain, "\n".join(members) + "\n", True, threads) if members else set()
    res = []
    for r in rows:
        src = r["src"]
        if src.endswith(".safetensors") or src in cand:
            res.append(scan_file(Path(src), rx, mrx))
        else:
            res.append((set(), src in mcand))
    return res


def private_gb(rows: list) -> float:
    return sum(r["size"] for r in rows if r["vis"] == "private") / 1e9


def cmd_scan(a) -> int:
    work = Path(a.work)
    secrets = secrets_from_stdin()
    members = [m for m in a.members.split(",") if m]
    rx, mrx = build_rx(secrets, members)
    rows = read_jsonl(work / "manifest.jsonl")
    if a.fast and gnu_grep():
        res = fast_scan(rows, secrets, members, rx, mrx, a.threads)
    else:
        with ThreadPoolExecutor(a.threads) as ex:
            res = list(ex.map(lambda r: scan_file(Path(r["src"]), rx, mrx), rows))
    allow = {}  # W/allow.tsv：<仓库内路径>\t<规则名,...>，人工看过、确认是占位串的通用样式命中；已知原值（known_*）不能放行
    if (work / "allow.tsv").exists():
        for x in (work / "allow.tsv").read_text().splitlines():
            if x.strip():
                d, rules = x.split("\t")
                allow[d] = set(rules.split(","))
    if any(r.startswith("known_") for v in allow.values() for r in v):
        sys.exit("allow.tsv 里不能放行已知原值，停下")
    keep, held, mref = [], [], []
    for r, (hits, member) in zip(rows, res):
        if r["vis"] == "private":  # 私有文件（多是代码）只按已知原值拦；通用样式在代码里多是占位串（user:pass@host）
            hits = {h for h in hits if h.startswith("known_")}
        if hits and not hits <= allow.get(r["dest"], set()):
            held.append((r["dest"], r["size"], ",".join(sorted(hits))))
            continue
        if member and r["vis"] == "public":
            r["vis"] = "private"
            r["member_ref"] = True
            mref.append((r["dest"], r["size"]))
        keep.append(r)
    write_jsonl(work / "manifest.jsonl", keep)
    old = (work / "held.tsv").read_text() if (work / "held.tsv").exists() else ""
    (work / "held.tsv").write_text(old + tsv(held))
    (work / "member_ref.tsv").write_text(tsv(mref))
    print(f"凭据样式 {len(GENERIC)} 种、已知原值 {len(secrets)} 个、成员名 {len(members)} 个；扫 {len(rows)} 个文件；"
          f"挡下 {len(held)} 个；因成员名改私有 {len(mref)} 个（{sum(s for _, s in mref) / 1e6:.1f} MB）；私有合计 {private_gb(keep):.3f} GB")
    if private_gb(keep) > a.max_private_gb:
        sys.exit(f"私有合计超过 {a.max_private_gb} GB，停下")
    print(f"SCAN_DONE {sum(r['vis'] == 'public' for r in keep)} {sum(r['vis'] == 'private' for r in keep)} {len(held)}")
    return 0


def free_gb(p: Path) -> float:
    return shutil.disk_usage(p).free / 1e9


def cmd_stage(a) -> int:
    work = Path(a.work)
    secrets = secrets_from_stdin()
    members = [m for m in a.members.split(",") if m]
    rx, mrx = build_rx(secrets, members)
    rows = read_jsonl(work / "manifest.jsonl")
    machine = rows[0]["dest"].split("/", 1)[0]
    if not (work / "member_ref.tsv").exists():
        sys.exit("还没跑 scan，停下")
    if private_gb(rows) > a.max_private_gb:
        sys.exit(f"私有合计 {private_gb(rows):.2f} GB 超过 {a.max_private_gb} GB，停下")
    if free_gb(work) < a.min_free_gb:
        sys.exit(f"可用 {free_gb(work):.1f} GB，低于 {a.min_free_gb} GB，停下")
    for vis in VIS:  # 先清这台机器旧的暂存：删的是硬链接，不动源文件
        for old in (work / "stage" / vis / machine, work / "stage" / vis / "INDEX" / machine):
            shutil.rmtree(old, ignore_errors=True)
    copied = 0
    for r in rows:
        dst = work / "stage" / r["vis"] / r["dest"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(r["src"], dst)
        except PermissionError:  # 别的用户（如 docker 里的 root）的文件，protected_hardlinks 不许建硬链接，就复制一份
            shutil.copyfile(r["src"], dst)
            copied += r["size"]
            if free_gb(work) < a.min_free_gb:
                sys.exit(f"复制后可用 {free_gb(work):.1f} GB，低于 {a.min_free_gb} GB，停下")
    names = ["symlinks.tsv", "excluded.tsv", "held.tsv", "member_ref.tsv"]
    for vis in VIS:
        idx = work / "stage" / vis / "INDEX" / machine
        idx.mkdir(parents=True, exist_ok=True)
        write_jsonl(idx / "manifest.jsonl", [{k: v for k, v in r.items() if k != "src"} for r in rows if r["vis"] == vis])
        for n in names:
            if n == "member_ref.tsv" and vis == "public":
                continue
            lines = (work / n).read_text().splitlines(keepends=True)
            if vis == "public" and mrx is not None:
                lines = [x for x in lines if not mrx.search(x.encode())]  # 有成员名的行只进私有索引
            (idx / n).write_text("".join(lines))
        for p in idx.iterdir():
            hits, member = scan_file(p, rx, mrx)
            if hits or (vis == "public" and member):
                sys.exit(f"索引 {p} 里有凭据或成员名（{','.join(sorted(hits)) or 'member'}），停下")
    print(f"暂存 {len(rows)} 个文件（复制 {copied / 1e6:.1f} MB，其余是硬链接）；可用 {free_gb(work):.1f} GB；STAGE_DONE")
    return 0


def need_env():
    if not os.environ.get("HF_TOKEN"):
        sys.exit("没有 HF_TOKEN")
    if os.environ.get("HF_ENDPOINT", "https://huggingface.co") != "https://huggingface.co":
        sys.exit("HF_ENDPOINT 不是 https://huggingface.co，停下")
    import huggingface_hub
    from huggingface_hub import HfApi
    api = HfApi()
    if not hasattr(api, "upload_large_folder"):
        sys.exit(f"huggingface_hub {huggingface_hub.__version__} 没有 upload_large_folder，要 <2，停下")
    return api


def check_repo(api, repo: str, vis: str, need_readme: bool = True) -> None:
    if not api.repo_exists(repo, repo_type="dataset"):
        sys.exit(f"{repo} 不存在，停下")
    priv = bool(api.repo_info(repo, repo_type="dataset").private)
    if priv != (vis == "private"):
        sys.exit(f"{repo} 的可见性是 {'private' if priv else 'public'}，期望 {vis}，停下")
    if need_readme and not api.file_exists(repo, "README.md", repo_type="dataset"):
        sys.exit(f"{repo} 根目录没有 README.md，先跑 readme")


def cmd_readme(a) -> int:
    api = need_env()
    from huggingface_hub import hf_hub_download
    if not api.repo_exists(a.repo, repo_type="dataset"):
        api.create_repo(a.repo, repo_type="dataset", private=a.vis == "private")
    check_repo(api, a.repo, a.vis, need_readme=False)
    api.upload_file(path_or_fileobj=a.file, path_in_repo="README.md", repo_id=a.repo, repo_type="dataset",
                    commit_message="README")
    tmp = Path(a.file).resolve().parent / ".readme_check"
    got = sha256(Path(hf_hub_download(a.repo, "README.md", repo_type="dataset", local_dir=str(tmp), force_download=True)))
    shutil.rmtree(tmp, ignore_errors=True)
    ok = got == sha256(Path(a.file))
    print(f"{a.repo} README.md " + ("HF_README_OK" if ok else "HF_README_FAIL"))
    return 0 if ok else 1


def cmd_upload(a) -> int:
    api = need_env()
    check_repo(api, a.repo, a.vis)
    folder = Path(a.work) / "stage" / a.vis
    if free_gb(folder) < a.min_free_gb:
        sys.exit(f"可用 {free_gb(folder):.1f} GB，低于 {a.min_free_gb} GB，停下")
    t0 = time.time()
    print(f"{time.strftime('%F %T')} 开始上传 {folder} → {a.repo}（{a.vis}）；可用 {free_gb(folder):.1f} GB", flush=True)
    api.upload_large_folder(repo_id=a.repo, folder_path=str(folder), repo_type="dataset",
                            private=a.vis == "private", num_workers=a.workers)
    print(f"{time.strftime('%F %T')} 上传结束，用时 {(time.time() - t0) / 60:.1f} 分钟；可用 {free_gb(folder):.1f} GB；"
          "HF_UPLOAD_DONE", flush=True)
    return 0


def expected(work: Path, vis: str):
    """{仓库内路径: (本地文件, sha256)} —— 清单里这一份的文件（sha256 来自清单），加暂存 INDEX 下的索引文件（现算）。"""
    root = work / "stage" / vis
    rows = read_jsonl(work / "manifest.jsonl")
    machine = rows[0]["dest"].split("/", 1)[0]
    out = {r["dest"]: (Path(r["src"]), r["sha256"]) for r in rows if r["vis"] == vis}
    for p in sorted((root / "INDEX" / machine).iterdir()):
        out[p.relative_to(root).as_posix()] = (p, sha256(p))
    return machine, out


def cmd_verify(a) -> int:
    api = need_env()
    work = Path(a.work)
    machine, want = expected(work, a.vis)
    hub = {}
    for pre in (machine, f"INDEX/{machine}"):
        for f in api.list_repo_tree(a.repo, repo_type="dataset", path_in_repo=pre, recursive=True, expand=True):
            if hasattr(f, "size"):
                lfs = getattr(f, "lfs", None)
                hub[f.path] = (f.size, getattr(lfs, "sha256", None) if lfs else None, getattr(f, "blob_id", None))
    bad = [(x, "仓库里多出来") for x in sorted(set(hub) - set(want))]
    for dest, (p, want_sha) in want.items():
        h = hub.get(dest)
        size = p.stat().st_size
        if h is None:
            bad.append((dest, "仓库里没有"))
        elif h[0] != size:
            bad.append((dest, f"大小 {h[0]} ≠ {size}"))
        elif h[1]:
            if h[1] != want_sha:
                bad.append((dest, "sha256 不同"))
        elif h[2] != git_blob_sha1(p) or sha256(p) != want_sha:  # 普通 git 文件比 blob sha1，并确认本地没改过
            bad.append((dest, "blob sha1 不同或本地已改"))
    (work / f"verify_{a.vis}.json").write_text(json.dumps({"checked": len(want), "hub": len(hub), "bad": bad},
                                                           ensure_ascii=False))
    print(f"{a.repo}（{a.vis}）{machine}：应有 {len(want)} 个，仓库里 {len(hub)} 个，不一致 {len(bad)} 个"
          + ("；VERIFY_OK" if not bad else "；VERIFY_FAIL"))
    for x in bad[:20]:
        print("  ", *x)
    return 0 if not bad else 1


def cmd_spotcheck(a) -> int:
    need_env()
    from huggingface_hub import hf_hub_download
    work = Path(a.work)
    rows = [r for r in read_jsonl(work / "manifest.jsonl") if r["vis"] == a.vis]
    rng = random.Random(a.seed)
    st = [r for r in rows if r["dest"].endswith(".safetensors")]
    pick = rng.sample(st, min(a.min_safetensors, len(st)))
    rest = [r for r in rows if r not in pick]
    pick += rng.sample(rest, min(a.n - len(pick), len(rest)))
    tmp = work / "spotcheck"
    bad = 0
    for r in pick:
        got = sha256(Path(hf_hub_download(a.repo, r["dest"], repo_type="dataset", local_dir=str(tmp), force_download=True)))
        ok = got == r["sha256"]
        bad += not ok
        print(f"  {'OK ' if ok else 'BAD'} {r['dest']}（{r['size'] / 1e6:.1f} MB）", flush=True)
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"抽查 {len(pick)} 个，不一致 {bad} 个；" + ("SPOTCHECK_OK" if not bad else "SPOTCHECK_FAIL"))
    return 0 if not bad else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--machine", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--src", action="append", required=True)
    p.add_argument("--pack", action="append", default=[])
    p.add_argument("--reuse", default="")
    p.add_argument("--threads", type=int, default=4)
    for name in ("scan", "stage"):
        p = sub.add_parser(name)
        p.add_argument("--work", required=True)
        p.add_argument("--members", default="")
        p.add_argument("--max-private-gb", type=float, default=3.0)
        if name == "scan":
            p.add_argument("--threads", type=int, default=4)
            p.add_argument("--fast", action="store_true")
        else:
            p.add_argument("--min-free-gb", type=float, default=3.0)
    p = sub.add_parser("readme")
    p.add_argument("--repo", required=True)
    p.add_argument("--vis", choices=VIS, required=True)
    p.add_argument("--file", required=True)
    for name in ("upload", "verify", "spotcheck"):
        p = sub.add_parser(name)
        p.add_argument("--work", required=True)
        p.add_argument("--repo", required=True)
        p.add_argument("--vis", choices=VIS, required=True)
        if name == "upload":
            p.add_argument("--workers", type=int, default=4)
            p.add_argument("--min-free-gb", type=float, default=3.0)
        if name == "spotcheck":
            p.add_argument("--n", type=int, default=5)
            p.add_argument("--min-safetensors", type=int, default=2)
            p.add_argument("--seed", type=int, default=int(time.time()))
    a = ap.parse_args()
    return {"plan": cmd_plan, "scan": cmd_scan, "stage": cmd_stage, "readme": cmd_readme, "upload": cmd_upload,
            "verify": cmd_verify, "spotcheck": cmd_spotcheck}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
