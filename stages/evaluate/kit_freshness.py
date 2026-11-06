"""提交前检查的 C6 与 R4（计划 agent-doc/plan-submission-check-2026-10-03.md）：Mac 上、仓库根执行，只读，只用标准库。
C6：下载官方 kit（几百 KB）到内存，算 sha256，与我们用的 update_0f2f9f4b 比；另打 HEAD 的 Last-Modified。
R4：Hugging Face 数据集 nvidia/video_to_data_challenge 当前 main 的提交与时间；本地 data/challenge/track_2 下的每个文件与 HF 上同路径的文件比哈希
    （LFS 文件比 sha256，普通文件比 git blob 的 sha1）。数据集若要登录，从 .env 读 HF_TOKEN，只在本进程的请求头里用，不打印、不进命令行。
    再从挑战页面里摘出含日期与 deadline / freeze / leaderboard 字样的行。
最后一行 KIT_FRESHNESS OK（kit 仍是 0f2f9f4b、本地数据与 HF 一致）或 KIT_FRESHNESS CHANGED（附原因）。
用法：scripts/dev/record submission/kit_freshness_<日期> python3 stages/evaluate/kit_freshness.py
"""
import hashlib
import html
import json
import re
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KIT_URL = "https://nvidia-isaac.github.io/video_to_data/v2d_challenge/assets/v2d_submission_kit.zip"
KIT_SHA = "0f2f9f4bb01082a48ab684750cbcbddd985c2de7031847928ae124e53fa605d7"
PAGE = "https://nvidia-isaac.github.io/video_to_data/v2d_challenge/"
REPO = "nvidia/video_to_data_challenge"
LOCAL = ROOT / "data/challenge"


def token():
    env = ROOT / ".env"
    if env.is_file():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("HF_TOKEN="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def get(url: str, auth: bool = False, method: str = "GET"):
    req = urllib.request.Request(url, method=method, headers={"User-Agent": "v2d-kit-freshness"})
    if auth and (t := token()):
        req.add_header("Authorization", f"Bearer {t}")
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.headers, (r.read() if method == "GET" else b"")


def hf_json(path: str):
    url = f"https://huggingface.co/api/datasets/{REPO}{path}"
    try:
        return json.loads(get(url)[1])
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            return json.loads(get(url, auth=True)[1])
        raise


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def main() -> int:
    changed = []
    print("== C6 kit")
    head, _ = get(KIT_URL, method="HEAD")
    print(f"HEAD {KIT_URL}：Last-Modified {head.get('Last-Modified')}，Content-Length {head.get('Content-Length')}，ETag {head.get('ETag')}")
    saved = {}
    for line in (ROOT / "third_party/v2d_submission_kit/update_0f2f9f4b/headers.txt").read_text(encoding="utf-8").splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            saved[k.strip().lower()] = v.strip()
    local_size = (ROOT / "third_party/v2d_submission_kit/update_0f2f9f4b/v2d_submission_kit.zip").stat().st_size
    for key, want in (("last-modified", saved.get("last-modified")), ("etag", saved.get("etag")),
                      ("content-length", saved.get("content-length") or str(local_size))):
        got_h = head.get(key)
        print(f"  {key}：现在 {got_h}，10/1 记下的 {want}：{'相同' if got_h == want else '不同'}")
    _, body = get(KIT_URL)
    got = hashlib.sha256(body).hexdigest()
    print(f"远端 kit sha256 {got}（{len(body)} 字节）；我们用的 {KIT_SHA}：{'相同' if got == KIT_SHA else '不同'}")
    if got != KIT_SHA:
        changed.append("kit 有新版本")

    print("\n== R4 数据集")
    info = hf_json("")
    print(f"HF {REPO}：sha {info.get('sha')}，lastModified {info.get('lastModified')}")
    tree = hf_json("/tree/main/track_2?recursive=true")
    remote = {e["path"]: e for e in tree if e.get("type") == "file"}
    local_files = sorted(p for p in (LOCAL / "track_2").rglob("*") if p.is_file())
    same = differ = absent = 0
    for p in local_files:
        rel = p.relative_to(LOCAL).as_posix()
        e = remote.get(rel)
        if e is None:
            absent += 1
            print(f"  HF 上没有 {rel}")
            continue
        data = p.read_bytes()
        if e.get("lfs"):
            ok = hashlib.sha256(data).hexdigest() == e["lfs"]["oid"]
        else:
            ok = git_blob_sha1(data) == e.get("oid")
        if ok:
            same += 1
        else:
            differ += 1
            print(f"  不同：{rel}")
    only_remote = sorted(set(remote) - {p.relative_to(LOCAL).as_posix() for p in local_files})
    print(f"本地 track_2 下 {len(local_files)} 个文件：与 HF 相同 {same}，不同 {differ}，HF 上没有 {absent}；HF 上有、本地没有的 {len(only_remote)} 个"
          f"（例：{only_remote[:5]}）")
    if differ:
        changed.append(f"{differ} 个数据文件与 HF 当前版本不同")

    print("\n== R4 挑战页面里与日期有关的行")
    _, page = get(PAGE)
    text = html.unescape(re.sub(r"<[^>]+>", "\n", page.decode("utf-8", "replace")))
    lines = [re.sub(r"\s+", " ", x).strip() for x in text.splitlines()]
    hits = [x for x in lines if x and re.search(r"(deadline|freeze|leaderboard|closes|final submission|November|Nov\.? \d)", x, re.IGNORECASE)]
    for x in dict.fromkeys(hits):
        print(f"  {x[:200]}")
    print(f"\nKIT_FRESHNESS {'OK' if not changed else 'CHANGED（' + '；'.join(changed) + '）'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
