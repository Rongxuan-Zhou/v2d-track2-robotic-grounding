"""hf_transfer.py 的行为测试（暂无契约 ID：运行规范待成文）。只用标准库，Mac 与训练机都能跑。

push：进程内调用，HfApi 换成假的（把 upload_large_folder 传的文件记在内存里，list_repo_tree 按它回报：大于 20 字节的带 LFS sha256，
其余带 git blob sha1；标签目录不存在时与真的一样抛 RemoteEntryNotFoundError）。
  1. 正常：清单按字节序、sha256 对，仓库里是全部文件加 MANIFEST.sha256，暂存目录删掉，HF_PUSH_OK；
  2. 仓库里已与清单一致：不再上传，HF_PUSH_OK；只传了一部分且内容都对：接着传；
  3. 缺 HF_TOKEN、HF_ENDPOINT 指向镜像、仓库不存在、是公开的、没有 README、标签下有清单外的文件或内容不对、目录里有符号链接、
     目录不安全、暂存目录已存在：都停，不上传；
  4. 上传时出错：暂存目录删掉、报错；上传后仓库里缺文件或大小不对：HF_PUSH_FAILED，暂存目录留着。
pull：子进程调用，本机起两个 HTTP 服务：一个当 HF（小文件直接给，大文件 302 到另一个端口），一个当 CDN。重试间隔设成 0。
  5. 正常：路径里有“=”和空格也能下，文件逐字节相同，没有 .part 残留，CDN 没收到 Authorization，HF 收到了；HF_PULL_OK；
  6. 一个文件前两次给错、第三次给对：照样成功；三次都错、文件 404、重定向转圈：HF_PULL_FAILED，没有 .part 残留；
  7. 清单 404、目标目录非空、缺 HF_TOKEN、清单里有 ../、.、绝对路径、反斜杠、./、//、大写或长度不对的 sha256、重复路径、空清单：
     都 HF_PULL_FAILED，目标目录外不写任何文件；
  8. 带不带 token 只看协议与主机：同主机换协议也不带。
用法：python3 tools/test_hf_transfer.py
"""
import argparse
import contextlib
import fnmatch
import hashlib
import http.server
import importlib.util
import io
import os
import subprocess
import sys
import tempfile
import threading
import urllib.parse
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("hf_transfer", HERE / "hf_transfer.py")
ht = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ht)


class RemoteEntryNotFoundError(Exception):
    """同 huggingface_hub 1.x：list_repo_tree 列一个不存在的目录时抛它（HTTP 404）。"""


def blob(b: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest()


class FakeApi:
    def __init__(self, exists=True, private=True, readme=True, preset=None, corrupt=None, drop=None, fail=False):
        self.exists, self.private, self.readme, self.corrupt, self.drop, self.fail = exists, private, readme, corrupt, drop, fail
        self.uploaded, self.calls = dict(preset or {}), 0

    def repo_exists(self, repo, repo_type):
        return self.exists

    def repo_info(self, repo, repo_type):
        return SimpleNamespace(private=self.private)

    def file_exists(self, repo, path, repo_type):
        return self.readme

    def upload_large_folder(self, repo_id, folder_path, repo_type, allow_patterns, private, num_workers):
        self.calls += 1
        if self.fail:
            raise RuntimeError("连续 8 次出错")
        root = Path(folder_path)
        (root / ".cache/huggingface").mkdir(parents=True, exist_ok=True)
        for p in root.rglob("*"):
            rel = p.relative_to(root).as_posix()
            if p.is_file() and ".cache" not in p.parts and fnmatch.fnmatch(rel, allow_patterns[0]):
                if not (self.drop and rel.endswith(self.drop)):
                    self.uploaded[rel] = p.read_bytes()

    def list_repo_tree(self, repo, repo_type, path_in_repo, recursive, expand):
        if not any(rel.startswith(path_in_repo + "/") for rel in self.uploaded):
            raise RemoteEntryNotFoundError(f"{path_in_repo} does not exist on main")
        out = [SimpleNamespace(path=path_in_repo + "/runs")]
        for rel, b in self.uploaded.items():
            if rel.startswith(path_in_repo + "/"):
                size = len(b) + (1 if self.corrupt and rel.endswith(self.corrupt) else 0)
                lfs = SimpleNamespace(sha256=hashlib.sha256(b).hexdigest()) if len(b) > 20 else None
                out.append(SimpleNamespace(path=rel, size=size, lfs=lfs, blob_id=blob(b)))
        return out


def make_tree(root: Path) -> dict:
    files = {"runs/a/x.bin": os.urandom(100), "runs/a/sub/sequence_id=s1/y.txt": b"small", "runs/b/z.json": b'{"tier": 3}'}
    for rel, b in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(b)
    return files


def manifest(files: dict, tag="t1") -> dict:
    man = "".join(f"{hashlib.sha256(files[r]).hexdigest()}  {r}\n" for r in sorted(files, key=str.encode)).encode()
    return {**{f"{tag}/{r}": b for r, b in files.items()}, f"{tag}/MANIFEST.sha256": man}


def push(tmp: Path, api: FakeApi, dirs=("runs/a", "runs/b"), tag="t1", env=None) -> tuple:
    ht._api = lambda: api
    out, rc = io.StringIO(), 0
    old, old_env = os.getcwd(), dict(os.environ)
    os.environ.update({"HF_TOKEN": "tok"})
    os.environ.pop("HF_ENDPOINT", None)
    for k, v in (env or {}).items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    os.chdir(tmp)
    try:
        with contextlib.redirect_stdout(out):
            rc = ht.cmd_push(argparse.Namespace(repo="u/r", tag=tag, dirs=list(dirs)))
    except SystemExit as e:
        rc = 1
        out.write(str(e.code) + "\n")
    except RuntimeError as e:
        rc = 1
        out.write(f"RuntimeError {e}\n")
    finally:
        os.chdir(old)
        os.environ.clear()
        os.environ.update(old_env)
    return rc, out.getvalue()


def test_push_ok() -> None:
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        files = make_tree(t)
        api = FakeApi()
        rc, out = push(t, api)
        assert rc == 0 and out.rstrip().endswith("HF_PUSH_OK 3"), out
        assert api.uploaded == manifest(files), sorted(api.uploaded)
        assert not (t / "runs/hf_xfer").exists(), "暂存目录没删"


def test_push_already_there_or_partial() -> None:
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        files = make_tree(t)
        api = FakeApi(preset=manifest(files))
        rc, out = push(t, api)
        assert rc == 0 and api.calls == 0 and "不重传" in out and out.rstrip().endswith("HF_PUSH_OK 3"), out
        part = {k: v for k, v in manifest(files).items() if k.endswith("x.bin")}
        api = FakeApi(preset=part)
        rc, out = push(t, api)
        assert rc == 0 and api.calls == 1 and api.uploaded == manifest(files) and "仓库里已有 1 个" in out, out


def test_push_refusals() -> None:
    cases = (({}, {"env": {"HF_TOKEN": None}}, "HF_TOKEN"), ({}, {"env": {"HF_ENDPOINT": "https://hf-mirror.com"}}, "HF_ENDPOINT"),
             (dict(exists=False), {}, "不存在"), (dict(private=False), {}, "不是私有"), (dict(readme=False), {}, "README"),
             (dict(preset={"t1/old.bin": b"x"}), {}, "已有与清单不符"), (dict(preset="bad"), {}, "已有与清单不符"),
             ({}, {"link": True}, "非普通文件"), ({}, {"dirs": ("../a",)}, "目录不对"), ({}, {"stage": True}, "暂存目录"))
    for api_kw, kw, want in cases:
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            files = make_tree(t)
            if api_kw.get("preset") == "bad":
                api_kw = dict(preset={"t1/runs/b/z.json": b'{"tier": 2}'})
            if kw.get("link"):
                (t / "runs/a/l.bin").symlink_to("x.bin")
            if kw.get("stage"):
                (t / "runs/hf_xfer/t1").mkdir(parents=True)
            api = FakeApi(**api_kw)
            before = dict(api.uploaded)
            rc, out = push(t, api, dirs=kw.get("dirs", ("runs/a", "runs/b")), env=kw.get("env"))
            assert rc == 1 and want in out and api.calls == 0 and api.uploaded == before, (api_kw, kw, out)


def test_push_upload_errors() -> None:
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        make_tree(t)
        rc, out = push(t, FakeApi(fail=True))
        assert rc == 1 and "连续 8 次出错" in out and not (t / "runs/hf_xfer/t1").exists(), out
    for kw, want in ((dict(corrupt="z.json"), "不符 1"), (dict(drop="y.txt"), "缺 1")):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            make_tree(t)
            rc, out = push(t, FakeApi(**kw))
            assert rc == 1 and "HF_PUSH_FAILED" in out and want in out and (t / "runs/hf_xfer/t1").exists(), (kw, out)


class Hub(http.server.BaseHTTPRequestHandler):
    files, cdn_port, seen, bad_n, loop = {}, 0, [], {}, False

    def log_message(self, *a):
        pass

    def do_GET(self):
        path = urllib.parse.unquote(self.path)
        Hub.seen.append(("hub", path, self.headers.get("Authorization")))
        if Hub.loop and not path.endswith("MANIFEST.sha256"):
            self.send_response(307)
            self.send_header("Location", self.path)
            self.end_headers()
            return
        pre = "/u/r/resolve/main/"
        rel = path[len(pre):] if path.startswith(pre) else None
        if rel not in Hub.files:
            self.send_response(404)
            self.end_headers()
            return
        if len(Hub.files[rel]) > 20:
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{Hub.cdn_port}/blob/{urllib.parse.quote(rel)}")
            self.end_headers()
            return
        self._send(rel)

    def _send(self, rel):
        b = Hub.files[rel]
        if Hub.bad_n.get(rel, 0) > 0:
            Hub.bad_n[rel] -= 1
            b = b[:-1] + b"!"
        self.send_response(200)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


class Cdn(Hub):
    def do_GET(self):
        Hub.seen.append(("cdn", self.path, self.headers.get("Authorization")))
        self._send(urllib.parse.unquote(self.path[len("/blob/"):]))


def serve(handler):
    s = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=s.serve_forever, daemon=True).start()
    return s


def pull(files: dict, dest_setup=None, man=None, bad_n=None, missing=(), loop=False, token="tok") -> tuple:
    hub, cdn = serve(Hub), serve(Cdn)
    Hub.cdn_port, Hub.seen, Hub.loop = cdn.server_address[1], [], loop
    Hub.files = {f"t1/{k}": v for k, v in files.items() if k not in missing}
    text = man if man is not None else "".join(f"{hashlib.sha256(v).hexdigest()}  {k}\n" for k, v in sorted(files.items()))
    if text is not False:
        Hub.files["t1/MANIFEST.sha256"] = text.encode()
    Hub.bad_n = {f"t1/{k}": n for k, n in (bad_n or {}).items()}
    try:
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            dest = t / "stage"
            dest.mkdir()
            if dest_setup:
                dest_setup(dest)
            env = {k: v for k, v in os.environ.items() if "proxy" not in k.lower() and k != "HF_TOKEN"}
            env["V2D_XFER_RETRY_SLEEP_S"] = "0"
            if token:
                env["HF_TOKEN"] = token
            r = subprocess.run([sys.executable, str(HERE / "hf_transfer.py"), "pull", "--repo", "u/r", "--tag", "t1", "--dest", str(dest),
                                "--endpoint", f"http://127.0.0.1:{hub.server_address[1]}"], env=env, capture_output=True, text=True, timeout=60)
            got = {p.relative_to(dest).as_posix(): p.read_bytes() for p in dest.rglob("*") if p.is_file()}
            outside = sorted(p.name for p in t.iterdir() if p.name != "stage")
            return r.returncode, r.stdout + r.stderr, got, outside, list(Hub.seen)
    finally:
        hub.shutdown()
        cdn.shutdown()


FILES = {"runs/motion/tier3_hm/whole_body/soma/sequence_id=s 1/robot_name=g1/data.parquet": os.urandom(4096),
         "runs/tier3/root_hm/meta/m.json": b'{"e": 1}', "runs/motion_src/tier3_hm/s1/v2d_meta.json": b'{"tier": 3, "episode": 0}'}


def test_pull_ok() -> None:
    rc, out, got, outside, seen = pull(FILES)
    assert rc == 0 and out.rstrip().endswith("HF_PULL_OK 3") and got == FILES and outside == [], (rc, out, sorted(got))
    assert any(s[0] == "cdn" for s in seen) and all(s[2] is None for s in seen if s[0] == "cdn"), seen
    assert all(s[2] == "Bearer tok" for s in seen if s[0] == "hub"), seen


def test_pull_retries_and_failures() -> None:
    big = next(iter(FILES))
    rc, out, got, _, _ = pull(FILES, bad_n={big: 2})
    assert rc == 0 and got == FILES, out
    for kw, want in ((dict(bad_n={"runs/tier3/root_hm/meta/m.json": 3}), "三次都不对"),
                     (dict(missing=("runs/tier3/root_hm/meta/m.json",)), "HTTPError 404"), (dict(loop=True), "RuntimeError")):
        rc, out, got, outside, _ = pull(FILES, **kw)
        assert rc == 1 and "HF_PULL_FAILED" in out and want in out and outside == [] and not any(k.endswith(".part") for k in got), (kw, out)


def test_pull_refusals() -> None:
    h = "0" * 64
    rc, out, _, _, _ = pull(FILES, man=False)
    assert rc == 1 and "HF_PULL_FAILED 下载 MANIFEST.sha256 失败" in out, out
    rc, out, _, _, _ = pull(FILES, dest_setup=lambda d: (d / "old.txt").write_text("x"))
    assert rc == 1 and "不是空目录" in out, out
    rc, out, _, _, _ = pull(FILES, token="")
    assert rc == 1 and "HF_TOKEN" in out, out
    for line in (f"{h}  ../evil.txt", f"{h}  .", f"{h}  /etc/x", f"{h}  a\\b", f"{h}  ./x", f"{h}  a//b", "0" * 63 + "  x",
                 "A" * 64 + "  x", f"{h}  x\n{h}  x", ""):
        rc, out, got, outside, _ = pull(FILES, man=line + "\n" if line else "")
        assert rc == 1 and "HF_PULL_FAILED" in out and outside == [] and got == {}, (line, out, outside, got)


def test_origin() -> None:
    assert ht._origin("https://huggingface.co/a") == ht._origin("https://huggingface.co/b?x=1")
    assert ht._origin("https://huggingface.co/a") != ht._origin("http://huggingface.co/a")
    assert ht._origin("https://huggingface.co/a") != ht._origin("https://cas-bridge.xethub.hf.co/a")


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_") and callable(f):
            f()
            print("ok", name)
    print("ALL_OK")
