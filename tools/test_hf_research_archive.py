"""tools/hf_research_archive.py 里不连网的部分（plan、scan、stage）的行为测试（暂无契约 ID：运维工具）。只用标准库。

在临时目录里造一台假机器：
  - runs/a/policy_1.safetensors、runs/a/log.txt → 公开；runs/a/run.sh、runs/x.py → 私有；runs/code_tier1/README → 私有
  - runs/a/latest → 符号链接，不上传，记 symlinks.tsv
  - data/challenge/x.parquet → 排除，记 sha256；runs/a/__pycache__/m.pyc、runs/hf_stage/f → 排除
  - runs/stage/ 打包：非代码进 stage.tar（公开），代码进 stage.code.tar（私有）
  - 扫描：已知原值、hf_ 样式、URL 带密码 → 挡下；公开文件出现成员名 → 改私有；safetensors 只扫头
  - 暂存：硬链接，INDEX 里公开那份没有 member_ref.tsv
"""
import json
import os
import struct
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

TOOL = Path(__file__).resolve().parent / "hf_research_archive.py"


def run(*args, stdin=""):
    r = subprocess.run([sys.executable, str(TOOL), *args], input=stdin, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    return r.stdout


def st(p: Path, header: dict, body: bytes) -> None:
    h = json.dumps(header).encode()
    p.write_bytes(struct.pack("<Q", len(h)) + h + body)


def main() -> int:
    with tempfile.TemporaryDirectory() as t:
        root, work = Path(t) / "m", Path(t) / "w"
        f = {
            "runs/a/log.txt": b"step 1 ok\n",
            "runs/a/run.sh": b"echo hi\n",
            "runs/x.py": b"print(1)\n",
            "runs/code_tier1/README": b"code\n",
            "runs/a/__pycache__/m.pyc": b"\0",
            "runs/hf_stage/f": b"x",
            "data/challenge/x.parquet": b"PAR1",
            "runs/stage/s1.json": b"{}",
            "runs/stage/s2.py": b"pass\n",
            "runs/leak1.log": b"token SECRETVALUE123 here\n",
            "runs/leak2.log": b"hf_" + b"A" * 34 + b"\n",
            "runs/leak3.log": b"https_proxy=http://user:pw@10.0.0.1:3128\n",
            "runs/member.log": b"/home/mbr_alice/data\n",
            "runs/stage/__pycache__/s.pyc": b"\0",
            "runs/stage/.env": b"X=1\n",
            "runs/b/.netrc": b"machine api.wandb.ai\n",
            "runs/b/kaggle.json": b"{}",
            "runs/b/Dockerfile": b"FROM x\n",
            "runs/b/k.cu": b"__global__\n",
            "runs/b/.github/w.yml": b"on: push\n",
            "runs/b/.git/config": b"[core]\n",
            "runs/b/.cache/x.bin": b"\0",
            "runs/c/rename.safetensors": b"\x10\0\0\0\0\0\0\0notjsonSECRETVALUE123",
            "top.log": b"single file\n",
            "runs/b/x.csv.abc.kaggle-receipt.json": b"{\"ref\": 1}",
            "runs/b/.env.example": b"HF_TOKEN=\n",
            "runs/b/fixture.py": b"P = 'http://user:pass@host:1'\n",
            "runs/b/leak.py": b"T = 'SECRETVALUE123'\n",
            "runs/b/gh.log": b"token ghp_" + b"a" * 36 + b"\n",
            "runs/b/.pypirc": b"[pypi]\n",
            "runs/b/mine.txt": b"x\n",
        }
        for k, v in f.items():
            (root / k).parent.mkdir(parents=True, exist_ok=True)
            (root / k).write_bytes(v)
        st(root / "runs/a/policy_1.safetensors", {"__metadata__": {"x": "y"}}, b"SECRETVALUE123" + b"\0" * 64)
        os.symlink("policy_1.safetensors", root / "runs/a/latest")
        os.chmod(root / "runs/b/mine.txt", 0o600)
        import io
        for name, members in (("gitpack.tar", {".git/HEAD": b"ref"}), ("codepack.tar", {"a/x.py": b"pass"}),
                              ("datapack.tar", {"a/x.json": b"{}"})):
            with tarfile.open(root / "runs/b" / name, "w") as tf:
                for mn, mb in members.items():
                    ti = tarfile.TarInfo(mn); ti.size = len(mb); tf.addfile(ti, io.BytesIO(mb))
        r = subprocess.run([sys.executable, str(TOOL), "plan", "--machine", "m", "--work", str(work), "--src", f"{root}=repo",
                            "--src", f"{root}/runs=x"], capture_output=True, text=True)
        assert r.returncode != 0 and "嵌套" in r.stderr, "源嵌套要停"
        out = run("plan", "--machine", "m", "--work", str(work), "--src", f"{root}/runs=repo/runs", "--src", f"{root}/data=repo/data",
                  "--src", f"{root}/top.log=repo/top.log", "--pack", f"{root}/runs/stage=repo/runs/stage.tar")
        assert "PLAN_DONE" in out, out
        rows = {r["dest"]: r for r in map(json.loads, (work / "manifest.jsonl").read_text().splitlines())}
        assert rows["m/repo/runs/a/log.txt"]["vis"] == "public"
        assert rows["m/repo/runs/a/policy_1.safetensors"]["vis"] == "public"
        assert rows["m/repo/top.log"]["vis"] == "public", "单个文件的源"
        assert "m/repo/runs/b/gitpack.tar" not in rows and rows["m/repo/runs/b/codepack.tar"]["vis"] == "private"
        assert rows["m/repo/runs/b/datapack.tar"]["vis"] == "public"
        assert rows["m/repo/runs/b/x.csv.abc.kaggle-receipt.json"]["vis"] == "public", "上传回执不按文件名挡"
        assert rows["m/repo/runs/b/.env.example"]["vis"] == "public", ".env.example 是模板"
        for k in ("runs/b/Dockerfile", "runs/b/k.cu", "runs/b/.github/w.yml"):
            assert rows[f"m/repo/{k}"]["vis"] == "private", k
        held0 = (work / "held.tsv").read_text()
        for k in ("runs/stage/.env", "runs/b/.netrc", "runs/b/kaggle.json"):
            assert f"m/repo/{k}\t" in held0 and f"m/repo/{k}" not in rows, k
        assert not any("/.git/" in k or "/.cache/" in k for k in rows)
        for k in ("runs/a/run.sh", "runs/x.py", "runs/code_tier1/README"):
            assert rows[f"m/repo/{k}"]["vis"] == "private", k
        assert rows["m/repo/runs/stage.tar"]["vis"] == "public" and rows["m/repo/runs/stage.code.tar"]["vis"] == "private"
        with tarfile.open(rows["m/repo/runs/stage.tar"]["src"]) as tf:
            assert tf.getnames() == ["s1.json"], tf.getnames()
        assert not any("s1.json" in k or "s2.py" in k for k in rows if not k.endswith(".tar"))
        assert "m/repo/runs/a/latest\tpolicy_1.safetensors" in (work / "symlinks.tsv").read_text()
        ex = (work / "excluded.tsv").read_text()
        assert "data/challenge/x.parquet" in ex and "a/__pycache__" in ex and "stage/__pycache__" in ex and "hf_stage/f" in ex
        assert "runs/b/.git/config" in ex and "runs/b/.cache/x.bin" in ex
        assert [x for x in ex.splitlines() if "x.parquet" in x][0].split("\t")[2], "官方数据要记 sha256"
        r = subprocess.run([sys.executable, str(TOOL), "scan", "--work", str(work)], input="", capture_output=True, text=True)
        assert r.returncode != 0, "没有已知原值要停"
        r = subprocess.run([sys.executable, str(TOOL), "stage", "--work", str(work), "--min-free-gb", "0"], input="SECRETVALUE123\n",
                           capture_output=True, text=True)
        assert r.returncode != 0 and "scan" in r.stderr, "没 scan 不能 stage"
        out = run("scan", "--work", str(work), "--members", "mbr_alice,mbr_bob", stdin="SECRETVALUE123\n\n")
        assert "SCAN_DONE" in out, out
        held = (work / "held.tsv").read_text()
        for k in ("leak1.log\t", "leak2.log\t", "leak3.log\t"):
            assert k in held, (k, held)
        assert "SECRETVALUE123" not in held
        assert "known_0" in held and "hf_token" in held and "url_password" in held
        assert "m/repo/runs/c/rename.safetensors\t" in held, "头不是 JSON 的 safetensors 要全量扫"
        assert "runs/b/.netrc" in held, "plan 按文件名挡下的记录要保留"
        assert "runs/b/fixture.py" not in held, "私有代码里的占位串不挡"
        assert "runs/b/gh.log\t" in held and "github_token" in held
        assert "m/repo/runs/b/.pypirc\t" in held0 and "m/repo/runs/b/mine.txt" in held0 and "owner_only_mode" in held0
        assert any(x.startswith("m/repo/runs/b/leak.py\t") and x.endswith("\tknown_0") for x in held.splitlines()), "私有文件里的已知原值照样挡"
        rows = {r["dest"]: r for r in map(json.loads, (work / "manifest.jsonl").read_text().splitlines())}
        assert "m/repo/runs/a/policy_1.safetensors" in rows, "safetensors 只扫头，数据区里的字节不算"
        assert rows["m/repo/runs/member.log"]["vis"] == "private"
        (work / "stage/public/m/repo/runs/leak1.log").parent.mkdir(parents=True, exist_ok=True)
        (work / "stage/public/m/repo/runs/leak1.log").write_text("stale")  # 旧暂存里的残留要被清掉
        out = run("stage", "--work", str(work), "--min-free-gb", "0", "--members", "mbr_alice,mbr_bob", stdin="SECRETVALUE123\n")
        assert "STAGE_DONE" in out
        pub = work / "stage/public"
        assert os.path.samefile(pub / "m/repo/runs/a/log.txt", root / "runs/a/log.txt")
        assert not (pub / "m/repo/runs/x.py").exists() and (work / "stage/private/m/repo/runs/x.py").exists()
        assert not (pub / "INDEX/m/member_ref.tsv").exists() and (work / "stage/private/INDEX/m/member_ref.tsv").exists()
        assert not (pub / "m/repo/runs/leak1.log").exists()
        run("stage", "--work", str(work), "--min-free-gb", "0", stdin="SECRETVALUE123\n")  # 重跑不报错
        for p in (pub / "INDEX/m").iterdir():
            assert b"mbr_alice" not in p.read_bytes(), p
    print("test_hf_research_archive ALL_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
