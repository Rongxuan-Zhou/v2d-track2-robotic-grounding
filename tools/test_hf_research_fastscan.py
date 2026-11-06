"""tools/hf_research_archive.py 的 scan --fast（GNU grep 预筛）与慢扫结果相同（暂无契约 ID：运维工具）。只在有 GNU grep 的机器上有意义。

每种通用样式造一个正例和一个只差一点的反例，加已知原值、成员名、safetensors（头里 / 数据区里）、二进制里的命中、
跨 16 MB 块边界的命中；同一份清单分别用慢扫和快扫跑，挡下表、成员名表、扫后的清单必须逐字相同。
"""
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

TOOL = Path(__file__).resolve().parent / "hf_research_archive.py"
POS = {
    "hf_token": b"x hf_" + b"Ab1" * 11,
    "private_key": b"-----BEGIN OPENSSH PRIVATE KEY-----",
    "aws_key": b"AKIA" + b"ABCD1234EFGH5678",
    "url_password": b"http://usr:pw9@10.1.2.3:8080",
    "wandb_key": b"WANDB_API_KEY = '" + b"a1" * 20 + b"'",
    "netrc_password": b"login x password " + b"0f" * 20 + b"\n",
    "api_key_assign": b'"apiKey": "' + b"Z" * 40 + b'"',
    "bearer": b"Authorization: Bearer " + b"t" * 30,
    "kaggle_token": b"KGAT_" + b"ab" * 15,
    "github_token": b"ghp_" + b"q" * 36,
    "sk_key": b" sk-" + b"k" * 40,
    "slack_token": b"xoxb-" + b"1234567890ab",
    "json_private_key": b'{"private_key" : "x"}',
}
NEG = {
    "hf_token": b"hf_" + b"A" * 29,
    "private_key": b"-----BEGIN PUBLIC KEY-----",
    "aws_key": b"AKIA" + b"abcd1234efgh5678",
    "url_password": b"http://host:8080/path@x",
    "wandb_key": b"WANDB_API_KEY = 'g1'",
    "netrc_password": b"password " + b"0f" * 19,
    "api_key_assign": b"api_key = short",
    "bearer": b"Authorization: Basic " + b"t" * 30,
    "kaggle_token": b"KGAT_" + b"zz" * 15,
    "github_token": b"ghx_" + b"q" * 36,
    "sk_key": b"ask-" + b"k" * 40,
    "slack_token": b"xoxz-1234567890ab",
    "json_private_key": b'{"private": "x"}',
}


def st(p: Path, header: bytes, body: bytes) -> None:
    p.write_bytes(struct.pack("<Q", len(header)) + header + body)


def run(*args, stdin=""):
    r = subprocess.run([sys.executable, str(TOOL), *args], input=stdin, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr + r.stdout
    return r.stdout


def main() -> int:
    if "GNU grep" not in subprocess.run(["grep", "--version"], capture_output=True, text=True).stdout:
        print("没有 GNU grep，跳过")
        return 0
    with tempfile.TemporaryDirectory() as t:
        root = Path(t) / "m" / "runs"
        root.mkdir(parents=True)
        for k, v in POS.items():
            (root / f"pos_{k}.log").write_bytes(b"pre\n" + v + b"\npost\n")
        for k, v in NEG.items():
            (root / f"neg_{k}.log").write_bytes(b"pre\n" + v + b"\npost\n")
        (root / "known.log").write_bytes(b"value SECRETVALUE123 end")
        (root / "known_bin.dat").write_bytes(os.urandom(4096) + b"SECRETVALUE123" + os.urandom(4096))
        (root / "member.log").write_bytes(b"/home/mbr_alice/x")
        (root / "clean.log").write_bytes(b"nothing here\n")
        big = bytearray(os.urandom((1 << 24) - 5)) + b"SECRETVALUE123" + os.urandom(100)
        (root / "boundary.bin").write_bytes(bytes(big.replace(b"\n", b" ")))
        st(root / "head.safetensors", b'{"__metadata__": {"k": "SECRETVALUE123"}}', b"\0" * 64)
        st(root / "body.safetensors", b'{"__metadata__": {}}', b"SECRETVALUE123")
        works = {}
        for mode in ("slow", "fast"):
            w = Path(t) / f"w_{mode}"
            run("plan", "--machine", "m", "--work", str(w), "--src", f"{root}=repo/runs")
            args = ["scan", "--work", str(w), "--members", "mbr_alice"] + (["--fast"] if mode == "fast" else [])
            run(*args, stdin="SECRETVALUE123\n")
            works[mode] = w
        for n in ("held.tsv", "member_ref.tsv", "manifest.jsonl"):
            a = (works["slow"] / n).read_text()
            b = (works["fast"] / n).read_text()
            assert a == b, (n, a, b)
        held = (works["fast"] / "held.tsv").read_text()
        for k in POS:
            assert f"pos_{k}.log\t" in held and k in held, k
        for k in NEG:
            assert f"neg_{k}.log\t" not in held, k
        for k in ("known.log", "known_bin.dat", "boundary.bin", "head.safetensors"):
            assert f"{k}\t" in held, k
        assert "body.safetensors" not in held and "clean.log" not in held
        assert "member.log" in (works["fast"] / "member_ref.tsv").read_text()
    print("test_hf_research_fastscan ALL_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
