"""tier3_stage_hf.sh 的行为测试（暂无契约 ID：运行规范待成文；计划 agent-doc/plan-t3-kfix-2026-10-16.md “方案 4 的执行细则”第 4 步）。
只用标准库，在 Mac 上跑。

临时目录里建三个根：mac（脚本与哈希表）、mf（mf4090 的仓库副本）、ml（ml40 的仓库副本），另一个目录当 HF 仓库。PATH 前面放假的 ssh（跳过选项与主机名，
把最后一个参数交给本机 bash，stdin 照传）与假的 docker（取 -w、-v <宿主>:/repo 与镜像名之后的命令，/repo 换成宿主路径，python 换成 python3）；
V2D_STAGEHF_HF_REMOTE 指向假的 hf_remote（V2D_ENV=ml40 时在 ml 根、否则在 mf 根执行），V2D_STAGEHF_PACK 指向假的打包脚本（打出 DEPLOYED、
1 个 parquet、1 份元数据的 tar）；两个根的 tools/hf_transfer.py 是假的（push 复制进 HF 目录并打 HF_PUSH_OK，pull 复制到 --dest 并打 HF_PULL_OK），
mf 根的 stages/recovery/tier3_root.py 是假的（按哈希表逐个核 sha256，全对打 ROOT_OK）。
  1. 正常：最后一行 ROOT_OK；mf 上 incoming/<标签>/ 里三个文件与打包的逐字节相同，.pull 与 .partial 都不在；ml 上 runs/hf_src/<标签>/pack 在、pack.tar 删了；
  2. mf 上的哈希表与 Mac 的不同：停在第 1 步，ml 上没建 runs/hf_src/<标签>；
  3. ml 上 runs/hf_src/<标签> 已存在：停在第 1 步，原有内容不动；
  4. 拉取失败：退出码非 0，mf 上 <标签>.pull 留着，<标签> 不存在；
  5. 拉下来的 .pull 里不只 pack/：停在搬目录那步，.pull 留着；
  6. 拉下来的文件被改了：假的 tier3_root.py 报 sha256 不符，退出码非 0；
  7. V2D_T3_SUPPORT=delivery（计划 agent-doc/plan-t3-gb3-switch-2026-10-17.md 做法 1b）：打包带上支撑面，tier3_root.py 收到 --support-from-incoming，
     incoming 里的支撑面与打包的逐字节相同；
  8. V2D_T3_SUPPORT=delivery 与 V2D_T3_SUPPORT_DIR 同时设：退出码 2，两台上什么也没建。
用法：python3 stages/recovery/test_tier3_stage_hf.py；最后一行 ALL_OK。
"""
import hashlib
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TAG = "t3kfix_test"
FAKE_SSH = r'''#!/usr/bin/env bash
cmd="${@: -1}"
exec bash -c "$cmd"
'''
FAKE_DOCKER = r'''#!/usr/bin/env bash
args=("$@"); wd=""; host=""; img=-1
for i in "${!args[@]}"; do
  a=${args[$i]}
  [ "$a" = "-w" ] && wd=${args[$((i+1))]}
  [ "$a" = "-v" ] && [[ ${args[$((i+1))]} == *:/repo ]] && host=${args[$((i+1))]%:/repo}
  [ "$a" = "v2d_cari4d:latest" ] && img=$i
done
cmd=("${args[@]:$((img+1))}")
for i in "${!cmd[@]}"; do cmd[$i]=${cmd[$i]//\/repo/$host}; [ "${cmd[$i]}" = python ] && cmd[$i]=python3; done
cd "${wd//\/repo/$host}" && exec "${cmd[@]}"
'''
FAKE_HFR = r'''#!/usr/bin/env bash
if [ "${V2D_ENV:-}" = ml40 ]; then cd "$FAKE_ML"; else cd "$FAKE_MF"; fi
bash -c "$1"
'''
FAKE_PACK = r'''set -e
d=$(mktemp -d)
printf 'deadbeef00000000000000000000000000000000 dirty=0\n' > "$d/DEPLOYED"
mkdir -p "$d/data/chunk-000" "$d/meta"
printf 'parquet-bytes-ep%s' "$1" > "$d/data/chunk-000/episode_000000.parquet"
printf '{"episode_index": 0}' > "$d/meta/episode_000000.metadata.json"
if [ "${V2D_T3_SUPPORT:-}" = delivery ]; then mkdir -p "$d/support_surface"; printf 'usda-bytes' > "$d/support_surface/seq_support.usda"; fi
tar -cf - -C "$d" DEPLOYED data meta $([ "${V2D_T3_SUPPORT:-}" = delivery ] && echo support_surface)
rm -rf "$d"
'''
FAKE_XFER = r'''import os, shutil, sys
from pathlib import Path
a = sys.argv[1:]
hub, tag = Path(os.environ["FAKE_HUB"]), a[a.index("--tag") + 1]
if a[0] == "push":
    files = [p for d in a[a.index("--dirs") + 1:] for p in Path(d).rglob("*") if p.is_file()]
    for p in files:
        (hub / tag / p).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, hub / tag / p)
    print("HF_PUSH_OK %d" % len(files))
else:
    if os.environ.get("FAKE_PULLFAIL"):
        print("HF_PULL_FAILED 假的"); sys.exit(1)
    dest = Path(a[a.index("--dest") + 1])
    shutil.copytree(hub / tag / "pack", dest / "pack")
    if os.environ.get("FAKE_EXTRA"):
        (dest / "extra.txt").write_text("x")
    if os.environ.get("FAKE_CORRUPT"):
        f = next(dest.rglob("*.parquet")); f.write_bytes(f.read_bytes() + b"x")
    print("HF_PULL_OK %d" % sum(1 for p in dest.rglob("*") if p.is_file()))
'''
FAKE_ROOT = r'''import hashlib, sys
from pathlib import Path
inc = Path(sys.argv[1]); exp = sys.argv[sys.argv.index("--expect") + 1]
print("ARGS " + " ".join(sys.argv[2:]))
for line in Path(exp).read_text().splitlines():
    rel, _, h = line.split("\t")
    if hashlib.sha256((inc / rel).read_bytes()).hexdigest() != h:
        print("ROOT_FAIL", rel); sys.exit(1)
print("ROOT_OK")
'''


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def setup(t: Path) -> dict:
    mac, mf, ml, hub, binp = t / "mac", t / "mf", t / "ml", t / "hub", t / "bin"
    for d in (mac / "stages/recovery/expect", mf / "stages/recovery/expect", mf / "tools", ml / "tools", hub, binp):
        d.mkdir(parents=True)
    shutil.copy2(HERE / "tier3_stage_hf.sh", mac / "stages/recovery/tier3_stage_hf.sh")
    rows = [("data/chunk-000/episode_000000.parquet", sha(b"parquet-bytes-ep0")),
            ("meta/episode_000000.metadata.json", sha(b'{"episode_index": 0}'))]
    exp = "".join(f"{r}\tsha256\t{h}\n" for r, h in rows)
    for root in (mac, mf):
        (root / f"stages/recovery/expect/{TAG}.tsv").write_text(exp)
    (mf / "stages/recovery/tier3_root.py").write_text(FAKE_ROOT)
    for root in (mf, ml):
        (root / "tools/hf_transfer.py").write_text(FAKE_XFER)
    for name, body in (("ssh", FAKE_SSH), ("docker", FAKE_DOCKER), ("hfr", FAKE_HFR)):
        (binp / name).write_text(body)
        (binp / name).chmod(0o755)
    (t / "pack.sh").write_text(FAKE_PACK)
    env = dict(os.environ, PATH=f"{binp}:{os.environ['PATH']}", FAKE_MF=str(mf), FAKE_ML=str(ml), FAKE_HUB=str(hub),
               V2D_STAGEHF_R=str(mf), V2D_STAGEHF_R40=str(ml), V2D_STAGEHF_HF_REMOTE=str(binp / "hfr"), V2D_STAGEHF_PACK=str(t / "pack.sh"),
               V2D_T3_SUBDIR="tier3_kfix_F", V2D_T3_SET="kfix")
    return {"mac": mac, "mf": mf, "ml": ml, "env": env}


def add_support_row(c: dict) -> None:
    for root in (c["mac"], c["mf"]):
        p = root / f"stages/recovery/expect/{TAG}.tsv"
        p.write_text(p.read_text() + f"support_surface/seq_support.usda\tsha256\t{sha(b'usda-bytes')}\n")


def run(c: dict, **extra) -> subprocess.CompletedProcess:
    env = dict(c["env"], **extra)
    return subprocess.run(["bash", str(c["mac"] / "stages/recovery/tier3_stage_hf.sh"), TAG, "0"], capture_output=True, text=True, env=env)


def test_normal() -> None:
    with tempfile.TemporaryDirectory() as t:
        c = setup(Path(t))
        r = run(c)
        assert r.returncode == 0 and r.stdout.strip().splitlines()[-1] == "ROOT_OK", r.stdout + r.stderr
        inc = c["mf"] / f"runs/tier3/incoming/{TAG}"
        assert (inc / "data/chunk-000/episode_000000.parquet").read_bytes() == b"parquet-bytes-ep0"
        assert (inc / "DEPLOYED").exists() and (inc / "meta/episode_000000.metadata.json").exists()
        assert not (c["mf"] / f"runs/tier3/incoming/{TAG}.pull").exists() and not (c["mf"] / f"runs/tier3/incoming/{TAG}.partial").exists()
        hs = c["ml"] / f"runs/hf_src/{TAG}"
        assert (hs / "pack/DEPLOYED").exists() and not (hs / "pack.tar").exists()
    print("  1. 正常：拼档目录的核对通过，incoming 与打包的逐字节相同，.pull、.partial 不在，ml 上 pack.tar 已删")


def test_refusals() -> None:
    with tempfile.TemporaryDirectory() as t:
        c = setup(Path(t))
        (c["mf"] / f"stages/recovery/expect/{TAG}.tsv").write_text("different\n")
        r = run(c)
        assert r.returncode != 0 and not (c["ml"] / f"runs/hf_src/{TAG}").exists(), r.stdout + r.stderr
    print("  2. mf 上的哈希表与 Mac 不同：停在第 1 步，ml 上什么也没建")
    with tempfile.TemporaryDirectory() as t:
        c = setup(Path(t))
        old = c["ml"] / f"runs/hf_src/{TAG}"
        old.mkdir(parents=True)
        (old / "keep.txt").write_text("keep")
        r = run(c)
        assert r.returncode != 0 and (old / "keep.txt").read_text() == "keep" and len(list(old.iterdir())) == 1, r.stdout + r.stderr
    print("  3. ml 上 runs/hf_src/<标签> 已存在：停在第 1 步，原有内容不动")
    with tempfile.TemporaryDirectory() as t:
        c = setup(Path(t))
        r = run(c, FAKE_PULLFAIL="1")
        assert r.returncode != 0 and (c["mf"] / f"runs/tier3/incoming/{TAG}.pull").exists()
        assert not (c["mf"] / f"runs/tier3/incoming/{TAG}").exists(), r.stdout + r.stderr
    print("  4. 拉取失败：退出码非 0，.pull 留着，incoming/<标签> 不存在")
    with tempfile.TemporaryDirectory() as t:
        c = setup(Path(t))
        r = run(c, FAKE_EXTRA="1")
        assert r.returncode != 0 and (c["mf"] / f"runs/tier3/incoming/{TAG}.pull/extra.txt").exists()
        assert not (c["mf"] / f"runs/tier3/incoming/{TAG}").exists(), r.stdout + r.stderr
    print("  5. .pull 里不只 pack/：停在搬目录那步，.pull 留着")
    with tempfile.TemporaryDirectory() as t:
        c = setup(Path(t))
        r = run(c, FAKE_CORRUPT="1")
        assert r.returncode != 0 and "ROOT_FAIL" in r.stdout, r.stdout + r.stderr
    print("  6. 拉下来的文件被改了：拼档目录的核对报 sha256 不符，退出码非 0")


def test_delivery_support() -> None:
    with tempfile.TemporaryDirectory() as t:
        c = setup(Path(t))
        add_support_row(c)
        r = run(c, V2D_T3_SUPPORT="delivery")
        lines = r.stdout.strip().splitlines()
        assert r.returncode == 0 and lines[-1] == "ROOT_OK", r.stdout + r.stderr
        assert any(x.startswith("ARGS ") and "--support-from-incoming" in x and "--support-dir" not in x for x in lines), r.stdout
        inc = c["mf"] / f"runs/tier3/incoming/{TAG}"
        assert (inc / "support_surface/seq_support.usda").read_bytes() == b"usda-bytes"
    print("  7. V2D_T3_SUPPORT=delivery：支撑面随包转来，tier3_root.py 收到 --support-from-incoming，核对通过")
    with tempfile.TemporaryDirectory() as t:
        c = setup(Path(t))
        r = run(c, V2D_T3_SUPPORT="delivery", V2D_T3_SUPPORT_DIR="runs/tier3/root_hm")
        assert r.returncode == 2 and not (c["ml"] / f"runs/hf_src/{TAG}").exists(), r.stdout + r.stderr
        assert not (c["mf"] / "runs/tier3/incoming").exists(), r.stdout + r.stderr
    print("  8. delivery 与 V2D_T3_SUPPORT_DIR 同时设：退出码 2，两台上什么也没建")


if __name__ == "__main__":
    test_normal()
    test_refusals()
    test_delivery_support()
    print("ALL_OK")
