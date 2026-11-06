"""tier3_relay.sh 的行为测试（暂无契约 ID：运行规范待成文）。只用标准库，在 Mac 上跑。

临时目录里建来源与目标两个“仓库根”，PATH 前面放一个假的 ssh（跳过选项与主机名，把最后一个参数交给本机 bash 执行），
V2D_RELAY_HF_REMOTE 指向一个假的 hf_remote（V2D_ENV=ml40 时在目标根、否则在来源根执行命令），两个根的 runs/tools/ 下放假的
hf_transfer.py（push 把目录复制进本地“仓库”目录，pull 再复制到暂存目录；按环境变量出错、改文件、建目标）。
来源有 2 集：parquet、v2d_meta.json、reconstructed_stage 支撑面各 2 份，root 的 support_surface 有同名同内容的支撑面，共 12 个文件。
每一例都核对：来源没变，目标上 runs/motion/tier2 下的哨兵文件没变。
  1. 正常：目标三个目录与来源逐字节相同，暂存目录删掉，最后一行 RELAY_OK；
  2. 目标已有其中一个目录、暂存目录残留、来源缺目录：停在第 1 步，什么都不传，残留的暂存目录不动；
  3. 上传失败、下载失败、来源里有符号链接（push 拒绝）、下载报的文件数不对、hf_remote 连的不是预期的机器与路径：停在第 2 步，删暂存目录；
  4. 下载后文件被改、来源清单失败、文件数给错：停在第 3 步，删暂存目录，目标不存在；
  5. 集数给 3、有一份档位是 2、两集集号相同、sequence_id 与目录不符、支撑面与 root 的不同、有 _g1_support：停在第 4 步，删暂存目录；
  6. 下载之后有人建了同名目标目录：停在第 5 步，不挪、不删，目标目录里没有被挪进去的东西。
用法：python3 stages/recovery/test_tier3_relay.py
"""
import json
import os
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAKE_SSH = r'''#!/usr/bin/env bash
cmd="${@: -1}"
if [ "${FAKE_LISTFAIL:-0}" = 1 ] && [[ $cmd == *sha256sum* ]] && [[ $cmd != *partial* ]]; then exit 1; fi
exec bash -c "$cmd"
'''
FAKE_HFR = r'''#!/usr/bin/env bash
if [ "${V2D_ENV:-}" = ml40 ]; then echo "→ ml40:${FAKE_DST}${FAKE_WRONGHOST:-}  (V2D_ENV=ml40)" >&2; cd "$FAKE_DST"
else echo "→ mf4090:$FAKE_SRC" >&2; cd "$FAKE_SRC"; fi
bash -c "$1"
'''
FAKE_XFER = r'''import os, shutil, sys
from pathlib import Path
a = sys.argv[1:]
hub, tag = Path(os.environ["FAKE_HUB"]), a[a.index("--tag") + 1]
if a[0] == "push":
    if os.environ.get("FAKE_PUSHFAIL"):
        sys.exit("HF_PUSH_FAILED 假的")
    files = [p for d in a[a.index("--dirs") + 1:] for p in Path(d).rglob("*") if not p.is_dir()]
    if any(p.is_symlink() for p in files):
        sys.exit("有非普通文件")
    for p in files:
        (hub / tag / p).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, hub / tag / p)
    print("HF_PUSH_OK %d" % len(files))
else:
    if os.environ.get("FAKE_PULLFAIL"):
        print("HF_PULL_FAILED 假的"); sys.exit(1)
    dest = Path(a[a.index("--dest") + 1])
    shutil.copytree(hub / tag / "runs", dest / "runs")
    if os.environ.get("FAKE_CORRUPT"):
        f = next(dest.rglob("data.parquet")); f.write_bytes(f.read_bytes() + b"x")
    if os.environ.get("FAKE_MKTARGET"):
        Path("runs/motion/tier3_hm").mkdir(parents=True)
    n = sum(1 for p in (dest / "runs").rglob("*") if p.is_file())
    print("HF_PULL_OK %s" % os.environ.get("FAKE_PULLCOUNT", n))
'''
D = ("runs/tier3/root_hm", "runs/motion_src/tier3_hm", "runs/motion/tier3_hm")
PART = "runs/.relay_tier3_hm.partial"


def build_src(src: Path, tiers=(3, 3), eps=(0, 1), seq_in_meta=("seqA", "seqB"), root_sup_differs=False, g1=False, link=False) -> None:
    st = src / D[2] / "whole_body/reconstructed_stage"
    st.mkdir(parents=True)
    (src / D[0] / "support_surface").mkdir(parents=True)
    (src / D[0] / "data").mkdir(parents=True)
    for i, seq in enumerate(("seqA", "seqB")):
        (src / D[0] / "data" / f"ep{i}.parquet").write_bytes(os.urandom(64))
        m = src / D[1] / seq
        m.mkdir(parents=True)
        (m / "v2d_meta.json").write_text(json.dumps({"tier": tiers[i], "episode": eps[i], "sequence_id": seq_in_meta[i]}))
        (m / "poses.npy").write_bytes(os.urandom(128))
        q = src / D[2] / "whole_body/soma" / f"sequence_id={seq}/robot_name=g1"
        q.mkdir(parents=True)
        (q / "data.parquet").write_bytes(os.urandom(256))
        (st / f"{seq}_support.usda").write_text(f"#usda 1.0 {seq}\n")
        (src / D[0] / "support_surface" / f"{seq}_support.usda").write_text(f"#usda 1.0 {seq}{' changed' if root_sup_differs and i else ''}\n")
    if g1:
        (st / "seqA_g1_support.usda").write_text("#usda 1.0 g1\n")
    if link:
        (src / D[0] / "data" / "alias.parquet").symlink_to("ep0.parquet")


def files(root: Path) -> dict:
    return {str(p.relative_to(root)): p.read_bytes() for d in D if (root / d).exists() for p in sorted((root / d).rglob("*")) if p.is_file()}


def run(n: str = "2", nf: str = "12", dst_has=None, src_missing: bool = False, src_kw=None, **kw) -> tuple:
    with tempfile.TemporaryDirectory() as t:
        t = Path(t)
        src, dst, bin_, hub = t / "src", t / "dst", t / "bin", t / "hub"
        for p in (src / "runs/tools", dst / "runs/motion_src", dst / "runs/motion/tier2", dst / "runs/tools", bin_, hub):
            p.mkdir(parents=True)
        build_src(src, **(src_kw or {}))
        if src_missing:
            subprocess.run(["rm", "-rf", str(src / D[0])], check=True)
        canary = dst / "runs/motion/tier2/x.parquet"
        canary.write_bytes(b"tier2 data")
        if dst_has:
            (dst / dst_has).mkdir(parents=True)
            (dst / dst_has / "keep.txt").write_text("someone else")
        before = files(src)
        for path, text in ((bin_ / "ssh", FAKE_SSH), (bin_ / "hf_remote", FAKE_HFR), (src / "runs/tools/hf_transfer.py", FAKE_XFER),
                           (dst / "runs/tools/hf_transfer.py", FAKE_XFER)):
            path.write_text(text)
            path.chmod(0o755)
        env = dict(os.environ, PATH=f"{bin_}:{os.environ['PATH']}", V2D_RELAY_SRC_ROOT=str(src), V2D_RELAY_DST_ROOT=str(dst),
                   FAKE_DST=str(dst), FAKE_SRC=str(src), FAKE_HUB=str(hub), V2D_RELAY_HF_REMOTE=str(bin_ / "hf_remote"), **kw)
        r = subprocess.run(["bash", str(HERE / "tier3_relay.sh"), "hm", n, nf, "t3hm_test"], env=env, capture_output=True, text=True, errors="replace",
                           timeout=120)
        assert files(src) == before and canary.read_bytes() == b"tier2 data", "来源或哨兵文件变了"
        seen = {"keep": (dst / PART / "keep.txt").exists(), "d3": sorted(x.name for x in (dst / D[2]).iterdir()) if (dst / D[2]).is_dir() else None}
        return r.returncode, r.stdout + r.stderr, before, files(dst), (dst / PART).exists(), seen


def last(out: str) -> str:
    return out.rstrip().splitlines()[-1]


def test_ok() -> None:
    rc, out, fs, fd, partial, _ = run()
    assert rc == 0 and last(out).endswith("RELAY_OK"), out
    assert fs == fd and len(fd) == 12 and not partial, (sorted(fs), sorted(fd), partial)
    assert "parquet 2、v2d_meta.json 2、支撑面 2（_g1_ 0）；档位 3；集号 2 个（0,1）；sequence_id 与目录不符 0" in out, out
    assert "同名比对：2 份，不同 0 份" in out and "挪到位后清单与来源逐行相同（12 个文件）" in out, out


def test_step1_refusals() -> None:
    for kw, want in ((dict(dst_has="runs/motion/tier3_hm"), "1（目标已有"), (dict(dst_has=PART), "1（目标已有"),
                     (dict(src_missing=True), "1（来源缺目录）")):
        rc, out, _, fd, partial, seen = run(**kw)
        assert rc == 1 and want in last(out), (kw, out)
        assert set(fd) <= {"runs/motion/tier3_hm/keep.txt"}, (kw, sorted(fd))
        if kw.get("dst_has") == PART:
            assert seen["keep"], "残留的暂存目录被动了"


def test_step2_failures() -> None:
    for kw, want in ((dict(FAKE_PUSHFAIL="1"), "2（上传到"), (dict(FAKE_PULLFAIL="1"), "2（从"), (dict(src_kw=dict(link=True)), "2（上传到"),
                     (dict(FAKE_PULLCOUNT="11"), "2（从"), (dict(FAKE_WRONGHOST="/x"), "2（下载连的不是")):
        rc, out, _, fd, partial, _ = run(**kw)
        assert rc == 1 and want in last(out) and not partial and fd == {}, (kw, out)


def test_step3_failures() -> None:
    for kw, want in ((dict(FAKE_CORRUPT="1"), "3（两端清单不一致"), (dict(FAKE_LISTFAIL="1"), "3（来源清单）"),
                     (dict(nf="13"), "2（上传到")):
        rc, out, _, fd, partial, _ = run(**kw)
        assert rc == 1 and want in last(out) and not partial and fd == {}, (kw, out)


def test_step4_failures() -> None:
    cases = (dict(n="3"), dict(src_kw=dict(tiers=(3, 2))), dict(src_kw=dict(eps=(0, 0))), dict(src_kw=dict(seq_in_meta=("seqA", "seqX"))),
             dict(src_kw=dict(root_sup_differs=True)), dict(src_kw=dict(g1=True)))
    for kw in cases:
        kw.setdefault("nf", "13" if kw.get("src_kw", {}).get("g1") else "12")
        rc, out, _, fd, partial, _ = run(**kw)
        assert rc == 1 and "RELAY_FAILED 4" in last(out) and not partial and fd == {}, (kw, out)


def test_target_appears_before_move() -> None:
    rc, out, _, fd, partial, seen = run(FAKE_MKTARGET="1")
    assert rc == 1 and "RELAY_FAILED 5" in last(out), out
    assert partial and fd == {} and seen["d3"] == [], (partial, sorted(fd), seen)


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_") and callable(f):
            f()
            print("ok", name)
    print("ALL_OK")
