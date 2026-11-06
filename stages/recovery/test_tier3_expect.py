"""tier3_expect.py 的测试（守计划 agent-doc/plan-t3-pilot-2026-10-09.md 第 3 节、审查 M1；暂无契约 ID）。只用标准库。
  1. 从两份假记录取出 parquet / 元数据的 sha256 与网格 / 地面 / 支撑面的 md5，别的行（标题、元数据 JSON、“输入”行）不取
  2. 同一路径两个不同的哈希：报错
  3. 一份记录一行也取不到：报错
  4. --support-from-remote（计划 agent-doc/plan-t3-hm-p2-2026-10-12.md）：支撑面的行取远端的 sha256、Tier 2 记录里的支撑面行丢掉；
     support_usda.json 那一行（以 “## ” 开头）与子目录里的文件不取；远端支撑面行数与 parquet 行数不等就报错；不加选项时远端的支撑面行不取
  5. --support-record <记录>（计划 agent-doc/plan-t3dyn-pilot-2026-10-14.md）：支撑面的行取另一份记录里的 sha256，只取主记录元数据 JSON 的
     support_surface 指到的那份（另一份记录里多出来的不取），Tier 2 记录里的支撑面行丢掉；另一份记录里缺这份、两个选项一起给、
     元数据没有 support_surface：都报错
  （两份假记录的 Tier 2 那份头一行带坏的 UTF-8 字节，同 scripts/dev/record 记下的含中文参数的 command 行）
用法：python3 stages/recovery/test_tier3_expect.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
A64, B64 = "a" * 64, "b" * 64
REMOTE = f"""# command  : ssh ml40 bash -s -- 8
== host 2026-10-10 04:20:39 CST
## episode 8
可读 962684 字节 2026-10-09 14:36:25.401319 sha256 {A64} data/chunk-000/episode_000008.parquet
可读 629 字节 2026-10-09 14:36:25.401319313 sha256 {B64} meta/episode_000008.metadata.json
{{"episode_index": 8, "mesh": "mesh/chair/chair.glb"}}
输入 mesh 666b6366a803130fc20050249ad67fa2 mesh/chair/chair.glb
"""
TIER2 = """== VM 2026-10-10 04:20:25 CST 来历 d8b309e9d52c
666b6366a803130fc20050249ad67fa2 mesh/chair/chair.glb
9626590db5bd1dc3a8188ee4d0d03f11 ground_plane/s8.json
e7b1026ef52ee91c15c1fcd6d216c653 support_surface/s8_support.usda
## 支撑面形状
s8	empty	no	e7b1026ef52ee91c15c1fcd6d216c653
"""


def run(remote: str, tier2: str, *opts: str):
    with tempfile.TemporaryDirectory() as t:
        r, s = Path(t) / "r.txt", Path(t) / "s.txt"
        r.write_text(remote, encoding="utf-8")
        s.write_bytes(b"# command  : bash -c \xe6\x94\\225\n" + tier2.encode("utf-8"))   # 同真记录：头一行有坏的 UTF-8
        return subprocess.run([sys.executable, str(HERE / "tier3_expect.py"), str(r), str(s), *opts], capture_output=True, text=True)


def run_rec(remote: str, tier2: str, rec: str, *opts: str):
    """同 run，另写一份支撑面记录，经 --support-record 传进去。"""
    with tempfile.TemporaryDirectory() as t:
        r, s, q = Path(t) / "r.txt", Path(t) / "s.txt", Path(t) / "q.txt"
        r.write_text(remote, encoding="utf-8")
        s.write_bytes(b"# command  : bash -c \xe6\x94\\225\n" + tier2.encode("utf-8"))
        q.write_text(rec, encoding="utf-8")
        return subprocess.run([sys.executable, str(HERE / "tier3_expect.py"), str(r), str(s), "--support-record", str(q), *opts],
                              capture_output=True, text=True)


C64, D64 = "c" * 64, "d" * 64
SUP_LINES = (f"可读 441 字节 2026-10-13 05:08:32.1 sha256 {C64} support_surface/s8_support.usda\n"
             f"## support_surface/support_usda.json sha256 {D64}\n"
             f"可读 9 字节 2026-10-13 05:08:32.1 sha256 {D64} support_surface/sub/x.usda\n")


def test_collects_both():
    p = run(REMOTE, TIER2)
    assert p.returncode == 0, p.stderr
    assert p.stdout.splitlines() == [
        f"data/chunk-000/episode_000008.parquet\tsha256\t{A64}",
        "ground_plane/s8.json\tmd5\t9626590db5bd1dc3a8188ee4d0d03f11",
        "mesh/chair/chair.glb\tmd5\t666b6366a803130fc20050249ad67fa2",
        f"meta/episode_000008.metadata.json\tsha256\t{B64}",
        "support_surface/s8_support.usda\tmd5\te7b1026ef52ee91c15c1fcd6d216c653",
    ], p.stdout


def test_conflict_refuses():
    p = run(REMOTE, TIER2 + "00000000000000000000000000000000 mesh/chair/chair.glb\n")
    assert p.returncode != 0 and "哈希不同" in p.stderr + p.stdout, p.stdout + p.stderr


def test_empty_record_refuses():
    p = run(REMOTE, "== 空\n")
    assert p.returncode != 0 and "一行 md5 也没取到" in p.stderr + p.stdout, p.stdout + p.stderr


def test_support_from_remote():
    p = run(REMOTE + SUP_LINES, TIER2, "--support-from-remote")
    assert p.returncode == 0, p.stderr
    assert p.stdout.splitlines() == [
        f"data/chunk-000/episode_000008.parquet\tsha256\t{A64}",
        "ground_plane/s8.json\tmd5\t9626590db5bd1dc3a8188ee4d0d03f11",
        "mesh/chair/chair.glb\tmd5\t666b6366a803130fc20050249ad67fa2",
        f"meta/episode_000008.metadata.json\tsha256\t{B64}",
        f"support_surface/s8_support.usda\tsha256\t{C64}",
    ], p.stdout
    q = run(REMOTE + SUP_LINES, TIER2)
    assert q.returncode == 0 and q.stdout == run(REMOTE, TIER2).stdout, q.stdout   # 不加选项：远端的支撑面行不取，与以前相同


def test_support_count_refuses():
    p = run(REMOTE, TIER2, "--support-from-remote")
    assert p.returncode != 0 and "支撑面 0 行、parquet 1 行" in p.stderr + p.stdout, p.stdout + p.stderr


DYN = REMOTE.replace('{"episode_index": 8, "mesh": "mesh/chair/chair.glb"}',
                     '{"episode_index": 8, "mesh": "mesh/chair/chair.glb", "support_surface": "support_surface/s8_support.usda"}')
HM_SUP = (f"可读 441 字节 2026-10-13 05:08:32.1 sha256 {C64} support_surface/s8_support.usda\n"
          f"可读 124 字节 2026-10-13 05:08:32.1 sha256 {D64} support_surface/s9_support.usda\n"
          f"## support_surface/support_usda.json sha256 {D64}\n")


def test_support_record():
    p = run_rec(DYN, TIER2, HM_SUP)
    assert p.returncode == 0, p.stderr
    assert p.stdout.splitlines() == [
        f"data/chunk-000/episode_000008.parquet\tsha256\t{A64}",
        "ground_plane/s8.json\tmd5\t9626590db5bd1dc3a8188ee4d0d03f11",
        "mesh/chair/chair.glb\tmd5\t666b6366a803130fc20050249ad67fa2",
        f"meta/episode_000008.metadata.json\tsha256\t{B64}",
        f"support_surface/s8_support.usda\tsha256\t{C64}",
    ], p.stdout


def test_support_record_refuses():
    cases = (
        (run_rec(DYN, TIER2, HM_SUP.replace("s8_support", "s7_support")), "里没有这些支撑面"),
        (run_rec(DYN, TIER2, HM_SUP, "--support-from-remote"), "只能给一个"),
        (run_rec(REMOTE, TIER2, HM_SUP), "不合要求"),
    )
    for p, want in cases:
        assert p.returncode != 0 and want in p.stderr + p.stdout, (want, p.stdout + p.stderr)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("ALL_OK")
