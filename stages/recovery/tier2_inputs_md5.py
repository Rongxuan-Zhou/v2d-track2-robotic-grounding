"""列出官方 Tier 2 目录里给定几集的网格、地面平面与支撑面的 md5（训练机上、仓库根执行，只读，只用标准库）。
计划 agent-doc/plan-t3-v2-retarget18-2026-10-10.md。输出给 stages/recovery/tier3_expect.py 当 <Tier 2 记录>：
每个文件一行“<md5> <相对路径>”，相对路径取自 Tier 2 的 meta/episodes_metadata.jsonl 里这一集的 mesh、ground_plane、support_surface 三列；
几集共用的文件（例如同一个物体的网格）只打一行。第一行是主机、时间与来历，最后一行 TIER2_INPUTS_DONE。
10/9 那份（results/ops/t3p_mf4090_inputs_1009.txt）是临时命令打的，md5 行的格式相同。
开头调 tier_guard.forbid_tier1()：本进程读不到 Tier 1 的任何文件。
用法：python3 stages/recovery/tier2_inputs_md5.py [--tier2 <目录>] <集>...
  落盘（Mac）：scripts/dev/record ops/<名字> ssh -o BatchMode=yes mf4090 'cd <仓库> && python3 stages/recovery/tier2_inputs_md5.py 0 3 5'
测试：stages/recovery/test_tier2_inputs_md5.py。
"""
import argparse
import datetime
import hashlib
import json
import socket
import sys
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "libs" / "v2d_infra"))
from paths import REPO, TIER2  # noqa: E402
from tier_guard import forbid_tier1  # noqa: E402

LINKED = ("mesh", "ground_plane", "support_surface")


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main(argv=None) -> int:
    forbid_tier1()
    ap = argparse.ArgumentParser()
    ap.add_argument("eps", type=int, nargs="+")
    ap.add_argument("--tier2", type=Path, default=TIER2)
    a = ap.parse_args(argv)
    src = REPO / ".v2d-source-commit"
    print(f"== {socket.gethostname()} {datetime.datetime.now().astimezone():%F %T %Z} "
          f"来历 {src.read_text().strip() if src.is_file() else 'unknown'}；Tier 2 目录 {a.tier2}")
    rows = {}
    for line in (a.tier2 / "meta" / "episodes_metadata.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[int(r["episode_index"])] = r
    missing = [e for e in a.eps if e not in rows]
    if missing:
        raise SystemExit(f"tier2_inputs_md5: Tier 2 的 episodes_metadata.jsonl 里没有 ep{missing}")
    seen = set()
    for ep in a.eps:
        for k in LINKED:
            rel = PurePosixPath(rows[ep][k])
            if rel.is_absolute() or ".." in rel.parts or not rel.parts:
                raise SystemExit(f"tier2_inputs_md5: ep{ep} 的 {k} 路径不安全：{rel}")
            if str(rel) in seen:
                continue
            f = a.tier2 / rel
            if not f.is_file():
                raise SystemExit(f"tier2_inputs_md5: ep{ep} 的 {k} 在 Tier 2 目录里没有：{rel}")
            seen.add(str(rel))
            print(f"{md5(f)} {rel}")
    print(f"TIER2_INPUTS_DONE（{len(a.eps)} 集，{len(seen)} 个文件）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
