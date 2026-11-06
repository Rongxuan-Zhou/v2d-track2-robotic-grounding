"""汇总一次 HF 归档的核对、抽查与本地状态（训练机上、仓库根执行，只读，只用标准库）。
计划 agent-doc/plan-hf-archive-policies-2026-10-08.md：第一段跑完后看核对与抽查，第二段删完后核对删除判据。
第 3 批（agent-doc/plan-hf-archive-batch3-2026-10-10.md）的清单混有 policy、旧 state 与最终 state，另报按类别的文件数与字节、
清单里的 run 还剩几份 state。
经 scripts/dev/pipe-py 在训练机上跑、scripts/dev/record 落盘：
  scripts/dev/record ops/<名字> scripts/dev/pipe-py tools/hf_archive_report.py <清单.json> <输出前缀>
输出前缀同 tools/hf_archive_run.sh 的第 4 个参数（如 runs/hf_archive_policy_mf4090），读得到哪些落盘文件就报哪些。
"""
import json
import os
import re
import struct
import sys
from collections import Counter
from pathlib import Path

RUNS = Path("runs/flash_chord")
POLICY = re.compile(r"policy_(\d+)\.safetensors")
STATE = re.compile(r"state_(\d+)\.safetensors")


def complete(p: Path) -> bool:
    """safetensors 头能解析、文件大小等于头里声明的数据末尾（同 tools/hf_archive.py 与 stages/policy/ckpt_latest.py）。"""
    try:
        size = p.stat().st_size
        with p.open("rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                return False
            (n,) = struct.unpack("<Q", raw)
            if n <= 0 or 8 + n > size:
                return False
            header = json.loads(f.read(n))
        end = max((v["data_offsets"][1] for k, v in header.items() if k != "__metadata__"), default=0)
        return size == 8 + n + end
    except (OSError, ValueError, KeyError, TypeError):
        return False


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def main() -> int:
    manifest, prefix = sys.argv[1], sys.argv[2]
    rows = json.loads(Path(manifest).read_text())
    runs = sorted({r["run"] for r in rows})
    names = {(r["run"], r["file"]) for r in rows}
    print(f"## 清单 {manifest}：{len(rows)} 个文件，{len(runs)} 条 run，共 {sum(r['size'] for r in rows)} 字节")
    steps_by_run = {}
    for r in rows:
        steps_by_run.setdefault(r["run"], []).append(r["step"])
    shape = Counter((len(v), min(v), max(v)) for v in steps_by_run.values())
    print(f"  清单按 run 分（每条 run 的文件数, 最小步, 最大步）：{dict(shape)}")
    kinds = sorted({r["kind"] for r in rows})
    print("  按类别：" + "；".join(f"{k} {sum(r['kind'] == k for r in rows)} 个、{sum(r['size'] for r in rows if r['kind'] == k)} 字节"
                                   for k in kinds))
    for tag in ("verified", "reverified"):
        v = load(f"{prefix}.{tag}.json")
        if v is None:
            print(f"## {tag}：没有这个文件")
            continue
        bad = [r["path_in_repo"] for r in v if not r["match"]]
        same = sorted((r["run"], r["file"]) for r in v) == sorted(names)
        print(f"## {tag}：{len(v)} 行，一致 {len(v) - len(bad)}，不一致 {len(bad)}；文件与清单逐项相同：{'是' if same else '否'}")
        for b in bad[:20]:
            print(f"  不一致 {b}")
    s = load(f"{prefix}.spotcheck.json")
    if s is None:
        print("## 抽查：没有这个文件")
    else:
        ch = s["checked"]
        per_run = Counter(c["path_in_repo"].split("/")[3] for c in ch)
        print(f"## 抽查：随机种子 {s['seed']}，{len(ch)} 个，一致 {sum(bool(c['match']) for c in ch)}，覆盖 {len(per_run)} 条 run"
              f"（每条最多 {max(per_run.values()) if per_run else 0} 个），ok = {s['ok']}")
        for c in ch:
            print(f"  {'ok ' if c['match'] else 'BAD'} {c['path_in_repo']}")
    print(f"## 本地：runs/flash_chord 下 policy 共 {sum(1 for _ in RUNS.glob('*/policy_*.safetensors'))} 份；"
          f"清单里的文件还在本地的 {sum((RUNS / r).joinpath(f).exists() for r, f in names)} / {len(rows)}")
    counts, newest = [], Counter()
    for run in runs:
        steps = sorted(int(m.group(1)) for p in (RUNS / run).glob("policy_*.safetensors") if (m := POLICY.fullmatch(p.name)))
        counts.append(len(steps))
        newest[steps[-1] if steps else None] += 1
    print(f"  清单里的 {len(runs)} 条 run：policy 份数最少 {min(counts)}、最多 {max(counts)}；最新一份的步数 {dict(newest)}")
    kept_shape, kept_total, kept_ok, unpaired = Counter(), 0, 0, []
    for run in runs:
        d = RUNS / run
        pols = {int(m.group(1)): p for p in d.glob("policy_*.safetensors") if (m := POLICY.fullmatch(p.name))}
        kept = sorted(st for st, p in pols.items() if (run, p.name) not in names)
        kept_total += len(kept)
        kept_ok += sum(complete(pols[st]) for st in kept)
        kept_shape[(len(kept), kept[0] if kept else None, kept[-1] if kept else None)] += 1
        unpaired += [f"{run}/{p.name}" for p in d.glob("state_*.safetensors")
                     if (m := STATE.fullmatch(p.name)) and int(m.group(1)) not in kept]
    print(f"  删除清单里的文件后每条 run 留下的 policy（份数, 最小步, 最大步）：{dict(kept_shape)}；共 {kept_total} 份，头部完整 {kept_ok} 份")
    print(f"  这些 run 的 state 都有留下的同步数 policy 配对：" + ("是" if not unpaired else f"否（{len(unpaired)} 个）：{', '.join(unpaired[:5])}"))
    left_states = Counter(sum(1 for _ in (RUNS / run).glob("state_*.safetensors")) for run in runs)
    print(f"  清单里的 {len(runs)} 条 run 现在各剩几份 state（份数: run 数）：{dict(sorted(left_states.items()))}")
    entries, recorded = Counter(), 0
    for run in runs:
        a = load(RUNS / run / "ARCHIVED.json")
        entries[len(a) if a else 0] += 1
        recorded += sum((run, f["file"]) in names for e in (a or []) for f in e["files"])
    print(f"## ARCHIVED.json：全机 {sum(1 for _ in RUNS.glob('*/ARCHIVED.json'))} 条 run 有；清单里的 run 按记录条数分 "
          f"{dict(sorted(entries.items()))}；清单文件记进去的 {recorded} / {len(rows)}")
    st = Path("runs/hf_stage")
    print(f"## 暂存目录 runs/hf_stage：" + (f"在，{sum(1 for _ in st.rglob('*.safetensors'))} 个 safetensors" if st.exists() else "不在"))
    for tag in ("files_before", "files_after", "gone", "allowed"):
        p = Path(f"{prefix}.{tag}.txt")
        if p.exists():
            with p.open() as f:
                print(f"## {tag}：{sum(1 for _ in f)} 行")
    fs = os.statvfs(".")
    print(f"## 磁盘：可用 {fs.f_bavail * fs.f_frsize // 1048576} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
