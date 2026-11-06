#!/usr/bin/env bash
# 提交前检查的服务器端部分（计划 agent-doc/plan-submission-check-2026-10-03.md 第 4 节）：训练机上、仓库根执行，只读，只用标准库。
# 对 stages/evaluate/exports/<机器>.tsv 的每一行（档位 episode run 步数），打一行 tab 分隔的 ROW：
#   档位 episode run 步数 policy 文件的 sha256 4096 世界评测的 world_count step_count non_finite 个数 checkpoint_sha256
#   add_auc chord_sr maniptrans_sr spider_sr_uncentered mppe_cm run_record 第一条 start 记录的 parquet 路径
# 文件缺失时对应格子写 MISSING。最后一行 REMOTE_CHECK_DONE <行数>。
# 用法：scripts/dev/record submission/check_remote_<机器> stages/evaluate/submission_check_remote.sh <机器>
set -uo pipefail
cd "$(dirname "$0")/../.."
M=${1:?用法：submission_check_remote.sh mf4090|ml40}
LIST=stages/evaluate/exports/$M.tsv
[ -f "$LIST" ] || { echo "缺 $LIST" >&2; exit 2; }
python3 - "$LIST" <<'EOF'
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    if not path.is_file():
        return "MISSING"
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


n = 0
for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    if not line.strip() or line.startswith("#"):
        continue
    tier, ep, run, step = line.split("\t")[:4]
    d = Path("runs/flash_chord") / run
    policy = sha256(d / f"policy_{step}.safetensors")
    ev = d / f"eval_{step}_w4096.json"
    if ev.is_file():
        e = json.loads(ev.read_text(encoding="utf-8"))
        m = e.get("metrics", {})
        evcols = [e.get("world_count"), e.get("step_count"), len(e.get("non_finite_worlds", [])), e.get("checkpoint_sha256"),
                  m.get("add_auc"), m.get("chord_sr"), m.get("maniptrans_sr"), m.get("spider_sr_uncentered"), m.get("mppe_cm")]
    else:
        evcols = ["MISSING"] * 9
    parquet = "MISSING"
    rec = d / "run_record.jsonl"
    if rec.is_file():
        for r in rec.read_text(encoding="utf-8").splitlines():
            try:
                j = json.loads(r)
            except json.JSONDecodeError:
                continue
            if j.get("event") == "start":
                parquet = j.get("parquet", "MISSING")
                break
    print("\t".join(["ROW", tier, ep, run, step, policy, *map(str, evcols), parquet]))
    n += 1
print(f"REMOTE_CHECK_DONE {n}")
EOF
