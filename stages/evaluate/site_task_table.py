"""SOP 页面“任务一览”用的任务表（训练机上跑，只用标准库）：每条 episode 一行。

列：ep、sequence_id、短名（去掉日期时间前缀，页面上当英文任务名用）、物体、帧数、支撑面（FlashCHORD 实际读的
runs/motion/tier1/whole_body/reconstructed_stage/<SEQ>_support.usda 里 Cube / Cylinder 的个数，都没有记 none）、官方英文任务原文。
来源：数据集 meta/episodes_metadata.jsonl（官方任务原文照抄）。这是给人看的文档素材，不进入任何 Tier 2 / Tier 3 管线；
exo 视频与任务原文在三档之间相同（9/30 review 逐条核对 tier_1 与 tier_2 的 episodes_metadata.jsonl 相同），读 tier_2 目录，新代码里不出现 Tier 1 路径。
用法（仓库根）：python3 stages/evaluate/site_task_table.py [--tsv 输出路径]
  Mac 上落盘：scripts/dev/record site/task_table scripts/dev/pipe-py stages/evaluate/site_task_table.py
"""
import json
import re
import sys
from pathlib import Path

META = Path("data/challenge/track_2/tier_2_synthetic_noise/meta/episodes_metadata.jsonl")
STAGE = Path("runs/motion/tier1/whole_body/reconstructed_stage")


def support_kind(seq: str) -> str:
    usda = STAGE / f"{seq}_support.usda"
    if not usda.exists():
        return "missing"
    text = usda.read_text(encoding="utf-8", errors="replace")
    cubes = len(re.findall(r'def Cube "', text))
    cyls = len(re.findall(r'def Cylinder "', text))
    parts = ([f"Cube×{cubes}"] if cubes else []) + ([f"Cylinder×{cyls}"] if cyls else [])
    return "+".join(parts) or "none"


def main() -> None:
    rows = []
    for line in META.read_text(encoding="utf-8").splitlines():
        d = json.loads(line)
        seq = d["sequence_id"]
        short = re.sub(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_", "", seq).rstrip("_")
        rows.append((d["episode_index"], seq, short, d["object"], d["frames"], support_kind(seq), d["task"].strip()))
    rows.sort()
    out = ["ep\tsequence_id\tshort_name\tobject\tframes\tsupport\ttask"]
    out += ["\t".join(str(x) for x in r) for r in rows]
    text = "\n".join(out)
    print(text)
    if "--tsv" in sys.argv:
        Path(sys.argv[sys.argv.index("--tsv") + 1]).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
