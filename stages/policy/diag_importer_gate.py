"""诊断队列的导入器核对：计划 agent-doc/plan-ep1-ep7-divergence-2026-09-30.md 附录 A.3 的有效性闸（训练机上跑，只用标准库）。

队列每行用 V2D_FC_SRC= 指定导入器：非空 = 挂载 make_patched_fc_src.sh 生成的 #185 副本；空 = in-flash-chord 默认挂载的仓库 third_party/video_to_data 里的 v0.4.0 源码（scripts/dev/in-flash-chord:9）。
scripts/dev/in-flash-chord 挂载副本时往 stderr 打一行 `in-flash-chord: /workspace ← <副本真实路径>（…）`；fc_run.sh:194 把训练段那次
in-flash-chord 的 stderr 写进 runs/logs/<run>.log。runs/logs/<run>.s<段>.log 是 docker logs，只有容器里的输出，不会有这一行
（A.3 原文写成 .s1.log，是写错了文件；这里两份都数，.s1.log 的行数只作对照）。
判定：#185 的 run，<run>.log 至少一行、且每行都指向该副本；v0.4.0 的 run 一行都没有。<run>.log 里没有“第 1 段”开始行的记“未跑”，
但已有 DONE 的记“无法核对”（算无效）。有无效的 run 时退出码 1。

用法（仓库根）：python3 stages/policy/diag_importer_gate.py <队列文件>...
  Mac 上：V2D_ENV=ml40 scripts/dev/record <名字> scripts/dev/pipe-py stages/policy/diag_importer_gate.py stages/policy/queues/diag_ml40_ep7.txt …
"""
import os
import re
import sys
from pathlib import Path

MOUNT = "in-flash-chord: /workspace ← "
ENV_TOKEN = re.compile(r"[A-Z_][A-Z0-9_]*=\S*")


def mount_lines(path: Path) -> list:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    return [line.split(MOUNT, 1)[1] for line in text.splitlines() if MOUNT in line]


def check(line: str) -> tuple:
    toks = line.split()
    env = {}
    while toks and ENV_TOKEN.fullmatch(toks[0]):
        k, v = toks.pop(0).split("=", 1)
        env[k] = v
    run = toks[0]
    if "V2D_FC_SRC" not in env:
        return run, "?", "队列行没写 V2D_FC_SRC", False
    src = env["V2D_FC_SRC"]
    log = Path("runs/logs") / f"{run}.log"
    started = log.exists() and re.search(rf"fc_run {re.escape(run)} 第 1 段", log.read_text(encoding="utf-8", errors="replace"))
    if not started:
        # fc_config_check.sh 走 fc_run.sh 的打印模式，会先 touch 出空的 <run>.log（fc_run.sh:47）；有 DONE 却没有训练段开始行才算无法核对
        if (Path("runs/flash_chord") / run / "DONE").exists():
            return run, "#185" if src else "v0.4.0", "无法核对（有 DONE，日志里却没有第 1 段开始行）", False
        return run, "#185" if src else "v0.4.0", "未跑（还没进训练段）", None
    got = mount_lines(log)
    s1 = len(mount_lines(Path("runs/logs") / f"{run}.s1.log"))
    if src:
        real = os.path.realpath(src)
        hit = sum(1 for g in got if g.startswith(real + "（"))
        ok = len(got) >= 1 and hit == len(got)
        note = f"<run>.log 挂载行 {len(got)}（指向 {real} 的 {hit}）；.s1.log 挂载行 {s1}"
        return run, "#185", note, ok
    note = f"<run>.log 挂载行 {len(got)}；.s1.log 挂载行 {s1}"
    return run, "v0.4.0", note, len(got) == 0


def main() -> int:
    counts = {"有效": 0, "无效": 0, "未跑": 0}
    for q in sys.argv[1:]:
        print(f"== {q}")
        for raw in Path(q).read_text(encoding="utf-8").splitlines():
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            run, kind, note, ok = check(line)
            tag = "未跑" if ok is None else ("有效" if ok else "无效")
            counts[tag] += 1
            print(f"  {run}  导入器 {kind}  {note}  → {tag}")
    print(f"合计：有效 {counts['有效']}、无效 {counts['无效']}、未跑 {counts['未跑']}")
    return 1 if counts["无效"] else 0


if __name__ == "__main__":
    sys.exit(main())
