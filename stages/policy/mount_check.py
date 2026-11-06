"""方盒 run 的挂载核对：训练挂的是不是 #185 补丁副本、评测挂的是不是 8c778f9 副本（训练机上、仓库根执行，只用标准库）。

计划 agent-doc/plan-box6-and-eval-runtime-2026-10-01.md 第 4.3、4.5 节。fc_run 的运行记录只记 third_party 的提交，看不出 /workspace 挂了
哪份源码；scripts/dev/in-flash-chord 挂副本时往 stderr 打一行 `in-flash-chord: /workspace ← <副本真实路径>（…）`。fc_run.sh 把训练段那次
in-flash-chord 的 stderr 写进 runs/logs/<run>.log（冒烟 runs/logs/smoke-<run>.log），fc_eval.sh 把评测的写进
runs/logs/<run>.eval_<步数>_w<世界数>.log。record 的结果文件里没有这一行。
判定：冒烟与正式的训练日志各至少一行挂载、全部指向训练副本；评测日志至少一行、全部指向评测副本。路径按真实路径比。
用法：python3 stages/policy/mount_check.py --train <训练副本> --eval <评测副本> [--steps 349999104] [--worlds 4096] <正式 run 名>...
  最后一行 MOUNT_OK 或 MOUNT_FAIL n；有不过的退出码 1。
"""
import argparse
import os
import re
import sys
from pathlib import Path

MOUNT = re.compile(r"in-flash-chord: /workspace ← (.+?)（")


def mounts(log: Path) -> list:
    if not log.exists():
        return []
    return MOUNT.findall(log.read_text(encoding="utf-8", errors="replace"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--steps", default="349999104")
    ap.add_argument("--worlds", default="4096")
    ap.add_argument("runs", nargs="+")
    a = ap.parse_args()
    want_train, want_eval = os.path.realpath(a.train), os.path.realpath(a.eval)
    bad = 0
    for run in a.runs:
        checks = ((Path(f"runs/logs/smoke-{run}.log"), want_train, "冒烟训练"), (Path(f"runs/logs/{run}.log"), want_train, "正式训练"),
                  (Path(f"runs/logs/{run}.eval_{a.steps}_w{a.worlds}.log"), want_eval, "评测"))
        for log, want, what in checks:
            got = mounts(log)
            ok = bool(got) and all(os.path.realpath(g) == want for g in got)
            bad += not ok
            print(f"  {'ok  ' if ok else 'FAIL'} {run} {what}：{log} 里挂载 {len(got)} 次，"
                  + ("全部指向 " + want if ok else f"应全部指向 {want}，实际 {sorted(set(got)) or '无'}"))
    print("MOUNT_OK" if bad == 0 else f"MOUNT_FAIL {bad}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
