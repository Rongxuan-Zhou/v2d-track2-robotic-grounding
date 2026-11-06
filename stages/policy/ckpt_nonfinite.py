"""扫 FlashCHORD policy checkpoint 里的非有限参数（训练机上跑，只用标准库）。

safetensors 格式：8 字节小端头长度 + JSON 头（张量名 → dtype、shape、data_offsets）+ 数据。policy 文件是 31 个 F32 张量。
先确认文件完整（头能解析、文件大小 = 8 + 头长 + 数据末尾，且最后修改已过 10 秒，同 ckpt_latest.py 的判定再加写入余量），
不完整的报“不完整”、不计入。逐个张量用 array('f') 读数据，数 NaN 与 ±inf；非 F32 的张量报出名字、不扫。

用法（仓库根，或经 scripts/dev/pipe-py 在训练机上跑）：
  python3 stages/policy/ckpt_nonfinite.py <run>...          每个 run 的全部 policy_<步数>.safetensors，按步数排
  python3 stages/policy/ckpt_nonfinite.py --last <run>...   每个 run 只看步数最大的一份完整 checkpoint
  加 --by-tensor：列出含非有限值的张量、形状与下标区间；加 --steps 40001536,50001920：只看这些步数
  加 --fail-on-nonfinite：看过的 checkpoint 里有非有限参数、或 --last 找不到完整的一份，退出码 1（冒烟判据与训练链用）
  python3 stages/policy/ckpt_nonfinite.py --watch <run> [--interval 120] [--timeout 14400] [--stop-after 2]
      看守：每 interval 秒看一次新写完的 policy；按步数连续 stop-after 份完整 checkpoint 都含非有限参数，就调用
      stages/policy/fc_stop.sh <run> 并退出；run 目录已有 DONE 或 STOP 时只报告、不止损（DONE 优先）。到 timeout 秒也退出。
      另每轮数一次本地 .wandb 里的 NaN 字样，变了就报（只报警）。--tag 写部署时的 commit，启动时连同脚本 md5 打出。
      --stop-at-step N：写出 N 步及以后的完整 checkpoint 就 fc_stop（截断复现：总步数不变、学习率曲线不变，只提前停）；非有限止损优先。
      --parent PID：这个进程不在了就退出（NAN_WATCH_ORPHAN），免得包装脚本被杀后看守一直挂着、日后误停同名 run。
      只有 fc_stop.sh 这一个写操作；计划 agent-doc/plan-ep1-ep7-divergence-2026-09-30.md 的 M1、M2。
时间一律丹佛时间（MDT，UTC−6）。类型注解用 typing，Mac 自带的 Python 3.9 也能跑。
"""
import argparse
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import sys
import time
from array import array
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

MDT = timezone(timedelta(hours=-6))
SETTLE_S = 10


def now() -> str:
    return datetime.now(tz=MDT).strftime("%m-%d %H:%M:%S")


def header(path: Path) -> Optional[Tuple[dict, int]]:
    """(头, 数据起点)；文件不完整或还在写返回 None。"""
    try:
        st = path.stat()
        if time.time() - st.st_mtime < SETTLE_S:
            return None
        with path.open("rb") as f:
            raw = f.read(8)
            if len(raw) != 8:
                return None
            (n,) = struct.unpack("<Q", raw)
            if n <= 0 or 8 + n > st.st_size:
                return None
            h = json.loads(f.read(n))
        end = max((v["data_offsets"][1] for k, v in h.items() if k != "__metadata__"), default=0)
        return (h, 8 + n) if st.st_size == 8 + n + end else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def runs_of(indices: List[int]) -> str:
    """把排好序的下标压成区间：0-3,7,9-12。"""
    out, start, prev = [], None, None
    for i in indices:
        if start is None:
            start = prev = i
        elif i == prev + 1:
            prev = i
        else:
            out.append(f"{start}-{prev}" if prev > start else f"{start}")
            start = prev = i
    if start is not None:
        out.append(f"{start}-{prev}" if prev > start else f"{start}")
    return ",".join(out)


def scan(path: Path, by_tensor: bool = False) -> Optional[dict]:
    """{'total', 'nan', 'inf', 'skipped', 'tensors'}；不完整返回 None。by_tensor 时 tensors 列出含非有限值的张量与下标区间。"""
    got = header(path)
    if got is None:
        return None
    h, base = got
    total = nan = inf = 0
    skipped = []
    tensors = []
    with path.open("rb") as f:
        for name, v in h.items():
            if name == "__metadata__":
                continue
            if v["dtype"] != "F32":
                skipped.append(name)
                continue
            start, end = v["data_offsets"]
            f.seek(base + start)
            a = array("f")
            a.frombytes(f.read(end - start))
            if sys.byteorder != "little":
                a.byteswap()
            total += len(a)
            bad = []
            for i, x in enumerate(a):
                if x != x:
                    nan += 1
                    bad.append(i)
                elif math.isinf(x):
                    inf += 1
                    bad.append(i)
            if by_tensor and bad:
                tensors.append((name, v["shape"], len(bad), runs_of(bad)))
    return {"total": total, "nan": nan, "inf": inf, "skipped": skipped, "tensors": tensors}


def policies(run: str) -> List[Tuple[int, Path]]:
    out = []
    for p in Path("runs/flash_chord", run).glob("policy_*.safetensors"):
        m = re.fullmatch(r"policy_(\d+)\.safetensors", p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def line(step: int, r: Optional[dict]) -> str:
    if r is None:
        return f"  {step:>11,d} 步：不完整或还在写，跳过"
    bad = r["nan"] + r["inf"]
    s = f"  {step:>11,d} 步：非有限 {bad} / {r['total']}（NaN {r['nan']}，inf {r['inf']}）"
    return s + (f"；未扫的非 F32 张量 {r['skipped']}" if r["skipped"] else "")


def tensor_lines(r: dict) -> None:
    for name, shape, n, ranges in r.get("tensors", []):
        print(f"      {name} {shape}：非有限 {n} 个，下标 {ranges}")


def report(runs: List[str], last_only: bool, by_tensor: bool = False, steps: Optional[set] = None) -> bool:
    """打印报告；返回是否“有问题”：看过的 checkpoint 里有非有限参数，或 --last 时某个 run 没有完整的一份。"""
    problem = False
    for run in runs:
        items = policies(run)
        print(f"== {run}（policy {len(items)} 份）")
        if not items:
            print("  没有 policy checkpoint")
            problem = problem or last_only
            continue
        if last_only:
            for step, p in reversed(items):
                r = scan(p, by_tensor)
                if r is not None:
                    print(line(step, r))
                    tensor_lines(r)
                    problem = problem or bool(r["nan"] + r["inf"])
                    break
            else:
                print("  没有完整的 checkpoint")
                problem = True
            continue
        first_bad = None
        for step, p in items:
            if steps and step not in steps:
                continue
            r = scan(p, by_tensor)
            print(line(step, r))
            if r is not None:
                tensor_lines(r)
            if r is not None and r["nan"] + r["inf"] and first_bad is None:
                first_bad = step
        print(f"  第一份含非有限参数：{first_bad:,d} 步" if first_bad else "  全部有限")
        problem = problem or first_bad is not None
    return problem


def wandb_nan_count(run: str) -> Optional[int]:
    """最新一个本地 .wandb 文件里 NaN 字样的个数（边训边写；只作报警，不作止损依据）。"""
    files = sorted(Path("runs/flash_chord", run, "wandb").glob("run-*/run-*.wandb"), key=lambda p: p.stat().st_mtime)
    if not files:
        return None
    try:
        return files[-1].read_bytes().count(b"NaN")
    except OSError:
        return None


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def watch(run: str, interval: int, timeout: int, stop_after: int, tag: str, stop_at_step: int = 0, parent: int = 0) -> int:
    d = Path("runs/flash_chord", run)
    md5 = hashlib.md5(Path(__file__).read_bytes()).hexdigest() if Path(__file__).is_file() else "?"
    extra = f"；写出 {stop_at_step:,d} 步及以后的 checkpoint 就 fc_stop" if stop_at_step else ""
    print(f"=== {now()} MDT 开始看守 {run}：每 {interval} 秒；连续 {stop_after} 份非有限就 fc_stop{extra}；最长 {timeout} 秒；"
          f"脚本 md5 {md5}；{tag}", flush=True)
    seen: Dict[int, bool] = {}
    last_wandb = None
    t0 = time.time()
    while True:
        if parent and not alive(parent):
            print(f"=== {now()} MDT 父进程 {parent} 已不在，扫过 {len(seen)} 份；NAN_WATCH_ORPHAN", flush=True)
            return 0
        finished = (d / "DONE").exists() or (d / "STOP").exists()
        for step, p in policies(run):
            if step in seen:
                continue
            r = scan(p)
            if r is None:
                continue
            seen[step] = bool(r["nan"] + r["inf"])
            print(f"{now()} MDT {line(step, r).strip()}", flush=True)
        w = wandb_nan_count(run)
        if w is not None and w != last_wandb:
            print(f"{now()} MDT 报警信号：.wandb 里 NaN 字样 {w} 处（只报不停）", flush=True)
            last_wandb = w
        if finished:  # 已结束的 run 不再止损
            print(f"=== {now()} MDT run 已结束（{'DONE' if (d / 'DONE').exists() else 'STOP'}），扫过 {len(seen)} 份；NAN_WATCH_DONE", flush=True)
            return 0
        steps = sorted(seen)
        tail = [seen[s] for s in steps[-stop_after:]]
        if len(tail) == stop_after and all(tail):
            if (d / "DONE").exists():  # 扫描期间 run 刚好写了 DONE：已结束的 run 不止损（DONE 优先），交给训完后的最终检查
                print(f"=== {now()} MDT 连续 {stop_after} 份非有限，但 run 已 DONE，不止损；NAN_WATCH_DONE", flush=True)
                return 0
            print(f"=== {now()} MDT 连续 {stop_after} 份非有限（{', '.join(f'{s:,d}' for s in steps[-stop_after:])} 步）：fc_stop {run}", flush=True)
            rc = subprocess.run(["stages/policy/fc_stop.sh", run]).returncode
            print(f"=== {now()} MDT fc_stop 退出码 {rc}；NAN_WATCH_STOPPED", flush=True)
            return 0
        if stop_at_step and any(s >= stop_at_step for s in seen):
            print(f"=== {now()} MDT 已写出 {max(seen):,d} 步的 checkpoint（≥ {stop_at_step:,d}）：按截断计划 fc_stop {run}", flush=True)
            rc = subprocess.run(["stages/policy/fc_stop.sh", run]).returncode
            print(f"=== {now()} MDT fc_stop 退出码 {rc}；NAN_WATCH_TRUNCATED", flush=True)
            return 0
        if time.time() - t0 > timeout:
            print(f"=== {now()} MDT 到时，扫过 {len(seen)} 份；NAN_WATCH_TIMEOUT", flush=True)
            return 0
        time.sleep(interval)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--last", action="store_true")
    ap.add_argument("--by-tensor", action="store_true", help="列出含非有限值的张量、形状与下标区间")
    ap.add_argument("--steps", default="", help="只看这些步数（逗号分隔）")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--interval", type=int, default=120)
    ap.add_argument("--timeout", type=int, default=14400)
    ap.add_argument("--stop-after", type=int, default=2)
    ap.add_argument("--tag", default="", help="启动时照打，记部署的 commit 等")
    ap.add_argument("--stop-at-step", type=int, default=0, help="看守：写出这一步数及以后的完整 checkpoint 就 fc_stop（截断复现用）")
    ap.add_argument("--fail-on-nonfinite", action="store_true", help="看过的 checkpoint 有非有限参数（或 --last 没有完整的一份）时退出码 1")
    ap.add_argument("--parent", type=int, default=0, help="看守：这个进程不在了就退出")
    a = ap.parse_args()
    for run in a.runs:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", run):
            sys.exit(f"run 名非法：{run}")
    if a.watch:
        if len(a.runs) != 1 or a.stop_after < 1:
            sys.exit("--watch 只看一个 run，--stop-after ≥ 1")
        if not re.fullmatch(r"[A-Za-z0-9_.:=, -]*", a.tag):
            sys.exit("--tag 只能是字母数字与 _.:=, -")
        return watch(a.runs[0], a.interval, a.timeout, a.stop_after, a.tag, a.stop_at_step, a.parent)
    steps = {int(x) for x in a.steps.split(",") if x}
    problem = report(a.runs, a.last, a.by_tensor, steps)
    return 1 if a.fail_on_nonfinite and problem else 0


if __name__ == "__main__":
    sys.exit(main())
