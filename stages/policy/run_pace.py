"""按 checkpoint 落盘时间看一条或几条 FlashCHORD 正式训练的节奏（宿主机上跑，只用标准库）。

每条 run 打印：训练链日志里的开始、完成时间与用时，以及每份 policy_<步数>.safetensors 的落盘时间和相邻两份的间隔
（约 1000 万步一份）。时间一律换成丹佛时间（MDT，UTC−6）；链日志是北京时间（CST，UTC+8）。
checkpoint 的时间取文件 mtime：训练写完这一份的时刻。还在跑的 run 只列到最新一份。

用法（仓库根）：python3 stages/policy/run_pace.py <训练链日志或 fc_queue.sh 的队列日志> <run 名>...
  Mac 上经 stages/policy/run_pace.sh 在训练机上执行，再由 scripts/dev/record 落盘。
"""
import re
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

MDT = timezone(timedelta(hours=-6))
CST = timezone(timedelta(hours=8))
LOG_LINE = re.compile(r"^=== (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) CST (.*)$")


def mdt(ts: datetime) -> str:
    return ts.astimezone(MDT).strftime("%m-%d %H:%M:%S")


def mmss(seconds: float) -> str:
    s = round(seconds)
    return f"{s // 60}:{s % 60:02d}"


def chain_times(log: Path, run: str) -> tuple[datetime | None, datetime | None]:
    """训练链日志（“=== <时间> CST 正式 <run>”…“<run> 完成”），或 fc_queue.sh 的队列日志（“=== <时间> CST 队列第 N 项：<run> ===”，
    完成行 “fc_run: <run> 完成” 不带时间，完成时刻取它后面第一条带时间的行，即下一项开始或“全部完成”）。"""
    start = end = None
    done = False
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        if start is not None and line.strip() == f"fc_run: {run} 完成":
            done = True
            continue
        m = LOG_LINE.match(line)
        if not m:
            continue
        ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=CST)
        msg = m.group(2)
        if msg == f"正式 {run}" or re.fullmatch(rf"队列第 \d+ 项：{re.escape(run)} ===", msg):
            start, end, done = ts, None, False  # 同名 run 重开时以最后一次为准
        elif msg.startswith(f"{run} 完成"):
            end = ts
        elif done and end is None:
            end = ts
    return start, end


def main(log_path: str, runs: list[str]) -> None:
    log = Path(log_path)
    for run in runs:
        print(f"== {run}")
        start, end = chain_times(log, run)
        if start is None:
            print("  链日志里没有这条的正式训练")
            continue
        head = f"  正式开始 {mdt(start)} MDT"
        head += f"；完成 {mdt(end)} MDT；用时 {mmss((end - start).total_seconds())}（分:秒）" if end else "；还在跑"
        print(head)
        ckpts = []
        for p in Path("runs/flash_chord", run).glob("policy_*.safetensors"):
            m = re.fullmatch(r"policy_(\d+)\.safetensors", p.name)
            if m:
                ckpts.append((int(m.group(1)), datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)))
        ckpts.sort()
        if not ckpts:
            print("  还没有 checkpoint")
            continue
        prev_t = start
        gaps = []
        for step, t in ckpts:
            gap = (t - prev_t).total_seconds()
            gaps.append(gap)
            print(f"  {step:>11,d} 步  {mdt(t)} MDT  间隔 {mmss(gap)}")
            prev_t = t
        # 第一份包含编译与预热，不计入节奏统计
        body = gaps[1:]
        if body:
            print(f"  间隔（不含第一份）：中位 {mmss(statistics.median(body))}，最短 {mmss(min(body))}，最长 {mmss(max(body))}，共 {len(body)} 段")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("用法：run_pace.py <训练链日志> <run 名>...")
    main(sys.argv[1], sys.argv[2:])
