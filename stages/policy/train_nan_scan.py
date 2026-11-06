"""扫训练日志末尾的 W&B 摘要，找训练发散（NaN）的 run（训练机上跑，只用标准库）。

FlashCHORD 训练结束时 wandb 在日志里打一段 "Run summary"：`wandb: <指标> <值>`。发散的 run 里动作均值、熵、回报变成 nan，
混合精度的 loss_scale 掉到 0。逐个看 runs/logs/*.s1.log，报出三项里有 nan 的 run，附 loss_scale 与回合平均长度。
还在跑的 run 没有摘要，不在此列（发散要看 checkpoint，见 ckpt_nonfinite.py）。

用法（仓库根）：python3 stages/policy/train_nan_scan.py [--table]（--table：每条一行，列出平均回合长度、平均回报、loss_scale）
  python3 stages/policy/train_nan_scan.py --log <日志>：只看这一个日志（冒烟判据用）。打一行摘要（回合长度、回报、loss_scale）；
      没有摘要、三项里有 nan、或 loss_scale 是 0 / nan 时退出码 1，否则 0。回合长度与回报只报数，不作判据。
  Mac 上：scripts/dev/record <名字> scripts/dev/pipe-py stages/policy/train_nan_scan.py（V2D_ENV 选机器）
"""
import re
import sys
from pathlib import Path

NAN_KEYS = (
    "policy/action_abs_mean_per_env_action",
    "policy/entropy_mean_per_actor_update",
    "episode/return_mean_per_completed_episode",
)
EXTRA_KEYS = ("mixed_precision/loss_scale", "episode/length_mean_per_completed_episode")
LINE = re.compile(r"^wandb:\s+(\S+)\s+(\S+)\s*$")


def summary(path: Path) -> dict[str, str]:
    """最后一段 Run summary 里的 指标 → 值（字符串）。"""
    out: dict[str, str] = {}
    in_summary = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("wandb: Run summary:"):
            in_summary, out = True, {}
            continue
        if in_summary:
            m = LINE.match(line)
            if m:
                out[m.group(1)] = m.group(2)
            elif not line.startswith("wandb:") or line.strip() == "wandb:":
                in_summary = False
    return out


def table(logs) -> None:
    """每个带摘要的训练日志一行：平均回合长度、平均回报、loss_scale、摘要有无 nan。"""
    print("run | 平均回合长度 | 平均回报 | loss_scale | 摘要有 nan")
    for p in logs:
        s = summary(p)
        if not s:
            continue
        nan = any(s.get(k, "").lower() == "nan" for k in NAN_KEYS)
        print(f"{p.name[:-len('.s1.log')]} | {s.get('episode/length_mean_per_completed_episode', '—')} | "
              f"{s.get('episode/return_mean_per_completed_episode', '—')} | {s.get('mixed_precision/loss_scale', '—')} | {'是' if nan else '否'}")


def check_log(path: Path) -> int:
    """单个日志的发散判据：没有摘要、NAN_KEYS 里有 nan、loss_scale 为 0 或 nan，任何一条都算发散（退出码 1）。"""
    s = summary(path)
    if not s:
        print(f"{path.name}：没有 W&B 摘要")
        return 1
    nan = [k.split("/")[-1] for k in NAN_KEYS if s.get(k, "").lower() == "nan"]
    ls = s.get("mixed_precision/loss_scale", "")
    try:
        ls_bad = ls.lower() == "nan" or float(ls) == 0.0
    except ValueError:
        ls_bad = False  # 摘要里没有这一项（或被截成 “...”）：不据此判
    print(f"{path.name}：平均回合长度 {s.get('episode/length_mean_per_completed_episode', '—')}，"
          f"平均回报 {s.get('episode/return_mean_per_completed_episode', '—')}，loss_scale {ls or '—'}；"
          f"摘要里 nan {len(nan)} 项{('（' + '、'.join(nan) + '）') if nan else ''}")
    return 1 if nan or ls_bad else 0


def main() -> int:
    if sys.argv[1:2] == ["--log"]:
        if len(sys.argv) != 3:
            raise SystemExit("用法：train_nan_scan.py --log <日志>")
        return check_log(Path(sys.argv[2]))
    logs = sorted(Path("runs/logs").glob("*.s1.log"))
    if "--table" in sys.argv[1:]:
        table(logs)
        return 0
    with_summary = 0
    hits = []
    for p in logs:
        s = summary(p)
        if not s:
            continue
        with_summary += 1
        nan = [k for k in NAN_KEYS if s.get(k, "").lower() == "nan"]
        if nan:
            hits.append((p, nan, s))
    for p, nan, s in hits:
        extra = "，".join(f"{k.split('/')[-1]}={s.get(k, '—')}" for k in EXTRA_KEYS)
        print(f"NaN：{p.name}  {len(nan)} 项（{'、'.join(k.split('/')[-1] for k in nan)}）；{extra}")
    print(f"训练日志 {len(logs)} 个，其中带 W&B 摘要 {with_summary} 个；摘要里有 NaN 的 {len(hits)} 个")
    return 0


if __name__ == "__main__":
    sys.exit(main())
