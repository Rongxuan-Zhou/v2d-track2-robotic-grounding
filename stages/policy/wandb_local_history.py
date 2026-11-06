"""解析 FlashCHORD 训练在本地写的 W&B 记录（run-*.wandb），找各指标第一次出现非有限值的位置（Mac 上用 uv 带 wandb 跑）。

W&B 边训边把每一行 history 写进 run 目录的 run-<id>.wandb（protobuf 记录，LevelDB 日志格式），不需要 W&B API 与凭据。
读法：wandb.sdk.internal.datastore.DataStore 逐条取记录，解析成 wandb_internal_pb2.Record，只看 history 记录。
每个指标报：第一次非有限（nan / inf）的行号与环境步数、之前最后一个有限值、之后有没有恢复。环境步数取 --step-key 指定的指标
（默认 `_step`：FlashCHORD 把环境步数记成 W&B 的 step，每行约 2048 世界 × 25 步 = 51,200 步）。另把 --around 列出的指标在最早发散行前后几行的值打出来。

用法：uv run --no-project --with wandb==0.22.2 python stages/policy/wandb_local_history.py [--keys] [--around KEY ...] <run-*.wandb>...
  --keys：只列出指标名与行数（先看有哪些键）
  --by-stage：按课程阶段汇总（每段最后 25% 行的均值）：voc、参考进度、回合长度、各终止原因、物体位置与朝向误差
  --before N / --after N：--around 那几个指标在最早发散行之前 N 行、之后 N 行都打出来（默认 4、2）；全程有限的 run 没有这一段
  --top KEY ...（配 --topk K，默认 5）：每个指标全程最大的 K 个有限值及所在行、环境步数，另报中位数；
      用来看发散之前有没有“很大但还有限”的先兆，以及没发散的 run 里同一指标到过多大
计划 agent-doc/plan-ep1-ep7-divergence-2026-09-30.md 的 C3。
"""
import argparse
import json
import math
from pathlib import Path

from wandb.proto import wandb_internal_pb2
from wandb.sdk.internal import datastore

DEFAULT_AROUND = (
    "curriculum/stage_index",
    "curriculum/voc_scale",
    "mixed_precision/loss_scale",
    "episode/length_mean_per_completed_episode",
    "episode/return_mean_per_completed_episode",
    "policy/action_abs_mean_per_env_action",
)


def rows(path: Path) -> list[dict]:
    ds = datastore.DataStore()
    ds.open_for_scan(str(path))
    out = []
    while True:
        data = ds.scan_data()
        if data is None:
            break
        rec = wandb_internal_pb2.Record()
        rec.ParseFromString(data)
        if rec.WhichOneof("record_type") != "history":
            continue
        row = {}
        for item in rec.history.item:
            key = item.key or "/".join(item.nested_key)
            try:
                row[key] = json.loads(item.value_json)
            except json.JSONDecodeError:
                row[key] = item.value_json
        if rec.history.HasField("step"):
            row.setdefault("_step", rec.history.step.num)
        out.append(row)
    return out


def numeric(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def fmt(v) -> str:
    if not numeric(v):
        return "—" if v is None else str(v)
    if isinstance(v, float) and not math.isfinite(v):
        return str(v)
    return f"{v:.6g}"


def analyse(path: Path, around: list[str], step_key: str | None, list_keys: bool, before: int = 4, after: int = 2,
            top: list[str] | None = None, topk: int = 5) -> None:
    hist = rows(path)
    keys = sorted({k for r in hist for k in r})
    print(f"== {path.name}：history {len(hist)} 行，指标 {len(keys)} 个")
    if list_keys:
        for k in keys:
            n = sum(1 for r in hist if k in r)
            print(f"  {k}  （{n} 行）")
        return
    if step_key is None:
        step_key = "_step" if "_step" in keys else None
    print(f"  环境步数取自：{step_key or '（没找到，只报行号）'}")

    def env_step(i: int):
        if step_key is None:
            return None
        for j in range(i, -1, -1):  # 这一行没记步数就往前找最近的一行
            v = hist[j].get(step_key)
            if numeric(v):
                return v
        return None

    for k in top or []:
        xs = sorted(((r[k], i) for i, r in enumerate(hist) if numeric(r.get(k)) and math.isfinite(r[k])), reverse=True)
        if not xs:
            print(f"  {k}：没有有限值")
            continue
        med = xs[len(xs) // 2][0]
        big = "；".join(f"{fmt(v)}（第 {i} 行，{env_step(i):,.0f} 步）" if numeric(env_step(i)) else f"{fmt(v)}（第 {i} 行）" for v, i in xs[:topk])
        print(f"  {k}：有限值 {len(xs)} 行，中位数 {fmt(med)}；最大的 {min(topk, len(xs))} 个：{big}")

    first_bad = {}
    for k in keys:
        last_ok = None
        for i, r in enumerate(hist):
            v = r.get(k)
            if not numeric(v):
                continue
            if math.isfinite(v):
                last_ok = (i, v)
                continue
            later_ok = any(numeric(r2.get(k)) and math.isfinite(r2.get(k)) for r2 in hist[i + 1:])
            first_bad[k] = (i, v, last_ok, later_ok)
            break
    if not first_bad:
        print("  所有指标全程有限")
        return
    print(f"  出现过非有限值的指标 {len(first_bad)} 个（按第一次出现的行排序）：")
    for k, (i, v, last_ok, later_ok) in sorted(first_bad.items(), key=lambda kv: kv[1][0]):
        es = env_step(i)
        prev = f"之前最后有限值 {fmt(last_ok[1])}（第 {last_ok[0]} 行）" if last_ok else "之前没有有限值"
        print(f"    第 {i} 行{f'（环境步数 {es:,.0f}）' if numeric(es) else ''}：{k} = {fmt(v)}；{prev}；之后{'有' if later_ok else '没有'}恢复")
    onset = min(i for i, *_ in first_bad.values())
    lo, hi = max(0, onset - before), min(len(hist), onset + after)
    show = [k for k in around if k in keys]
    print(f"  最早发散在第 {onset} 行；第 {lo}–{hi - 1} 行的关键指标：")
    for i in range(lo, hi):
        es = env_step(i)
        vals = "；".join(f"{k.split('/')[-1]}={fmt(hist[i].get(k))}" for k in show)
        print(f"    第 {i} 行{f'（{es:,.0f} 步）' if numeric(es) else ''}：{vals}")


STAGE_KEYS = (
    ("voc", "curriculum/voc_scale"),
    ("首帧重置概率", "curriculum/reset_to_first_frame_probability"),
    ("立即首帧概率", "curriculum/immediate_first_frame_probability"),
    ("进度", "episode/reference_progress_mean_per_completed_episode"),
    ("回合长", "episode/length_mean_per_completed_episode"),
    ("终止:物体", "termination/object_fraction_per_completed_episode"),
    ("终止:手掌", "termination/palm_fraction_per_completed_episode"),
    ("终止:骨盆", "termination/pelvis_fraction_per_completed_episode"),
    ("走完", "termination/reference_end_fraction_per_completed_episode"),
    ("物体位置误差", "tracking_error/object_position_m_mean_env_max_per_step"),
    ("物体朝向误差", "tracking_error/object_orientation_rad_mean_env_max_per_step"),
)


def by_stage(path: Path) -> None:
    """按 curriculum/stage_index 分段，每段取最后 25% 的行求均值（同 wb_summary.py 的口径）。"""
    hist = rows(path)
    print(f"== {path.name}：history {len(hist)} 行")
    stages = {}
    for r in hist:
        st = r.get("curriculum/stage_index")
        if numeric(st):
            stages.setdefault(int(st), []).append(r)
    print("  段 | 步数范围 | " + " | ".join(name for name, _ in STAGE_KEYS))
    for st in sorted(stages):
        rs = stages[st]
        tail = rs[len(rs) - max(1, len(rs) // 4):]
        vals = []
        for _, key in STAGE_KEYS:
            xs = [r[key] for r in tail if numeric(r.get(key)) and math.isfinite(r[key])]
            vals.append(f"{sum(xs) / len(xs):.3g}" if xs else "—")
        lo, hi = rs[0].get("_step"), rs[-1].get("_step")
        print(f"  {st} | {lo / 1e6:.0f}–{hi / 1e6:.0f}M | " + " | ".join(vals))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--keys", action="store_true")
    ap.add_argument("--by-stage", action="store_true", help="按课程阶段汇总关键指标（每段最后 25% 的均值）")
    ap.add_argument("--around", nargs="*", default=list(DEFAULT_AROUND))
    ap.add_argument("--step-key", default=None)
    ap.add_argument("--before", type=int, default=4, help="最早发散行之前打几行")
    ap.add_argument("--after", type=int, default=2, help="从最早发散行起打几行")
    ap.add_argument("--top", nargs="*", default=[], help="这些指标各报全程最大的几个有限值")
    ap.add_argument("--topk", type=int, default=5)
    a = ap.parse_args()
    for f in a.files:
        if a.by_stage:
            by_stage(f)
        else:
            analyse(f, a.around, a.step_key, a.keys, a.before, a.after, a.top, a.topk)


if __name__ == "__main__":
    main()
