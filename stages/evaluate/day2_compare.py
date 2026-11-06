"""Day 2 的判定（只读 results/ 下的文本结果，只用标准库，任何机器都能跑）。

判据见 agent-doc/plan-day2-2026-09-28.md 第 2 版“判据（事先登记）”（复审后修订，跑之前定下）。输入是一份 JSON 规格
（stages/evaluate/day2_spec.json），列出：
  sequences：每条序列的 key、src_frames、是否计票（counts）、两种训练回放的四个格子（训练@评测）的评测文件、两种回放的不动基线文件、
             两种训练在匹配回放下三个 checkpoint 的评测文件（稳定性标记用）
  b：mf4090 的三条与它们在 ml40 上的对照（评测文件与不动基线文件）
每份评测文件里要有 summary_json 行（eval_summary.py 或 eval_playback_override.py 打的）；基线文件是 static_bar_from_rollout.py 的输出。

Q1（每条序列一张 2×2 表；三个场景的最小最大损失）：
  场景：固定按逐帧评（R@F 对 F@F）、固定按真实时长评（R@R 对 F@R）、按各自 checkpoint 的设置评（R@R 对 F@F）。
  指标三项：gap（ADD-AUC − 该回放的不动基线）、SR（SP-SR 与 MP-SR 合成一项：两者都判同一臂赢才算）、MPPE。
  每项在每个场景里算两臂之差，一臂在某场景的“损失” = 它落后另一臂的量（不落后为 0）；取三个场景里的最大损失，
  最大损失小的那臂赢这一项，差要 ≥ 阈值（gap 0.05、SP-SR / MP-SR 0.2、MPPE 1.0 cm；测到种子噪声时取阈值与两颗种子之差的绝对值中较大的）。
  一条序列：一臂赢 ≥ 2 项且一项不输 → 记这臂；四格 SP-SR、MP-SR 都 ≤ 3 个世界且 gap 都 < 0 → “地板”（不算票）；其余“无偏好”。
  至少 2 条有效且非地板的计票序列才下结论：≥ 2 条 R 且没有 F → 训练用真实时长更稳；对称 → 逐帧更稳；其余 → 无明确偏好，沿用配方默认（逐帧）。
  稳定性标记（不改判定）：匹配回放下三个 checkpoint 的 gap，中间那个偏离两端连线 > 0.05 标“不稳”。另报不计票的序列（ep20）。
Q2：每格 Day 1 判据；> 0 只靠 ≤ 3 个世界或 ADD-AUC 在基线 ±0.01 内标“边缘”。
Q3：mf4090 400 万是否可用只看 B-rt：两颗种子都过 Day 1 判据，且 gap ≥ 0.5 × 对照、SP-SR 与 MP-SR ≥ 0.8 × 对照（对照是 ml40 的 ep20 真实时长满配）。
  B-frame 只报数（它的对照两个成功率约为 0，判不出好坏）。
有效性：文件都在、来历不带 -dirty / -syncing、step_count 与世界数对、同一 checkpoint 的两份评测 sha256 相同、基线帧数对得上、
四个格子与两份基线出自同一个 parquet；不满足的序列只列数、不计票。

用法：python3 stages/evaluate/day2_compare.py stages/evaluate/day2_spec.json
"""
import json
import sys
from pathlib import Path

GAP_T, SR_T, MPPE_T, TREND_T = 0.05, 0.2, 1.0, 0.05
PB = ("frame", "real")
LABEL = {"frame": "逐帧", "real": "真实时长"}
METRICS = (("gap", GAP_T, 1), ("sp", SR_T, 1), ("mp", SR_T, 1), ("mppe", MPPE_T, -1))
SCENARIOS = {"固定逐帧": (("real", "frame"), ("frame", "frame")), "固定真实时长": (("real", "real"), ("frame", "real")),
             "各按自己的设置": (("real", "real"), ("frame", "frame"))}


def exp_frames(pb: str, src: int) -> int:
    return src if pb == "frame" else int(round((src - 1) / 30.0 * 50.0)) + 1


def load(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    text = p.read_text()
    src = next((ln.split(":", 1)[1].strip() for ln in text.splitlines() if ln.startswith("# source")), "")
    for ln in text.splitlines():
        s = ln.strip()
        if s.startswith("summary_json "):
            d = json.loads(s[len("summary_json "):])
            d["_source"] = src
            d["_path"] = path
            return d
    raise ValueError(f"{path}: 没有 summary_json 行")


def clean(d: dict) -> bool:
    return bool(d["_source"]) and "-dirty" not in d["_source"] and "-syncing" not in d["_source"]


def parquet_of(d: dict) -> str:
    return str(d.get("parquet", "")).strip("'")


def day1(m: dict, bar: float) -> str:
    w = m.get("world_count") or 4096
    ok = m["chord_sr"] > 0 and m["maniptrans_sr"] > 0 and m["add_auc"] > bar
    marginal = any(0 < m[k] * w <= 3 for k in ("chord_sr", "maniptrans_sr")) or abs(m["add_auc"] - bar) <= 0.01
    return ("过" if ok else "不过") + ("（边缘）" if marginal else "")


def cell(m: dict, bar: float) -> dict:
    return {"gap": m["add_auc"] - bar, "sp": m["spider_sr_uncentered"], "mp": m["maniptrans_sr"], "mppe": m["mppe_cm"]}


def fmt(c: dict) -> str:
    return f"gap {c['gap']:+.3f} SP {c['sp']:.3f} MP {c['mp']:.3f} MPPE {c['mppe']:.2f}"


def minimax(cells: dict, noise: dict) -> dict:
    """每项指标：三个场景里各臂的损失（落后另一臂的量），取最大损失；最大损失小的那臂赢，差要 ≥ max(阈值, |种子噪声|)。
    返回 {指标: (R | F | =, R 的最大损失, F 的最大损失, 阈值)}；SP 与 MP 合成 SR（两者都判同一臂赢才算）。"""
    per = {}
    for k, t, sign in METRICS:
        lr = max(max(0.0, -sign * (cells[r][k] - cells[f][k])) for r, f in SCENARIOS.values())
        lf = max(max(0.0, sign * (cells[r][k] - cells[f][k])) for r, f in SCENARIOS.values())
        thr = max(t, abs(noise.get(k, 0.0)))
        per[k] = ("R" if lf - lr >= thr else "F" if lr - lf >= thr else "=", lr, lf, thr)
    sp, mp = per.pop("sp"), per.pop("mp")
    per["sr"] = (sp[0] if sp[0] == mp[0] else "=", max(sp[1], mp[1]), max(sp[2], mp[2]), max(sp[3], mp[3]))
    return per


def sequence(sq: dict, noise: dict) -> dict:
    key, src = sq["key"], sq["src_frames"]
    problems, cells, raw, bars = [], {}, {}, {}
    try:
        bars = {pb: load(sq["bars"][pb]) for pb in PB}
        for pb in PB:
            if bars[pb]["frames"] != exp_frames(pb, src):
                problems.append(f"{LABEL[pb]}基线 {bars[pb]['frames']} 帧，应为 {exp_frames(pb, src)}")
            if not clean(bars[pb]):
                problems.append(f"{sq['bars'][pb]} 来历 {bars[pb]['_source']!r}")
        for x in PB:
            for e in PB:
                m = load(sq["evals"][f"{x}@{e}"])
                raw[(x, e)] = m
                if m.get("step_count") != exp_frames(e, src):
                    problems.append(f"{x}@{e} step_count {m.get('step_count')}，应为 {exp_frames(e, src)}")
                if m.get("world_count") != 4096:
                    problems.append(f"{x}@{e} world_count {m.get('world_count')}")
                if not clean(m):
                    problems.append(f"{m['_path']} 来历 {m['_source']!r}")
                cells[(x, e)] = cell(m, bars[e]["add_auc"])
            if raw[(x, "frame")].get("checkpoint_sha256") != raw[(x, "real")].get("checkpoint_sha256"):
                problems.append(f"{LABEL[x]}训练的两份评测不是同一个 checkpoint")
        pq = {parquet_of(d) for d in list(raw.values()) + list(bars.values())}
        if len(pq) != 1 or "" in pq:
            problems.append(f"四个格子与两份基线不是同一个 parquet：{sorted(pq)}")
    except (FileNotFoundError, ValueError, KeyError) as e:
        problems.append(f"缺文件或缺字段：{e}")
    out = {"key": key, "counts": sq.get("counts", True), "problems": problems, "cells": cells, "raw": raw}
    if problems:
        out["verdict"] = "无效"
        return out
    w = {k: (raw[k].get("world_count") or 4096) for k in raw}
    if all(c["sp"] * w[k] <= 3 and c["mp"] * w[k] <= 3 and c["gap"] < 0 for k, c in cells.items()):
        out["verdict"] = "地板"
    mm = minimax(cells, noise)
    rw, fw = [k for k, v in mm.items() if v[0] == "R"], [k for k, v in mm.items() if v[0] == "F"]
    out.update(minimax=mm, real_wins=rw, frame_wins=fw)
    if "verdict" not in out:
        out["verdict"] = "R" if len(rw) >= 2 and not fw else "F" if len(fw) >= 2 and not rw else "无偏好"
    out["day1"] = {k: day1(raw[k], bars[k[1]]["add_auc"]) for k in raw}
    flags = []
    for x in PB:
        files = sq.get("trend", {}).get(x) or []
        if len(files) == 3:
            try:
                g = [load(f)["add_auc"] - bars[x]["add_auc"] for f in files]
                if abs(g[1] - (g[0] + g[2]) / 2) > TREND_T:
                    flags.append(f"{LABEL[x]}训练不稳（三个 checkpoint 的 gap {g[0]:+.3f} / {g[1]:+.3f} / {g[2]:+.3f}）")
            except (FileNotFoundError, ValueError, KeyError) as e:
                flags.append(f"{LABEL[x]}训练的趋势文件缺：{e}")
    out["flags"] = flags
    return out


def report_sequence(r: dict) -> None:
    print(f"== {r['key']}（{'计票' if r['counts'] else '只作参考'}）：{r['verdict']}")
    for p in r["problems"]:
        print(f"   有效性：{p}")
    for (x, e), c in sorted(r["cells"].items()):
        extra = f"；Day 1 判据 {r['day1'][(x, e)]}" if "day1" in r else ""
        print(f"   {LABEL[x]}训练 @ {LABEL[e]}评测：{fmt(c)}{extra}")
    if "minimax" in r:
        print("   三个场景的最大损失（R / F，阈值）：" + "；".join(f"{k} {v[1]:.3f} / {v[2]:.3f}（{v[3]:g}）→ {v[0]}"
                                                        for k, v in r["minimax"].items()))
        print(f"   真实时长赢 {r['real_wins'] or '无'}；逐帧赢 {r['frame_wins'] or '无'}")
    for f in r.get("flags", []):
        print(f"   标记：{f}")


def b_noise(b: dict) -> dict:
    """B-rt 两颗种子四项之差（43 − 42），没有就返回空。"""
    try:
        c42 = cell(load(b["b_rt"]["eval"]), load(b["b_rt"]["bar"])["add_auc"])
        c43 = cell(load(b["b_rt_s43"]["eval"]), load(b["b_rt_s43"]["bar"])["add_auc"])
        return {k: c43[k] - c42[k] for k in c42}
    except (FileNotFoundError, ValueError, KeyError):
        return {}


def b_section(b: dict, noise: dict) -> None:
    print("== Q3 mf4090（400 万 buffer）")

    def get(name):
        e = load(b[name]["eval"])
        bar = load(b[name]["bar"])["add_auc"]
        return e, bar, cell(e, bar)

    try:
        _, _, crt_c = get("ctrl_rt")
        verdicts = []
        for name in ("b_rt", "b_rt_s43"):
            e, bar, c = get(name)
            ok = (day1(e, bar).startswith("过") and c["gap"] >= 0.5 * crt_c["gap"]
                  and c["sp"] >= 0.8 * crt_c["sp"] and c["mp"] >= 0.8 * crt_c["mp"])
            verdicts.append(ok)
            print(f"   {name} {fmt(c)}（Day 1 判据 {day1(e, bar)}）→ {'达标' if ok else '不达标'}")
        print(f"   对照 ctrl_rt {fmt(crt_c)}；mf4090 400 万 → {'可用' if all(verdicts) else '不可用'}（两颗种子都要达标）")
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"   B-rt：缺结果（{e}），不下结论")
    try:
        bf, _, bf_c = get("b_frame")
        cf, _, cf_c = get("ctrl_frame")
        print(f"   B-frame（只报数）{fmt(bf_c)} chord {bf['chord_sr']:.3f}；对照 {fmt(cf_c)} chord {cf['chord_sr']:.3f}")
    except (FileNotFoundError, ValueError, KeyError) as e:
        print(f"   B-frame：缺结果（{e}）")
    if noise:
        print("   种子噪声（43 − 42）：" + "，".join(f"{k} {v:+.3f}" for k, v in noise.items()) + "（已并入 Q1 的阈值）")
    else:
        print("   种子噪声：缺结果，Q1 用固定阈值")


def main() -> int:
    spec = json.loads(Path(sys.argv[1]).read_text())
    noise = b_noise(spec["b"]) if spec.get("b") else {}
    results = [sequence(sq, noise) for sq in spec["sequences"]]
    for r in results:
        report_sequence(r)
    voting = [r for r in results if r["counts"] and r["verdict"] in ("R", "F", "无偏好")]
    nr, nf = sum(r["verdict"] == "R" for r in voting), sum(r["verdict"] == "F" for r in voting)
    if len(voting) < 2:
        verdict = f"不下结论（有效且非地板的计票序列只有 {len(voting)} 条）"
    elif nr >= 2 and nf == 0:
        verdict = "训练用真实时长更稳"
    elif nf >= 2 and nr == 0:
        verdict = "训练用逐帧更稳"
    else:
        verdict = "无明确偏好，沿用配方默认（逐帧）"
    print(f"Q1 判定：{verdict}（计票 {len(voting)} 条：R {nr}、F {nf}、无偏好 {len(voting) - nr - nf}）")
    if spec.get("b"):
        b_section(spec["b"], noise)
    return 0


if __name__ == "__main__":
    sys.exit(main())
