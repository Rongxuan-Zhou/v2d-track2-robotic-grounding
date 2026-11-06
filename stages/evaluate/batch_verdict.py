"""按期望表逐条判 Day 1 判据（只读 results/ 下的文本结果，只用标准库，任何机器都能跑）。

判据（Day 1 登记，Day 2 / 今晚批量沿用）：看**最终 checkpoint**（policy_349999104）按匹配回放的评测；
通过 = chord_sr > 0 且 maniptrans_sr > 0 且 ADD-AUC > 这条序列这种回放的不动基线（冒烟 rollout 用评测器顶点算的，bar_<冒烟>.txt）。
“> 0” 只靠 ≤ 3 个世界，或 ADD-AUC 落在基线 ±0.01 内 → 另标“边缘”。另两个 checkpoint 只报走势（gap = ADD-AUC − 基线）。
有效性闸（任何一条不满足 → 这条只列数、不判定）：三份评测都在、step_count = 期望帧数（逐帧 = 源帧数，真实时长 = round((N−1)·5/3)+1）、
基线帧数 = 期望帧数、评测输出里 non_finite_worlds 为空、结果头的来历不带 -dirty / -syncing、三份评测的 checkpoint sha256 两两不同。
用法：python3 stages/evaluate/batch_verdict.py <期望表> <结果目录> [--steps 349999104] [--bar-root <目录>]
  结果目录里要有 eval_<run>_s<步数>.txt 与 bar_<冒烟 run>.txt（例：results/ml40/day2、results/day2）。
  --steps：只看列出的步数（逗号分隔，最大的当最终 checkpoint）；Tier 2 批量只评最终 checkpoint（计划 agent-doc/plan-t2-batch-2026-10-01.md）。
  只有一个步数时没有走势，也不做“checkpoint sha256 两两不同”这一条。
  --bar-root：不动基线 bar_<冒烟 run>.txt 另放在这个目录（默认与评测结果同目录）。ep7 发散前 checkpoint 的评测记在 results/ep7/，
  不动基线仍在 results/day2/ 或 results/ml40/day2/（计划 agent-doc/plan-ep7-eval-and-guards-2026-10-02.md 的 A 部分）。
"""
import json
import re
import sys
from pathlib import Path

STEPS = (330012672, 340013056, 349999104)
RATES = ("chord_sr", "maniptrans_sr", "spider_sr_uncentered")


def load(path: Path) -> tuple[dict, str, str]:
    """返回 (summary_json, 来历, non_finite_worlds 原文)。"""
    text = path.read_text()
    summary = source = nonfinite = None
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("summary_json "):
            summary = json.loads(s[len("summary_json "):])
        elif s.startswith("# source"):
            source = s.split(":", 1)[1].strip()
    m = re.search(r"non_finite_worlds (\[[^\]]*\])", text)
    if m:
        nonfinite = m.group(1)
    if summary is None:
        raise SystemExit(f"{path}: 没有 summary_json 行")
    return summary, source or "", nonfinite


def expected_frames(playback: str, src: int) -> int:
    return src if playback == "frame" else int(round((src - 1) / 30.0 * 50.0)) + 1


def main() -> int:
    args = sys.argv[1:]
    steps = STEPS
    if "--steps" in args:
        i = args.index("--steps")
        steps = tuple(sorted(int(x) for x in args[i + 1].split(",") if x))
        del args[i:i + 2]
        if not steps:
            raise SystemExit("--steps 不能为空")
    bar_root = None
    if "--bar-root" in args:
        i = args.index("--bar-root")
        bar_root = Path(args[i + 1])
        del args[i:i + 2]
    expect, root = Path(args[0]), Path(args[1])
    bar_root = bar_root or root
    rows = [ln.split("\t") for ln in expect.read_text().splitlines()[1:] if ln.strip()]
    tally = {"通过": 0, "不通过": 0, "无效": 0}
    for run, smoke, seq, pb, src, *_ in rows:
        want = expected_frames(pb, int(src))
        problems = []
        bar_path = bar_root / f"bar_{smoke}.txt"
        evals = {s: root / f"eval_{run}_s{s}.txt" for s in steps}
        missing = [str(p) for p in [bar_path, *evals.values()] if not p.exists()]
        if missing:
            print(f"\n== {run}（{seq}，{pb}）：无效，缺 {', '.join(missing)}")
            tally["无效"] += 1
            continue
        bar, bar_src, _ = load(bar_path)
        if bar.get("frames") != want:
            problems.append(f"基线帧数 {bar.get('frames')}，期望 {want}")
        loaded = {s: load(p) for s, p in evals.items()}
        shas = set()
        for s, (m, source, nonfinite) in loaded.items():
            if m.get("step_count") != want:
                problems.append(f"第 {s} 步 step_count {m.get('step_count')}，期望 {want}")
            if nonfinite != "[]":
                problems.append(f"第 {s} 步 non_finite_worlds {nonfinite}")
            if re.search(r"-dirty|-syncing", source) or not source:
                problems.append(f"第 {s} 步来历 {source or '缺'}")
            shas.add(m.get("checkpoint_sha256"))
        if len(steps) > 1 and len(shas) != len(steps):
            problems.append(f"{len(steps)} 份评测的 checkpoint sha256 不是两两不同")
        final, _, _ = loaded[steps[-1]]
        w = final["world_count"]
        b = bar["add_auc"]
        gaps = [loaded[s][0]["add_auc"] - b for s in steps]
        print(f"\n== {run}（{seq}，{pb}，{want} 帧）：不动基线 {b:.4f}；gap 走势 {' / '.join(f'{g:+.3f}' for g in gaps)}")
        if problems:
            print("  无效：" + "；".join(problems))
            tally["无效"] += 1
            continue
        crit = [("chord_sr", final["chord_sr"], 0.0), ("maniptrans_sr", final["maniptrans_sr"], 0.0), ("add_auc", final["add_auc"], b)]
        marginal = []
        for k, v, lim in crit:
            extra = f"（{round(v * w)} / {w} 个世界）" if k in RATES else ""
            print(f"  {k:<14} {v:.4g} > {lim:.4g}{extra}：{'过' if v > lim else '不过'}")
            if k in RATES and v > 0 and round(v * w) <= 3:
                marginal.append(f"{k} 只靠 {round(v * w)} 个世界")
        if abs(final["add_auc"] - b) <= 0.01:
            marginal.append(f"ADD-AUC 与基线差 {final['add_auc'] - b:+.4f}")
        ok = all(v > lim for _, v, lim in crit)
        print(f"  另报：SP-SR {final['spider_sr_uncentered']:.4g}（{round(final['spider_sr_uncentered'] * w)} / {w}），MPPE {final['mppe_cm']:.2f} cm")
        print(f"  判定：{'通过' if ok else '不通过'}" + (f"（边缘：{'；'.join(marginal)}）" if marginal else ""))
        tally["通过" if ok else "不通过"] += 1
        print("  summary_json " + json.dumps({"run": run, "sequence_id": seq, "playback": pb, "frames": want, "bar": b,
                                              "add_auc": final["add_auc"], "gap": final["add_auc"] - b,
                                              "chord_sr": final["chord_sr"], "maniptrans_sr": final["maniptrans_sr"],
                                              "spider_sr_uncentered": final["spider_sr_uncentered"], "mppe_cm": final["mppe_cm"],
                                              "verdict": "PASS" if ok else "FAIL", "marginal": marginal}, ensure_ascii=False))
    print(f"\n合计：通过 {tally['通过']}、不通过 {tally['不通过']}、无效 {tally['无效']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
