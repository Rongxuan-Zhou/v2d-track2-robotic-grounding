"""按事先登记的规则挑 Day 2 试点序列（只用标准库，读 results/ 下的文本）。

规则（agent-doc/plan-day2-2026-09-28.md 第 2 版，设计审查第 3 条）：
  1. 排除 ep20（已做）、ep29（两只手都没有接触帧）、ep1（支撑面差 −22.5 mm，重定向检查 ALIGN）。
  2. 其余按帧数升序，均分三档（短 / 中 / 长）；每档取帧数离该档中位数最近的一条（并列取帧数小的）。
  3. 三条若全是“只有地面”（支撑面文件没有 Cube/Cylinder）或全有支撑面，把长档那条换成长档里支撑面类型相反、帧数离中位数最近的一条。
另外打出每条的真实时长帧数 round((N − 1) × 5/3) + 1。
用法：python3 stages/policy/pick_pilot_sequences.py results/day2/sequence_stats_tier1.txt results/batch_summary_v040.tsv <空支撑面清单> [已选的 ep，逗号分隔]
  第 4 个参数给出时，把这些 ep 也排除，同一规则在剩下的序列上再挑一轮（Day 2 改为“固定逐帧”后补跑的序列，见执行记录）。
  空支撑面清单：每行一个 sequence_id（由 palm_frame_probe.py 的 EMPTY_SUPPORT 行或直接查 usda 得到）。
"""
import sys
from pathlib import Path

EXCLUDE = {20: "已做", 29: "没有接触帧", 1: "支撑面差 −22.5 mm"}


def main() -> int:
    stats_txt, summary_tsv, empty_txt = sys.argv[1:4]
    ep_of, frames = {}, {}
    for line in Path(summary_tsv).read_text().splitlines()[1:]:
        f = line.split("\t")
        if f[0] == "1":
            ep_of[f[2]] = int(f[1])
    for line in Path(stats_txt).read_text().splitlines():
        f = line.split("\t")
        if len(f) >= 12 and f[1].isdigit():
            frames[f[0]] = int(f[1])
    empty = {ln.strip() for ln in Path(empty_txt).read_text().splitlines() if ln.strip()}
    done = {int(x) for x in sys.argv[4].split(",")} if len(sys.argv) > 4 and sys.argv[4] else set()
    rows = sorted((frames[s], ep_of[s], s) for s in frames if ep_of.get(s) not in EXCLUDE and ep_of.get(s) not in done)
    print(f"候选 {len(rows)} 条（排除 {', '.join(f'ep{e}（{w}）' for e, w in EXCLUDE.items())}"
          + (f"；另排除已选的 {', '.join(f'ep{e}' for e in sorted(done))}" if done else "") + "）")
    k = len(rows) // 3
    tiers = [rows[:k], rows[k:2 * k], rows[2 * k:]]
    picks = []
    for name, tier in zip(("短", "中", "长"), tiers):
        med = sorted(r[0] for r in tier)[len(tier) // 2]
        pick = min(tier, key=lambda r: (abs(r[0] - med), r[0]))
        print(f"{name}档 {tier[0][0]}–{tier[-1][0]} 帧，中位 {med}：选 ep{pick[1]} {pick[2]}（{pick[0]} 帧）")
        picks.append(pick)
    kinds = {p[2] in empty for p in picks}
    if len(kinds) == 1:
        want_empty = not kinds.pop()
        tier = tiers[2]
        med = sorted(r[0] for r in tier)[len(tier) // 2]
        alt = min((r for r in tier if (r[2] in empty) == want_empty), key=lambda r: (abs(r[0] - med), r[0]))
        print(f"三条支撑面类型相同，长档换成 ep{alt[1]} {alt[2]}（{alt[0]} 帧，{'只有地面' if want_empty else '有支撑面'}）")
        picks[2] = alt
    print("结果：")
    for n, ep, s in picks:
        print(f"  ep{ep}\t{s}\t{n} 帧\t真实时长 {round((n - 1) * 5 / 3) + 1} 帧\t{'只有地面（scene.support=false）' if s in empty else '有支撑面'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
