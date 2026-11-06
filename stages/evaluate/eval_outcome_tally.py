"""数已落盘的 4096 世界评测：多少份成功、其中多少份 MP-SR 为 0，以及只有地面的序列的判定（Mac 上跑，只用标准库，只读 results/）。

给“评测器遇到被判终止的世界就报错”这件事当证据（决定记录 agent-doc/decision-organizer-q1-q3-2026-10-01.md）：评测器只要有一个世界被判终止
就整条报错，所以一份 4096 世界的评测成功，说明这 4096 个世界没有一个被判终止。MP-SR 为 0 的评测也成功，说明任务失败不会触发终止。
另按 results/day2/empty_support_tier1.txt 把批量判定里的 run 分成“只有地面”（训练加 scene.support=false）与其他，各列判定与 MP-SR。
用法：python3 stages/evaluate/eval_outcome_tally.py
  落盘：scripts/dev/record submission/eval_outcome_tally python3 stages/evaluate/eval_outcome_tally.py
"""
import glob
import json
import re
from pathlib import Path


def summary(path: str):
    text = Path(path).read_text(errors="replace")
    m = re.search(r"^\s*summary_json (\{.*\})\s*$", text, flags=re.M)
    nf = re.search(r"non_finite_worlds (\[[^\]]*\])", text)
    return (json.loads(m.group(1)) if m else None), (nf.group(1) if nf else None)


def main() -> None:
    files = sorted(glob.glob("results/day2/eval_*.txt") + glob.glob("results/ml40/day2/eval_*.txt"))
    ok = zero = worlds = nonfinite = 0
    worst = (0.0, "")
    for f in files:
        s, nf = summary(f)
        if s is None or "maniptrans_sr" not in s:
            continue
        ok += 1
        worlds += int(s.get("world_count", 0))
        nonfinite += nf not in (None, "[]")
        if s["maniptrans_sr"] == 0:
            zero += 1
        if s.get("mppe_cm", 0) > worst[0]:
            worst = (s["mppe_cm"], Path(f).name)
    print(f"评测结果文件 {len(files)} 个，其中成功出分的 {ok} 份，合计 {worlds} 个世界；non_finite_worlds 非空的 {nonfinite} 份")
    print(f"成功出分的评测里 MP-SR 为 0 的 {zero} 份；MPPE 最大的一份 {worst[0]:.1f} cm（{worst[1]}）")

    empty = {l.strip() for l in Path("results/day2/empty_support_tier1.txt").read_text().splitlines() if l.strip() and not l.startswith("#")}
    seen = {}
    for f in sorted(glob.glob("results/day2/batch_verdict_*.txt")):
        for line in Path(f).read_text(errors="replace").splitlines():
            if "summary_json " in line:
                j = json.loads(line.split("summary_json ", 1)[1])
                if j.get("playback") == "frame" and j.get("verdict") in ("PASS", "FAIL"):
                    seen[j["run"]] = j
    floor = [j for j in seen.values() if j["sequence_id"] in empty]
    other = [j for j in seen.values() if j["sequence_id"] not in empty]
    for name, rows in (("只有地面（scene.support=false）", floor), ("有支撑面", other)):
        seqs = {j["sequence_id"] for j in rows}
        p = sum(j["verdict"] == "PASS" for j in rows)
        hi = sorted((j["maniptrans_sr"] for j in rows), reverse=True)[:5]
        print(f"{name}：逐帧 run {len(rows)} 条（{len(seqs)} 条序列），通过 {p} 条；MP-SR 最高的 5 个 {', '.join(f'{x:.3f}' for x in hi)}")


if __name__ == "__main__":
    main()
