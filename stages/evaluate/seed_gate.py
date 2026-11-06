"""补下一个 seed 之前的闸（训练机上、仓库根执行，只用标准库，只读已落盘的评测与导出）。计划 agent-doc/plan-ep25-sweep-seeds-2026-10-18.md 第 3 节。

补 seed 是为了让导出那 1 局在某一项上成功（ep14 看 MP-SR，ep11 看 SP-SR）。上一个 seed 的结果决定下一个 seed 还值不值得训：
  - 导出那 1 局这一项已经成功：目的已达到，不再补（FAIL，原因“导出已成功”）；
  - 4096 世界这一项的成功率低于给定下限：这一集用这个配方补 seed 不划算，不再补（FAIL，原因“4096 世界太低”）；
  - 读不到合格的 4096 世界评测（world_count 4096、non_finite_worlds 空列表、checkpoint 是这一步）或导出 JSON：不补（FAIL）；
  - 其余（4096 世界够、导出那 1 局失败）：补（PASS，退出码 0）。
用法：python3 stages/evaluate/seed_gate.py <run> <episode> <指标 maniptrans_sr|spider_sr_uncentered> <4096 世界下限> [步数，默认 349999104]
最后一行 SEED_GATE PASS 或 SEED_GATE FAIL <原因>；PASS 退出码 0，FAIL 退出码 1，参数不对退出码 2。
"""

import json
import sys
from pathlib import Path


def main():
    a = sys.argv[1:]
    if len(a) not in (4, 5) or a[2] not in ("maniptrans_sr", "spider_sr_uncentered") or not a[1].isdigit():
        print("用法：seed_gate.py <run> <episode> maniptrans_sr|spider_sr_uncentered <4096 世界下限> [步数]", file=sys.stderr)
        sys.exit(2)
    run, ep, key, low = a[0], int(a[1]), a[2], float(a[3])
    step = a[4] if len(a) == 5 else "349999104"
    o = Path("runs/flash_chord") / run

    def fail(why: str):
        print(f"SEED_GATE FAIL {why}")
        sys.exit(1)

    try:
        e = json.loads((o / f"eval_{step}_w4096.json").read_text())
        x = json.loads((o / "export_8c778f9" / f"s{step}_w1" / f"episode_{ep:06d}.json").read_text())
    except (OSError, ValueError) as err:
        fail(f"读不到评测或导出：{type(err).__name__} {err}")
    nf = e.get("non_finite_worlds")
    if e.get("world_count") != 4096 or not isinstance(nf, list) or nf or not str(e.get("checkpoint", "")).endswith(f"{run}/policy_{step}.safetensors"):
        fail("4096 世界评测不合格（世界数、非有限世界或 checkpoint 不对）")
    if x.get("world_count") != 1 or not str(x.get("checkpoint", "")).endswith(f"{run}/policy_{step}.safetensors"):
        fail("导出 JSON 不是这一步的 1 世界导出")
    w, v = float(e["metrics"][key]), float(x["metrics"][key])
    print(f"{run}：4096 世界 {key} {w:.4f}（下限 {low:g}），导出那 1 局 {key} {v:g}")
    if v >= 0.5:
        fail("导出已成功")
    if w < low:
        fail("4096 世界太低")
    print("SEED_GATE PASS")


if __name__ == "__main__":
    main()
