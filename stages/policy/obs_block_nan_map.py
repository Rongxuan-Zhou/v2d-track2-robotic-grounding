"""把输入归一化统计量（486 维）里的非有限下标逐块对到 ReconBody 观测的排列上（Mac 上跑，只用标准库，只读已落盘的结果）。

输入：divergence_code_facts.sh 的输出（第 8 节，RECON_BODY_OBSERVATION_BLOCKS 的名字与宽度），
      ckpt_nonfinite.py --by-tensor 的输出（某个 [486] 张量的非有限下标区间）。
输出：每块的宽度、非有限个数、有限个数。哪些块只取自参考轨迹、哪些依赖仿真状态，由读者按块名判断，脚本不下结论。
用法：python3 stages/policy/obs_block_nan_map.py results/diverge/code_facts.txt results/diverge/ckpt_nonfinite_by_tensor_ep1.txt [张量名]
计划 agent-doc/plan-ep1-ep7-divergence-2026-09-30.md 的 C2b / C4。
"""
import re
import sys


def blocks(code_facts: str) -> list[tuple[str, int]]:
    out = []
    for line in open(code_facts, encoding="utf-8"):
        m = re.search(r'\("([^"]+)", (\d+)\),', line)
        if m and "envs/recon_body.py" in line:
            out.append((m.group(1), int(m.group(2))))
    return out


def bad_indices(by_tensor: str, tensor: str) -> set[int]:
    for line in open(by_tensor, encoding="utf-8"):
        m = re.search(rf"{re.escape(tensor)} \[486\]：非有限 \d+ 个，下标 (\S+)", line)
        if m:
            out = set()
            for part in m.group(1).split(","):
                a, _, b = part.partition("-")
                out.update(range(int(a), int(b or a) + 1))
            return out
    raise SystemExit(f"{by_tensor} 里没有 {tensor} [486] 的下标")


def main() -> None:
    if len(sys.argv) not in (3, 4):
        raise SystemExit("用法：obs_block_nan_map.py <code_facts.txt> <by_tensor.txt> [张量名，默认 state.0029]")
    tensor = sys.argv[3] if len(sys.argv) == 4 else "state.0029"
    bl = blocks(sys.argv[1])
    width = sum(w for _, w in bl)
    if width != 486:
        raise SystemExit(f"观测块宽度合计 {width}，不是 486")
    bad = bad_indices(sys.argv[2], tensor)
    print(f"{tensor}：486 维里非有限 {len(bad)} 个")
    print(f"{'下标':>9}  {'宽':>3}  {'非有限':>4}  {'有限':>4}  块")
    cursor = 0
    for name, w in bl:
        n = sum(1 for i in range(cursor, cursor + w) if i in bad)
        print(f"{cursor:>4}-{cursor + w - 1:<4}  {w:>3}  {n:>4}  {w - n:>4}  {name}")
        cursor += w


if __name__ == "__main__":
    main()
