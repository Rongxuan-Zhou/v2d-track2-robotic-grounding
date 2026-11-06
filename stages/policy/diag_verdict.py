"""按计划附录 A.3（第 5 版，登记）给 ml40 诊断对照下判定（Mac 上跑，只用标准库，只读已落盘的结果）。

计划：agent-doc/plan-ep1-ep7-divergence-2026-09-30.md 附录 A。输入都是 scripts/dev/record 落盘的文件，每类可给多份，后给的覆盖先给的同名 run：
  --summary  train_nan_scan.py --table 的输出（run | 平均回合长度 | 平均回报 | loss_scale | 摘要有 nan）
  --ckpt     ckpt_nonfinite.py 的输出（每个 run 各份 checkpoint 的非有限参数个数）
  --gate     diag_importer_gate.py 的输出（每条 run 有效 / 无效 / 未跑）
单条“发散”（A.3）：冒烟 = 摘要有 nan 或最终 checkpoint 有非有限参数；截断的 run（名字带 -t60m-）= 任何一份 checkpoint 非有限。
导入器核对不是“有效”的 run 不计数；缺摘要或缺 checkpoint 扫描的记“缺数据”，也不计数。
分组按 run 名：R0 = ep7 cap10m v040 种子 42；R1 = ep7 v040 fp16；R3′ = ep7 p185 fp16；R2 = ep7 v040 fp32；S = ep1 / ep28 p185 冒烟；
L0–L3 = -t60m-（L0 ep1 v040 fp16，L1 ep1 p185 fp16，L2 ep28 p185 fp16，L3 ep1 v040 fp32）。
检验：单侧 Fisher 精确检验（备择：R1 的发散比例更高），用超几何分布的尾概率。R1 对 R3′、R1 对 R2 只用两组都有的种子（队列按种子配对）。
用法：python3 stages/policy/diag_verdict.py --summary <文件>... --ckpt <文件>... --gate <文件>...
"""
import argparse
import re
from math import comb

SMOKE = re.compile(r"diag-t1-ep7-cap4m-(v040|p185)-(fp16|fp32)-s(\d+)-ml40$")
GROUP = {("v040", "fp16"): "R1", ("p185", "fp16"): "R3′", ("v040", "fp32"): "R2"}
R0 = "diag-t1-ep7-cap10m-v040-s42-ml40"
S = {"ep1": "diag-t1-ep1-cap4m-p185-fp16-s42-ml40", "ep28": "diag-t1-ep28-cap4m-p185-fp16-s42-ml40"}
L = {"L0": "diag-t1-ep1-cap4m-v040-fp16-s42-t60m-ml40", "L1": "diag-t1-ep1-cap4m-p185-fp16-s42-t60m-ml40",
     "L2": "diag-t1-ep28-cap4m-p185-fp16-s42-t60m-ml40", "L3": "diag-t1-ep1-cap4m-v040-fp32-s42-t60m-ml40"}
L_FINAL = 60002304
LEN_MIN, RET_MIN = 64.0, 2.37   # 非方盒冒烟的最低回合长度与回报（A.0）


def num(text: str) -> float:
    try:
        return float(text)
    except ValueError:   # 摘要里缺这一项时表里是 “—”
        return float("nan")


def read_summary(paths: list) -> dict:
    out = {}
    for p in paths:
        for line in open(p, encoding="utf-8"):
            f = [x.strip() for x in line.split(" | ")]
            if len(f) == 5 and f[0].startswith("diag-"):
                out[f[0]] = {"len": num(f[1]), "ret": num(f[2]), "nan": f[4] == "是"}
    return out


def read_ckpt(paths: list) -> dict:
    out, run = {}, None
    for p in paths:
        for line in open(p, encoding="utf-8"):
            m = re.match(r"^== (\S+?)（policy", line)
            if m:
                run = m.group(1)
                out[run] = {}
                continue
            m = re.match(r"^\s+([\d,]+) 步：非有限 (\d+) /", line)
            if m and run:
                out[run][int(m.group(1).replace(",", ""))] = int(m.group(2))
    return out


def read_gate(paths: list) -> dict:
    out = {}
    for p in paths:
        for line in open(p, encoding="utf-8"):
            m = re.match(r"^\s+(diag-\S+)\s+导入器 .*→ (\S+)$", line)
            if m:
                out[m.group(1)] = m.group(2)
    return out


def fisher_greater(a: int, n1: int, c: int, n2: int) -> float:
    """P(X ≥ a)，X ~ 超几何(总数 n1+n2，发散 a+c，抽 n1)：两组合计 a+c 条发散时，R1 这边至少 a 条的概率。"""
    k, n = a + c, n1 + n2
    return sum(comb(k, x) * comb(n - k, n1 - x) for x in range(a, min(k, n1) + 1)) / comb(n, n1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", nargs="+", required=True)
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--gate", nargs="+", required=True)
    a = ap.parse_args()
    summ, ckpt, gate = read_summary(a.summary), read_ckpt(a.ckpt), read_gate(a.gate)

    def smoke_status(run: str) -> tuple:
        """(是否计数, 是否发散, 说明)。"""
        if gate.get(run) != "有效":
            return False, None, f"导入器核对：{gate.get(run, '缺')}"
        if run not in summ or not ckpt.get(run):
            return False, None, "缺数据（摘要或 checkpoint 扫描）"
        s, last = summ[run], max(ckpt[run])
        bad = ckpt[run][last]
        div = s["nan"] or bad > 0
        return True, div, f"摘要 nan {'是' if s['nan'] else '否'}，最终 checkpoint 非有限 {bad}，回合 {s['len']:.1f}，回报 {s['ret']:.2f}"

    print("== 各条冒烟（导入器核对有效、数据齐全才计数）")
    groups = {"R1": {}, "R3′": {}, "R2": {}}
    for run in sorted(set(summ) | set(ckpt) | set(gate)):
        m = SMOKE.match(run)
        if not m:
            continue
        g, seed = GROUP[(m.group(1), m.group(2))], int(m.group(3))
        counted, div, note = smoke_status(run)
        print(f"  {g:3} 种子 {seed}：{'发散' if div else ('未发散' if counted else '不计数')}（{note}）")
        if counted:
            groups[g][seed] = (div, summ[run])

    counted, div, note = smoke_status(R0)
    print(f"\n== R0 {R0}：{note}")
    print("  → 发散，可复现" if div else ("  → 未发散：不能稳定复现（GPU 非确定性），不影响下面" if counted else "  → 不计数"))

    def pair(g: str) -> tuple:
        seeds = sorted(set(groups["R1"]) & set(groups[g]))
        k1 = sum(groups["R1"][s][0] for s in seeds)
        k2 = sum(groups[g][s][0] for s in seeds)
        return seeds, k1, k2, len(seeds)

    r1_seeds = sorted(groups["R1"])
    k = sum(groups["R1"][s][0] for s in r1_seeds)
    n = len(r1_seeds)
    print(f"\n== H3（R1 发散 {k} / {n}，种子 {r1_seeds}）")
    h3 = None
    if n == 4:
        h3 = "拒绝" if k >= 2 else ("补种子" if k == 1 else "停")
    elif n == 8:
        h3 = "拒绝" if k >= 3 else "不拒绝"
    print({"拒绝": "  → 拒绝 H3", "补种子": "  → R1 = 1/4：按 A.3 跑 ⑤⑥ 补到 8 个种子", "停": "  → R1 = 0/4：停；R2、R3′ 只描述",
           "不拒绝": "  → 不拒绝 H3", None: f"  → 种子数 {n} 不是登记的 4 或 8，只报数"}[h3])

    for g, name, ext in (("R3′", "H1′", "⑤"), ("R2", "H4", "⑥")):
        seeds, k1, k2, m = pair(g)
        p = fisher_greater(k1, m, k2, m) if m else float("nan")
        print(f"\n== {name}：R1 {k1} / {m} 对 {g} {k2} / {m}（配对种子 {seeds}），单侧 Fisher p = {p:.4f}")
        if h3 != "拒绝":
            print("  → H3 未被拒绝，按 A.3 不判")
            continue
        neg = (m == 4 and k2 >= 2) or (m == 8 and k2 >= 3)
        if g == "R2" and neg:   # H4 说 fp16 存储是必要条件：fp32 也照样死就拒绝，不看 p
            verdict = f"拒绝 H4（{g} {k2} / {m}：fp32 也会死）"
        elif p < 0.05:
            verdict = "支持"
        elif m == 4:
            verdict = f"p ≥ 0.05、4 个种子 → 按 A.3 跑 {ext} 补到 8 个再检验"
        elif m == 8:
            verdict = "不支持"
        else:
            verdict = f"配对种子数 {m} 不是登记的 4 或 8，只报数"
        print(f"  → {name}：{verdict}")
        if g == "R3′":
            print(f"  否定结论（R3′ ≥ 2/4 或 ≥ 3/8 → 仅修导入挡不住发散）：{'成立' if neg else '不成立'}（R3′ {k2} / {m}）")
            ok = sum(1 for s in seeds if groups[g][s][1]["len"] >= LEN_MIN and groups[g][s][1]["ret"] >= RET_MIN)
            print(f"  次要读数：R3′ 回合长度 ≥ {LEN_MIN:g} 且回报 ≥ {RET_MIN} 的 {ok} / {m}")

    print("\n== S（#185 冒烟）")
    for ep, run in S.items():
        counted, div, note = smoke_status(run)
        if ep == "ep1" and counted:
            s = summ[run]
            ok = not div and s["len"] >= LEN_MIN and s["ret"] >= RET_MIN
            print(f"  ep1：{note} → {'与非方盒冒烟一致' if ok else '与非方盒冒烟不一致（如实记）'}")
        else:
            print(f"  {ep}：{note}{' → 只描述，不判假设' if ep == 'ep28' else ''}")

    print("\n== L 组（L40S、单种子，只作旁证；发散 = 6000 万步内任何一份 checkpoint 非有限）")
    ldiv = {}
    for name, run in L.items():
        if gate.get(run) != "有效":
            print(f"  {name}：导入器核对 {gate.get(run, '缺')}，不计")
            continue
        steps = ckpt.get(run, {})
        if L_FINAL not in steps:
            print(f"  {name}：未完成或缺扫描（已扫 {len(steps)} 份）")
            continue
        bad = [s for s in sorted(steps) if steps[s] > 0]
        ldiv[name] = bool(bad)
        print(f"  {name}：{'发散，第一份非有限 ' + format(bad[0], ',') + ' 步' if bad else '6 份 checkpoint 全部有限'}")
    if "L0" in ldiv and "L1" in ldiv:
        if ldiv["L1"]:
            print("  L0 / L1 → 与“仅修导入不够”一致")
        elif ldiv["L0"]:
            print("  L0 发散、L1 不发散 → 与 H1′ 一致")
        else:
            print("  L0、L1 都不发散 → 这个种子在 L40S 上没复现，不下结论")
    if "L0" in ldiv and "L3" in ldiv and ldiv["L0"]:
        print("  L3 → " + ("都发散：fp32 单独不够" if ldiv["L3"] else "L0 发散而 L3 不发散：与 H4 一致"))
    print("  次要读数（第 1 段 35–60M 平均回合长度）另见 wandb_local_history.py --by-stage")


if __name__ == "__main__":
    main()
