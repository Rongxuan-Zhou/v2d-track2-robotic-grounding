"""按计划 agent-doc/plan-t3-pilot-v2-2026-10-10.md 第 4 节登记的规则，从两版校正的悬空复查记录里选版（Mac 上跑，只用标准库）。

输入：<名>=<记录> 若干，记录是 stages/recovery/t3_float_facts.py <目录> --all-rest --offset-gate 30 经 scripts/dev/record 落盘的输出。
规则（Fable 5.1 审查 m3 之后定，执行前登记）：
  1. 只看打了 OFFSET_GATE_PASS 的版；一版也没有，打 PICK_NONE。
  2. 只有一版过就选它。两版都过：| |o_a| − |o_b| | > 2·√(SE_a² + SE_b²) 时选 |o| 小的；否则看 ep8、ep24 两集间隙的绝对值，
     取两集里较大的那个，选它更小的那版（仍相同就按参数顺序取前一版）。
  3. 选出的版里 ep8、ep24 的间隙绝对值都 ≤ 50 mm，打 PICK <名>（B 段照做）；否则打 PICK_BLOCKED <名>（B 段不做）。
每版先打一行：o、标准误、每集标准差、地面上的集数，以及 ep8、ep24 的间隙与 parquet sha256 前 12 位（B 段核对同一份文件用）。
用法：python3 stages/recovery/t3_pick_version.py floor=results/t3float/gate_floor_ok20_mf4090.txt f830=results/t3float/gate_f830_ok20_mf4090.txt
测试：stages/recovery/test_t3_pick_version.py。
"""
import math
import re
import sys
from pathlib import Path

GATE = re.compile(r"^OFFSET_GATE_(PASS|FAIL|NONE)（地面上 (\d+) 集(?:，常数模型 o = ([+-]?\d+\.\d) ± (\d+\.\d) mm，每集标准差 (\d+\.\d) mm)?")
EPS = (8, 24)
EP_GATE_MM = 50.0


def parse(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not any(x.startswith("T3_FLOAT_FACTS_DONE") for x in lines):
        raise SystemExit(f"t3_pick_version: {path} 没有结束标记 T3_FLOAT_FACTS_DONE")
    gates = [m for m in map(GATE.match, lines) if m]
    if len(gates) != 1:
        raise SystemExit(f"t3_pick_version: {path} 里 OFFSET_GATE 行有 {len(gates)} 行，应当恰好 1 行")
    g = gates[0]
    out = {"verdict": g.group(1), "n_floor": int(g.group(2))}
    if g.group(3) is not None:
        out.update(o=float(g.group(3)), se=float(g.group(4)), sd=float(g.group(5)))
    for x in lines:
        c = x.split("\t")
        if len(c) == 15 and c[0].isdigit() and int(c[0]) in EPS and c[7]:   # 窗口里没有可见帧的那种行，间隙列是空的
            out[int(c[0])] = (float(c[7]), c[14])
    missing = [e for e in EPS if e not in out]
    if missing:
        raise SystemExit(f"t3_pick_version: {path} 里没有 ep{missing} 的间隙")
    return out


def pick(vs: dict) -> tuple:
    """vs：名 → parse() 的结果（保持参数顺序）。返回 (结论, 名或 None, 说明)。"""
    ok = [k for k, v in vs.items() if v["verdict"] == "PASS"]
    if not ok:
        return "PICK_NONE", None, "两版都没有过 OFFSET_GATE"
    if len(ok) == 1:
        name, why = ok[0], "只有这一版过 OFFSET_GATE"
    else:
        a, b = ok[:2]
        d = abs(abs(vs[a]["o"]) - abs(vs[b]["o"]))
        lim = 2 * math.sqrt(vs[a]["se"] ** 2 + vs[b]["se"] ** 2)
        if d > lim:
            name = min((a, b), key=lambda k: abs(vs[k]["o"]))
            why = f"两版都过，|o| 之差 {d:.1f} mm > 2·√(SE²+SE²) = {lim:.1f} mm，取 |o| 小的"
        else:
            worst = {k: max(abs(vs[k][e][0]) for e in EPS) for k in (a, b)}
            name = a if worst[a] <= worst[b] else b
            why = (f"两版都过，|o| 之差 {d:.1f} mm ≤ {lim:.1f} mm，看 ep8、ep24 间隙绝对值的较大者"
                   f"（{a} {worst[a]:.1f} mm，{b} {worst[b]:.1f} mm），取小的")
    gaps = [abs(vs[name][e][0]) for e in EPS]
    if max(gaps) <= EP_GATE_MM:
        return "PICK", name, why + f"；ep8、ep24 的间隙绝对值 {gaps[0]:.1f}、{gaps[1]:.1f} mm 都 ≤ {EP_GATE_MM:.0f} mm"
    return "PICK_BLOCKED", name, why + f"；但 ep8、ep24 的间隙绝对值 {gaps[0]:.1f}、{gaps[1]:.1f} mm 有超过 {EP_GATE_MM:.0f} mm 的，B 段不做"


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    vs = {}
    for a in args:
        name, sep, path = a.partition("=")
        if not sep or not re.fullmatch(r"[a-z0-9]+", name) or name in vs:
            raise SystemExit(f"t3_pick_version: 参数要写成 <名>=<记录>，名只许小写字母与数字、不重复：{a}")
        vs[name] = parse(Path(path))
    if not vs:
        raise SystemExit("用法：t3_pick_version.py <名>=<记录> ...")
    for k, v in vs.items():
        head = (f"o = {v['o']:+.1f} ± {v['se']:.1f} mm，每集标准差 {v['sd']:.1f} mm" if "o" in v else "没有常数模型")
        eps = "，".join(f"ep{e} 间隙 {v[e][0]:+.1f} mm（{v[e][1]}）" for e in EPS)
        print(f"{k}：OFFSET_GATE_{v['verdict']}，地面上 {v['n_floor']} 集，{head}；{eps}")
    verdict, name, why = pick(vs)
    print(why)
    print(verdict if name is None else f"{verdict} {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
