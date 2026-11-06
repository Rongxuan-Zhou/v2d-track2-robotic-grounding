"""checkpoint 扫描的选点（训练机上、仓库根执行，只用标准库，只读 runs/flash_chord 下已落盘的 4096 世界评测）。
计划 agent-doc/plan-ep25-sweep-seeds-2026-10-18.md 第 2 节。

附加表沿用 day2_eval_chain.sh 的格式（tab 分隔，# 开头是注释）：eval <run> <步数> <回放> <帧数>。
一条评测“合格”：runs/flash_chord/<run>/eval_<步数>_w4096.json 在，world_count 是 4096，non_finite_worlds 是空列表，
step_count 等于附加表这条 run 的帧数（回放只许 frame），checkpoint 指向这条 run 的这一步，四项指标都是有限的数。

两个子命令：
  refine <粗扫表> <细扫表输出>：粗扫表里的每条 run，在它全部合格评测里选一步（选法见下），把 checkpoint 序列
    （policy_<步数>.safetensors 按步数排）里紧挨着它的前后各一份、还没有合格评测的，写成细扫表（同样的格式）。没有要补的就写空表。
  final <粗扫表> <导出清单输出> <MPPE 上限 cm>：按档位（run 名 t<档>-ep<N>-…）打印全部合格评测点，每档按同样的选法选一点；
    它的 MPPE 低于上限就写进导出清单（档位 episode run 步数，export_chain.sh 的格式），否则这一档不写。
选法（计划第 2.3 节 E1，复核第 4 条）：MPPE（metrics.mppe_cm）不超过最低值 + 1 cm 的点里取步数最大的，再按 MPPE、run 名。
早期 checkpoint 多半一碰不碰、都落在“物体不动”的 49–50 cm 附近，彼此只差噪声；这时取训练得更久的那一份，不让噪声挑出没学会走路的策略。
最后一行 CKPT_SWEEP_PICK_DONE。参数或表格不对时退出码 2。
"""

import json
import math
import re
import sys
from pathlib import Path

FC = Path("runs/flash_chord")
RUN_RE = re.compile(r"^t([123])-ep([0-9]+)-[a-z0-9_-]+$")
KEYS = (("AUC", "add_auc"), ("SP-SR", "spider_sr_uncentered"), ("MP-SR", "maniptrans_sr"), ("MPPE", "mppe_cm"))
TOL_CM = 1.0


def die(msg: str):
    print(f"ckpt_sweep_pick: {msg}", file=sys.stderr)
    sys.exit(2)


def read_table(path: str) -> dict:
    """附加表 → {run: 帧数}；同一条 run 的帧数必须一致。"""
    runs = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        f = line.split("\t")
        if len(f) != 5 or f[0] != "eval" or not RUN_RE.match(f[1]) or not f[2].isdigit() or f[3] != "frame" or not f[4].isdigit():
            die(f"{path} 这一行不对（要 eval <run> <步数> frame <帧数>）：{line!r}")
        if runs.setdefault(f[1], int(f[4])) != int(f[4]):
            die(f"{path} 里 {f[1]} 的帧数前后不一致")
    if not runs:
        die(f"{path} 没有行")
    return runs


def steps_of(run: str) -> list:
    return sorted(int(p.name[len("policy_"):-len(".safetensors")]) for p in (FC / run).glob("policy_*.safetensors")
                  if p.name[len("policy_"):-len(".safetensors")].isdigit())


def qualified(run: str, frames: int) -> dict:
    """{步数: {指标名: 值}}，只收合格的评测。"""
    out = {}
    for j in (FC / run).glob("eval_*_w4096.json"):
        m = re.fullmatch(r"eval_([0-9]+)_w4096\.json", j.name)
        if not m:
            continue
        step = int(m.group(1))
        try:
            e = json.loads(j.read_text())
        except (OSError, ValueError):
            continue
        nf, met = e.get("non_finite_worlds"), e.get("metrics") or {}
        vals = {k: met.get(k) for _, k in KEYS}
        if (e.get("world_count") != 4096 or not isinstance(nf, list) or nf or e.get("step_count") != frames
                or not str(e.get("checkpoint", "")).endswith(f"runs/flash_chord/{run}/policy_{step}.safetensors")
                or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals.values())):
            continue
        out[step] = vals
    return out


def choose(points: list) -> tuple:
    """points：[(run, 步数, 指标)]。MPPE 不超过最低值 + TOL_CM 的点里取步数最大的，再按 MPPE、run 名。"""
    low = min(v["mppe_cm"] for _, _, v in points)
    near = [p for p in points if p[2]["mppe_cm"] <= low + TOL_CM]
    return min(near, key=lambda p: (-p[1], p[2]["mppe_cm"], p[0]))


def refine(table: str, out: str):
    rows = []
    for run, frames in read_table(table).items():
        q, steps = qualified(run, frames), steps_of(run)
        if not q:
            print(f"{run}：没有合格评测，不细扫")
            continue
        best = choose([(run, s, q[s]) for s in q])[1]
        add = []
        if best in steps:
            i = steps.index(best)
            add = [s for s in (steps[i - 1] if i > 0 else None, steps[i + 1] if i + 1 < len(steps) else None) if s is not None and s not in q]
        print(f"{run}：合格评测 {len(q)} 个，选第 {best} 步（{q[best]['mppe_cm']:.2f} cm，最低 {min(v['mppe_cm'] for v in q.values()):.2f} cm）；细扫补 {add if add else '无'}")
        rows += [f"eval\t{run}\t{s}\tframe\t{frames}" for s in add]
    Path(out).write_text("".join(r + "\n" for r in rows))
    print(f"细扫表 {out}：{len(rows)} 行")


def final(table: str, out: str, cap: float):
    by_tier = {}
    for run, frames in read_table(table).items():
        t, ep = RUN_RE.match(run).groups()
        for step, v in qualified(run, frames).items():
            by_tier.setdefault(int(t), []).append((run, int(ep), step, v))
    rows = []
    for t in sorted(by_tier):
        pts = sorted(by_tier[t], key=lambda p: (p[0], p[2]))
        print(f"== Tier {t}：合格评测点 {len(pts)} 个（4096 世界）")
        for run, ep, step, v in pts:
            print(f"  {run} 第 {step} 步：" + "  ".join(f"{n} {v[k]:.4f}" for n, k in KEYS))
        run, step, v = choose([(p[0], p[2], p[3]) for p in pts])
        ep = int(RUN_RE.match(run).group(2))
        ok = v["mppe_cm"] < cap
        print(f"Tier {t} 本机选点：{run} 第 {step} 步，MPPE {v['mppe_cm']:.2f} cm（本档最低 {min(p[3]['mppe_cm'] for p in pts):.2f} cm，最低 + {TOL_CM:g} cm 内取步数最大；上限 {cap:g} cm，{'导出' if ok else '不导出'}）")
        if ok:
            rows.append(f"{t}\t{ep}\t{run}\t{step}")
    Path(out).write_text("".join(r + "\n" for r in rows))
    print(f"导出清单 {out}：{len(rows)} 行")


def main():
    a = sys.argv[1:]
    if len(a) == 3 and a[0] == "refine":
        refine(a[1], a[2])
    elif len(a) == 4 and a[0] == "final":
        try:
            cap = float(a[3])
        except ValueError:
            die(f"MPPE 上限不是数：{a[3]}")
        final(a[1], a[2], cap)
    else:
        die("用法：ckpt_sweep_pick.py refine <粗扫表> <细扫表输出> | final <粗扫表> <导出清单输出> <MPPE 上限 cm>")
    print("CKPT_SWEEP_PICK_DONE")


if __name__ == "__main__":
    main()
