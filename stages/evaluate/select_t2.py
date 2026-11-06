"""Tier 2 按乙逐条选 run（Mac 上跑，只用标准库，只读已落盘的结果）。计划 agent-doc/plan-t2-select-b-2026-10-12.md 第 2 节第 2 步。

规则（乙，10/12 定，agent-doc/plan-next-training-2026-10-12.md 第 4a 节第 1 条）：每条 episode 在有资格的候选里，
4096 世界 MP-SR（summary_json 的 maniptrans_sr，不四舍五入）高者 → ADD-AUC 高者 → 现在交的那条 → run 名字典序。
资格：在给定的导出结果表里，档位 2、步数等于 --step、状态 ok 的行恰好一行（export_chain.sh 判 ok 时已核过导出与 4096 世界评测是同一个
checkpoint）；有评测记录 eval_<run>_s<步数>.txt，其中 summary_json 的 checkpoint 属于这条 run、步数等于 --step。
现在交的 run 必须有资格，否则报错退出（不然选择会悄悄丢掉它）。评测对照的是 v5.4 参考，不是干净真值；不读 Tier 1。
用法：python3 stages/evaluate/select_t2.py --current <清单.tsv>… --tables <结果表.tsv>… --evals <目录>… [--step 349999104] --out-tsv <选择清单.tsv>
输出：选择清单每行“2 episode run 步数 机器”；标准输出逐条打印候选与选中者，最后一行 SELECT_T2_DONE。
"""
import argparse
import json
import re
import sys
from pathlib import Path


def current_runs(files) -> dict:
    out = {}
    for f in files:
        for line in Path(f).read_text().splitlines():
            c = line.split("\t")
            if len(c) >= 4 and c[0] == "2":
                ep = int(c[1])
                if ep in out:
                    raise SystemExit(f"select_t2: 现在交的清单里 ep{ep} 不止一行（{out[ep]}、{c[2]}）")
                out[ep] = c[2]
    return out


def ok_rows(files, step: str) -> dict:
    """run → [(表名, 行)]：档位 2、这一步数、状态 ok 的行。"""
    out = {}
    for f in files:
        for line in Path(f).read_text().splitlines():
            c = line.split("\t")
            if len(c) == 9 and c[0] == "2" and c[3] == step and c[4] == "ok":
                out.setdefault(c[2], []).append((Path(f).name, c))
    return out


def eval_of(dirs, run: str, step: str):
    for d in dirs:
        p = Path(d) / f"eval_{run}_s{step}.txt"
        if p.exists():
            m = re.search(r"summary_json (\{.*\})", p.read_text(encoding="utf-8"))
            if not m:
                return None, f"{p} 里没有 summary_json 行"
            j = json.loads(m.group(1))
            ck = Path(j["checkpoint"])
            if ck.parent.name != run or str(j["environment_steps"]) != step:
                return None, f"{p} 的 checkpoint 是 {ck}、步数 {j['environment_steps']}，不是 {run} 的 {step}"
            return {"mp": float(j["maniptrans_sr"]), "auc": float(j["add_auc"]), "file": str(p)}, None
    return None, "没有评测记录"


def choose(cands: list, current: str) -> dict:
    """乙：MP-SR 高 → ADD-AUC 高 → 现在交的那条 → run 名字典序。"""
    return sorted(cands, key=lambda c: (-c["mp"], -c["auc"], c["run"] != current, c["run"]))[0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", nargs="+", required=True)
    ap.add_argument("--tables", nargs="+", required=True)
    ap.add_argument("--evals", nargs="+", required=True)
    ap.add_argument("--step", default="349999104")
    ap.add_argument("--out-tsv", required=True)
    a = ap.parse_args(argv)
    cur = current_runs(a.current)
    if sorted(cur) != list(range(30)):
        raise SystemExit(f"select_t2: 现在交的清单里 Tier 2 的 episode 不是 0–29：{sorted(cur)}")
    rows = ok_rows(a.tables, a.step)
    by_ep, skipped = {}, []
    for run, hits in sorted(rows.items()):
        if len(hits) != 1:
            skipped.append(f"{run}：状态 ok 的行有 {len(hits)} 行（{', '.join(h[0] for h in hits)}）")
            continue
        ev, why = eval_of(a.evals, run, a.step)
        if ev is None:
            skipped.append(f"{run}：{why}")
            continue
        ep = int(hits[0][1][1])
        mach = run.rsplit("-", 1)[1]
        by_ep.setdefault(ep, []).append({"run": run, "ep": ep, "machine": mach, "table": hits[0][0], **ev})
    for line in skipped:
        print(f"  没有资格：{line}")
    out, changed, n_r3a = [], [], 0
    for ep in range(30):
        cands = by_ep.get(ep, [])
        if cur[ep] not in {c["run"] for c in cands}:
            raise SystemExit(f"select_t2: ep{ep} 现在交的 {cur[ep]} 没有资格（见上面“没有资格”），停")
        c = choose(cands, cur[ep])
        for o in sorted(cands, key=lambda x: x["run"]):
            mark = "→" if o is c else " "
            print(f"  ep{ep:<3}{mark} {o['run']}：MP-SR {o['mp']:.6f}  ADD-AUC {o['auc']:.6f}（{o['table']}）{'  现在交的' if o['run'] == cur[ep] else ''}")
        if c["run"] != cur[ep]:
            old = next(o for o in cands if o["run"] == cur[ep])
            changed.append(f"ep{ep}：{cur[ep]}（MP-SR {old['mp']:.3f}、ADD-AUC {old['auc']:.3f}）→ {c['run']}（MP-SR {c['mp']:.3f}、ADD-AUC {c['auc']:.3f}）")
        n_r3a += "-r3a-" in c["run"]
        out.append(f"2\t{ep}\t{c['run']}\t{a.step}\t{c['machine']}")
    Path(a.out_tsv).write_text("\n".join(out) + "\n")
    print(f"换了的 episode {len(changed)} 条：")
    for line in changed:
        print(f"  {line}")
    print(f"选中者里名字带 -r3a- 的 {n_r3a} 条；选择清单 {a.out_tsv}")
    print("SELECT_T2_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
