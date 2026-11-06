"""生成提交用的导出清单：每档 30 条 episode 各用哪条 run、哪一步、在哪台机器（Mac 上跑，只用标准库，只读已提交的结果与队列）。

计划 agent-doc/plan-submission-exports-2026-10-02.md 第 3.1 节。登记的选择规则（不按分数挑）：
  Tier 1：results/day2/batch_verdict_*.txt 里有逐帧回放判定的那条 run（27 条，与误差表、SOP 的口径相同），加上方盒重训的
          t1-ep{1,7,28}-…-p185-mf4090（stages/policy/queues/mf4090_box.expect.tsv）；
  Tier 2：第 1 批的期望表 stages/policy/queues/{ml40,mf4090}_t2{a,b,c}.expect.tsv（27 条，ep10 以 C 段为准），加上方盒的
          stages/policy/queues/{mf4090,ml40}_box.expect.tsv 里的 t2 行。
  步数一律用最终 checkpoint 349999104；规矩二的回退由导出链按需处理，不写进清单。
  10/3 加一条（ep7 带非有限防护的重训，计划 agent-doc/plan-ep7-eval-and-guards-2026-10-02.md 第 5 版 B.5）：两档的来源再加
  stages/policy/queues/{mf4090,ml40}_nfg.expect.tsv；同一条 episode 有名字带 -p185g- 的 run 时只用它（带防护的重训替换发散的方盒 run）。
每条 episode 必须恰好一条 run，否则报错退出。机器取自 run 名的后缀。
输出 stages/evaluate/exports/<机器>.tsv（tab 分隔：档位 episode run 步数），已存在就拒绝覆盖。V2D_EXPORT_DIR 可改输出目录（测试用）。
用法：python3 stages/evaluate/export_map.py [--root <仓库根>]   --root 只给测试换输入的根目录
"""
import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STEP = 349999104
MACHINES = ("mf4090", "ml40")
GUARDED = "-p185g-"


def tier1(root: Path) -> dict:
    out = {}
    for f in sorted(glob.glob(str(root / "results/day2/batch_verdict_*.txt"))):
        for line in open(f, encoding="utf-8"):
            if "summary_json " not in line:
                continue
            d = json.loads(line.split("summary_json ", 1)[1])
            m = re.fullmatch(r"t1-ep(\d+)-w2048-(cap10m-ml40|cap4m-mf4090)", d["run"])
            if m and d.get("playback") == "frame" and d.get("verdict") in ("PASS", "FAIL"):
                out.setdefault(int(m.group(1)), set()).add(d["run"])
    return out


def expect_runs(paths: list, tier: int) -> dict:
    out = {}
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines()[1:]:
            run = line.split("\t")[0]
            m = re.fullmatch(rf"t{tier}-ep(\d+)-w2048-.+-(ml40|mf4090)", run)
            if m:
                out.setdefault(int(m.group(1)), set()).add(run)
    return out


def merge(a: dict, b: dict) -> dict:
    out = {k: set(v) for k, v in a.items()}
    for k, v in b.items():
        out.setdefault(k, set()).update(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    root = ap.parse_args().root
    q = root / "stages/policy/queues"
    nfg = [q / f"{m}_nfg.expect.tsv" for m in MACHINES]
    t1 = merge(merge(tier1(root), expect_runs([q / "mf4090_box.expect.tsv"], 1)), expect_runs(nfg, 1))
    t2a = expect_runs([q / f"{m}_t2{s}.expect.tsv" for m in MACHINES for s in "ab"], 2)
    t2c = expect_runs([q / "ml40_t2c.expect.tsv"], 2)
    t2 = merge(merge(merge(t2a, t2c), expect_runs([q / f"{m}_box.expect.tsv" for m in MACHINES], 2)), expect_runs(nfg, 2))
    rows, bad = [], []
    for tier, table in ((1, t1), (2, t2)):
        for ep in range(30):
            runs = sorted(table.get(ep, ()))
            guarded = [r for r in runs if GUARDED in r]
            if guarded:
                runs = guarded
            if len(runs) != 1:
                bad.append(f"Tier {tier} ep{ep}：{runs or '没有 run'}")
                continue
            rows.append((tier, ep, runs[0], runs[0].rsplit("-", 1)[1]))
    if bad:
        sys.exit("每条 episode 必须恰好一条 run：\n  " + "\n  ".join(bad))
    out = Path(os.environ.get("V2D_EXPORT_DIR", root / "stages/evaluate/exports"))
    out.mkdir(parents=True, exist_ok=True)
    for m in MACHINES:
        p = out / f"{m}.tsv"
        if p.exists():
            sys.exit(f"{p} 已存在，拒绝覆盖")
    for m in MACHINES:
        mine = [r for r in rows if r[3] == m]
        (out / f"{m}.tsv").write_text(
            f"# 导出清单（{m}，计划 agent-doc/plan-submission-exports-2026-10-02.md）：档位 episode run 步数；export_map.py 生成，不要手改\n"
            + "".join(f"{t}\t{ep}\t{run}\t{STEP}\n" for t, ep, run, _ in mine), encoding="utf-8")
        print(f"写出 {out / f'{m}.tsv'}：Tier 1 {sum(r[0] == 1 for r in mine)} 条，Tier 2 {sum(r[0] == 2 for r in mine)} 条")
    return 0


if __name__ == "__main__":
    sys.exit(main())
