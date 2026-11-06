"""核对按选择清单拉回的导出与打出的 CSV 确实是选中的那些 run（Mac 上跑，只用标准库）。计划 agent-doc/plan-t2-select-b-2026-10-12.md 第 2 版
第 2 节第 5、7 步（审查 M1：行数与预检分不出新旧 CSV）。
对选择清单里的每条 episode：
  1. 选中 run 的评测记录 eval_<run>_s<步数>.txt 的 summary_json 里，checkpoint 属于这条 run，记下 checkpoint_sha256；
  2. 结果目录里的 episode_XXXXXX.json：checkpoint_sha256 与第 1 步相同，world_count 为 1；
  3. 给了 --csv 时：CSV 里这条 episode 的行（row_id 形如 track_2/e000017/w000/f000000/o0）只有一个 checkpoint_sha256，且与第 1 步相同；
     CSV 里的 episode 恰好是清单里的那些。
逐条打印，最后一行 CHECK_T2_SELECT_OK 或 CHECK_T2_SELECT_BAD（退出码 0 / 1）。
用法：python3 stages/evaluate/check_t2_select.py --selection <清单.tsv> --evals <目录>… --results <目录> [--csv <csv>] [--step 349999104]
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path


def eval_sha(dirs, run: str, step: str):
    for d in dirs:
        p = Path(d) / f"eval_{run}_s{step}.txt"
        if p.exists():
            m = re.search(r"summary_json (\{.*\})", p.read_text(encoding="utf-8"))
            if m:
                j = json.loads(m.group(1))
                if Path(j["checkpoint"]).parent.name == run:
                    return j["checkpoint_sha256"]
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", required=True)
    ap.add_argument("--evals", nargs="+", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--csv")
    ap.add_argument("--step", default="349999104")
    a = ap.parse_args(argv)
    sel = {}
    for line in Path(a.selection).read_text().splitlines():
        c = line.split("\t")
        if len(c) >= 5 and c[0] == "2":
            sel[int(c[1])] = c[2]
    per_ep = {}
    if a.csv:
        with open(a.csv, newline="") as f:
            for r in csv.DictReader(f):
                ep = int(r["row_id"].split("/")[1].lstrip("e"))
                per_ep.setdefault(ep, set()).add(r["checkpoint_sha256"])
    bad = 0
    for ep in sorted(sel):
        run = sel[ep]
        want = eval_sha(a.evals, run, a.step)
        jp = Path(a.results) / f"episode_{ep:06d}.json"
        js = json.loads(jp.read_text()) if jp.exists() else {}
        problems = []
        if want is None:
            problems.append("没有这条 run 的评测记录或 checkpoint_sha256")
        if js.get("checkpoint_sha256") != want:
            problems.append(f"导出 JSON 的 checkpoint_sha256 是 {str(js.get('checkpoint_sha256'))[:16]}")
        if js.get("world_count") != 1:
            problems.append(f"导出 JSON 的 world_count 是 {js.get('world_count')}")
        if a.csv and per_ep.get(ep) != {want}:
            problems.append(f"CSV 里这条的 checkpoint_sha256 是 {sorted(s[:16] for s in per_ep.get(ep, set()))}")
        bad += bool(problems)
        print(f"{'BAD ' if problems else 'OK  '} ep{ep:<3}{run}：{str(want)[:16]}{('；' + '；'.join(problems)) if problems else ''}")
    if a.csv and sorted(per_ep) != sorted(sel):
        bad += 1
        print(f"BAD  CSV 里的 episode {sorted(per_ep)} 与清单 {sorted(sel)} 不同")
    print("CHECK_T2_SELECT_OK" if bad == 0 else f"CHECK_T2_SELECT_BAD（{bad} 处不符）")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
