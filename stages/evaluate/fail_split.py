"""逐条选的结果按 1 世界成败与 4096 世界的摔倒比例分组（Mac 上跑，只用标准库，只读已落盘的结果）。

输入：逐条选的清单 t1_select_*.tsv（档位、episode、run、步数、主机）、同一次的 1 世界打分 scores_*.json，
以及 4096 世界评测文本（summary_json 行，fc_eval 打的）。评测文本按主机找：--eval-dir <主机>=<目录>，
文件名 eval_<run>_s<步数>.txt。
每条 episode 一行：选中的 run、1 世界 MP-SR / SP-SR、4096 世界 MP-SR / SP-SR、4096 世界失败原因里骨盆的比例。
1 世界 MP-SR 没过的，按骨盆比例分三组：>= 0.9、<= 0.4、其余。界线只用于描述，不是判据。
用法：python3 stages/evaluate/fail_split.py <清单.tsv> <打分.json> --eval-dir mf4090=results/day2 --eval-dir ml40=results/ml40/day2
"""
import argparse
import json
import re
from pathlib import Path

GROUPS = ["1世界过", "骨盆>=0.9", "骨盆<=0.4", "其余"]


def summary(path: Path) -> dict:
    m = re.search(r"summary_json (\{.*\})", path.read_text(encoding="utf-8"))
    if not m:
        raise SystemExit(f"fail_split: {path} 里没有 summary_json 行")
    return json.loads(m.group(1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("select_tsv", type=Path)
    ap.add_argument("scores_json", type=Path)
    ap.add_argument("--eval-dir", action="append", required=True, help="<主机>=<目录>")
    a = ap.parse_args()
    dirs = dict(x.split("=", 1) for x in a.eval_dir)
    scores = json.loads(a.scores_json.read_text(encoding="utf-8"))
    groups = {g: [] for g in GROUPS}
    pels = {g: [] for g in GROUPS}
    fail_mp4096 = []
    r3a = []
    print("\t".join(["ep", "run", "1世界_MP-SR", "1世界_SP-SR", "4096世界_MP-SR", "4096世界_SP-SR", "4096世界_失败_骨盆", "组"]))
    for line in a.select_tsv.read_text(encoding="utf-8").splitlines():
        _tier, ep, run, step, host = line.split("\t")
        s = scores[run]
        j = summary(Path(dirs[host]) / f"eval_{run}_s{step}.txt")
        if Path(j["checkpoint"]).parent.name != run or str(j["environment_steps"]) != step:
            raise SystemExit(f"fail_split: {run} 的评测文本对不上（{j['checkpoint']}，{j['environment_steps']}）")
        if abs(j["maniptrans_sr"] - s["mp_sr_4096"]) > 1e-9:
            raise SystemExit(f"fail_split: {run} 的 4096 世界 MP-SR 与打分文件不一致")
        pel = j["reference_tracking"]["failure_cause_fraction"].get("pelvis", 0.0)
        if s["maniptrans_sr"] == 1.0:
            g = "1世界过"
        else:
            g = "骨盆>=0.9" if pel >= 0.9 else "骨盆<=0.4" if pel <= 0.4 else "其余"
            fail_mp4096.append(j["maniptrans_sr"])
        groups[g].append(f"ep{ep}")
        pels[g].append(pel)
        if "-r3a" in run:
            r3a.append((f"ep{ep}", s["maniptrans_sr"]))
        print("\t".join([f"ep{ep}", run, f"{s['maniptrans_sr']:.0f}", f"{s['spider_sr']:.0f}", f"{j['maniptrans_sr']:.3f}",
                         f"{j['spider_sr_uncentered']:.3f}", f"{pel:.3f}", g]))
    print(f"1 世界 MP-SR 过 {len(groups['1世界过'])} 条：{'、'.join(groups['1世界过'])}")
    n_fail = len(fail_mp4096)
    print(f"1 世界 MP-SR 没过 {n_fail} 条，按 4096 世界失败原因里骨盆的比例分：")
    for g in GROUPS[1:]:
        rng = f"，失败_骨盆 {min(pels[g]):.3f}–{max(pels[g]):.3f}" if pels[g] else ""
        print(f"  {g}：{len(groups[g])} 条{rng}：{'、'.join(groups[g]) or '无'}")
    if fail_mp4096:
        print(f"没过的 {n_fail} 条，选中 run 的 4096 世界 MP-SR 最大 {max(fail_mp4096):.3f}")
    print(f"选中 run 名含 r3a（变体 A）的 {len(r3a)} 条：1 世界 MP-SR 过 {sum(v == 1.0 for _, v in r3a)} 条"
          f"（{'、'.join(e for e, _ in r3a)}）")
    print("FAIL_SPLIT_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
