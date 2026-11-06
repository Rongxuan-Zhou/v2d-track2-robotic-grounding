"""t3static 的选集表（Mac 上跑，只用标准库）：计划 agent-doc/plan-t3static-2026-10-21.md 第 2 节 S3（第 2 版），规则先于出表提交。
输入：清单 stages/evaluate/exports/t3_select_t3m1022a.tsv（档位 episode run 步数 机器），t3_static_floor.py 的结果文件（可多份），
各 run 的 4096 世界评测记录（results/ 下的 eval_<run>_s<步数>.txt，读其中 world_count 4096 的 summary_json）。
规则：一集换成站立保底，当且仅当（a）原 run 4096 世界对自己参考的 mppe_cm > 这一集参考的“不动”MPPE，且（b）原 run 的 SP-SR 不高于“不动”的 SP-SR。
原 run 的 SP-SR 取 summary_json 的 spider_sr，没有就取 spider_sr_uncentered，表里写明取的是哪个键；“不动”的 SP-SR 用工具包公式（t3_static_floor.py）。ADD-AUC 并列报，不进规则。
缺读数的集不判（记“缺”）。用法：python3 stages/evaluate/t3static_select.py <清单> <floor 结果文件> …   最后一行 T3STATIC_SELECT_DONE
"""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def floors(paths):
    out = {}
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            f = line.split("\t")
            if len(f) >= 8 and f[0].isdigit():
                out[int(f[0])] = {"T": int(f[1]), "mppe": float(f[2]), "sp": float(f[3]), "mp": float(f[4]), "auc": float(f[5]), "sha": f[6]}
    return out


def run_eval(run, step):
    hits = sorted(REPO.glob(f"results/**/eval_{run}_s{step}.txt"))
    for h in hits:
        for line in h.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.startswith("summary_json "):
                d = json.loads(line[len("summary_json "):])
                d = {**d, **d.get("metrics", {})}   # 原始评测 JSON 把指标放在 metrics 下（eval_json_record.py 的记录）
                if d.get("world_count", 4096) == 4096:
                    key = "spider_sr" if "spider_sr" in d else "spider_sr_uncentered"
                    return {"mppe": d["mppe_cm"], "sp": d[key], "key": key, "auc": d["add_auc"], "file": str(h.relative_to(REPO))}
    return None


def main() -> int:
    lst, fl = Path(sys.argv[1]), floors(sys.argv[2:])
    print("集\trun\t原 MPPE\t不动 MPPE\t原 SP-SR（键）\t不动 SP-SR\t原 AUC\t不动 AUC（近似）\t（a）\t（b）\t结论\t评测记录")
    swap, missing = [], []
    for line in lst.read_text(encoding="utf-8").splitlines():
        f = line.split("\t")
        if not f or line.startswith("#") or len(f) < 4:
            continue
        ep, run, step = int(f[1]), f[2], f[3]
        e, s = run_eval(run, step), fl.get(ep)
        if e is None or s is None:
            missing.append(ep)
            print(f"ep{ep}\t{run}\t" + ("缺评测记录" if e is None else "") + ("缺“不动”分" if s is None else "") + "\t\t\t\t\t\t\t\t缺\t")
            continue
        a = e["mppe"] > s["mppe"]
        b = e["sp"] <= s["sp"]
        if a and b:
            swap.append(ep)
        print(f"ep{ep}\t{run}\t{e['mppe']:.2f}\t{s['mppe']:.2f}\t{e['sp']:.3f}（{e['key']}）\t{s['sp']:.3f}\t{e['auc']:.3f}\t{s['auc']:.3f}\t"
              f"{'是' if a else '否'}\t{'是' if b else '否'}\t{'换成站立保底' if a and b else '留原 run'}\t{e['file']}")
    print(f"换成站立保底的集（{len(swap)}）：{swap}")
    print(f"缺读数、不判的集（{len(missing)}）：{missing}")
    print("T3STATIC_SELECT_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
