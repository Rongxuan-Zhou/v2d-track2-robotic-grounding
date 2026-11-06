"""按集取导出那一局的四项，算一份 Tier 1 清单打包后的榜分（Tier 1 的参考就是赛方的干净轨迹，本地复算与榜上逐位相同，
plan-t1push 第 1 节事实 1）。只报，不用来选 run（换集按 4096 世界期望，plan-t12night 第 3 节）。

基线取 select 文件里 30 行 CHOSEN 的导出值；--swap ep=run 换掉的集从导出检查文件
（results/**/submission/exports/*_<run>_s349999104_check.txt 的 info 行）取。
用法：python3 tools/t1_list_export.py results/ml40/select_t1/select_t1x1018.txt [--swap 8=t1-ep8-...] ...
"""
import glob
import re
import sys

KEYS = ("add_auc", "spider_sr_uncentered", "maniptrans_sr", "mppe_cm")


def base_rows(path):
    rows = {}
    for line in open(path, encoding="utf-8"):
        m = re.match(r"CHOSEN ep(\d+): (\S+?)（", line)
        if not m:
            continue
        vals = dict(re.findall(r"(add_auc|spider_sr|maniptrans_sr|mppe_cm)=([0-9.]+)", line))
        rows[int(m.group(1))] = (m.group(2), float(vals["add_auc"]), float(vals["spider_sr"]),
                                 float(vals["maniptrans_sr"]), float(vals["mppe_cm"]), path)
    assert sorted(rows) == list(range(30)), sorted(rows)
    return rows


def check_row(run):
    hits = sorted(set(glob.glob(f"results/**/exports/*_{run}_s349999104_check.txt", recursive=True)))
    if not hits:
        sys.exit(f"找不到导出检查：{run}")
    found = []
    for h in hits:
        txt = open(h, encoding="utf-8").read()
        if "EXPORT_OK" not in txt:
            sys.exit(f"导出检查没过：{h}")
        info = re.search(r"info (add_auc .*)", txt).group(1)
        vals = dict(re.findall(r"(\w+) ([0-9.]+)", info))
        found.append(tuple(float(vals[k]) for k in KEYS))
    if len(set(found)) != 1:
        sys.exit(f"{run} 的导出检查不止一种值：{hits}")
    return found[0], hits[0]


def main(argv):
    rows = base_rows(argv[0])
    for a in argv[1:]:
        if a == "--swap":
            continue
        ep, run = a.split("=", 1)
        if not run.startswith(f"t1-ep{int(ep)}-"):
            sys.exit(f"run 名与集号不符：{a}")
        v, src = check_row(run)
        rows[int(ep)] = (run,) + v + (src,)
    n = len(rows)
    for ep in sorted(rows):
        r = rows[ep]
        print(f"ep{ep:<2} {r[0]:<48} auc {r[1]:.4f} sp {r[2]:.0f} mp {r[3]:.0f} mppe {r[4]:8.4f}  {r[5]}")
    tot = [sum(rows[e][i] for e in rows) for i in range(1, 5)]
    print(f"TOTAL n={n} add_auc {tot[0] / n:.5f} spider_sr {tot[1] / n:.5f}（{tot[1]:.0f}/30） "
          f"maniptrans_sr {tot[2] / n:.5f}（{tot[2]:.0f}/30） mppe_cm {tot[3] / n:.5f}")


if __name__ == "__main__":
    main(sys.argv[1:])
