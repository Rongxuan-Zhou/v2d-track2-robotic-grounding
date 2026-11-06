"""t3b2 判据 T0 的逐项比对（Mac 上跑，只用标准库）。计划 agent-doc/plan-t3b2-2026-10-28.md 判据 T0。
A：t3_frame0_table.py 在已交帧参考上的报表，对 align_frefs_t3.py 当时的报表（results/t3align/align_t3m1022a_auto.txt），逐集比 自动方案、对齐前转角差、倾斜、物体偏航、人体偏航（差 ≤ 0.01°）。
B：新出的 hm 帧参考对已交的（多一个 4.652° 世界偏航）：倾斜、人体偏航相同（≤ 0.01°），物体偏航差的绝对值是 4.652°（±0.01°，取模 360）。
C/D：负对照（obj、--swap-quat）的 对齐输出差 每集 > 30°；阳性对照（obj）每集 ≤ 1°。
用法：python3 stages/evaluate/t3b2_t0_compare.py <align_t3m1022a_auto.txt> <table_cmp> <table_hm> <table_negctl> <table_posctl>   最后一行 T0_OK 或 T0_FAIL <项>。
"""
import sys


def rows(path, key_cols):
    out = {}
    for line in open(path, encoding="utf-8"):
        f = line.rstrip("\n").split("\t")
        if f and f[0].isdigit() and len(f) >= key_cols:
            out[int(f[0])] = f
    return out


def num(x):
    return float(x)


def main():
    old, cmp_, hm, neg, pos = (rows(p, 9) for p in sys.argv[1:6])
    bad = []
    # old: ep 方案 理由 静止帧 位置差 转角差 倾斜 偏航物体 偏航人体 ...；table: ep 档位 自动方案 转角差 倾斜 偏航物体 偏航人体 ...
    for ep, t in sorted(cmp_.items()):
        o = old[ep]
        same = o[1] == t[2] and all(abs(num(o[i]) - num(t[j])) <= 0.01 for i, j in ((5, 3), (6, 4), (7, 5), (8, 6)) if o[i] != "-" and t[j] != "-")
        print(f"A ep{ep}: 旧 {o[1]} {o[5]}/{o[6]}/{o[7]}/{o[8]}  表 {t[2]} {t[3]}/{t[4]}/{t[5]}/{t[6]}  {'同' if same else '不同'}")
        bad += [] if same else [f"A ep{ep}"]
    for ep, t in sorted(hm.items()):
        c = cmp_[ep]
        dy = abs((num(t[5]) - num(c[5]) + 180) % 360 - 180)
        ok = abs(num(t[4]) - num(c[4])) <= 0.01 and (t[6] == c[6] == "-" or abs(num(t[6]) - num(c[6])) <= 0.01) and abs(dy - 4.652) <= 0.01
        print(f"B ep{ep}: 倾斜 {c[4]}→{t[4]}  人体偏航 {c[6]}→{t[6]}  物体偏航差 {dy:.3f}  {'过' if ok else '不过'}")
        bad += [] if ok else [f"B ep{ep}"]
    for name, tab, test in (("C", neg, lambda d: d > 30), ("D", pos, lambda d: d <= 1)):
        ds = {ep: num(t[-1].split("(")[0]) for ep, t in tab.items() if t[-1] != "-"}
        fails = [ep for ep, d in ds.items() if not test(d)]
        print(f"{name}: {len(ds)} 集，对齐输出差 {min(ds.values()):.2f}–{max(ds.values()):.2f}°，不过 {fails}")
        bad += [f"{name} ep{ep}" for ep in fails]
    print("T0_OK" if not bad else f"T0_FAIL {', '.join(bad)}")
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
