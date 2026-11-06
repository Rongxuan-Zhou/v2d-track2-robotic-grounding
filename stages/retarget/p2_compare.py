"""把同一批集在两版 Tier 3 重建上的 P2′ 表逐集并列（Mac 上跑，只用标准库，只读已落盘的结果）。
计划 agent-doc/plan-t3-simreg-p2-2026-10-11.md 第 2 节第 4 步：simreg 与 f830 逐集并列，只报，不改任何一版的结论。
输入：两份 stages/retarget/p2_table.py 的输出（经 scripts/dev/record 落盘），写成 <名>=<路径>。按表头行（以 “ep\t序列” 开头）认列，
数据行是第 1 列为集号、列数与表头相同的行；“失败” 行的数值列是 “-”。
输出：两版都有的集，每集一行：集、序列、间隙（A、B、B−A）、左手接触（A、B）、右手接触（A、B）、IK 中位（A、B）、P2′（A、B）、结论变没变。
然后几行汇总：
  SAME_VERDICT <n>/<n> 或 VERDICT_CHANGED <集号…>；
  MAX_ABS_GAP_DIFF <mm>（是哪一集）与 GAP_DIFF_GT_1MM <集号…>（两版都有间隙的集里，|B−A| > 1 mm 的）；
  ONLY_A、ONLY_B：只在一版里有的集。
用法：python3 stages/retarget/p2_compare.py <A 名>=<A 表> <B 名>=<B 表>
测试：stages/retarget/test_p2_compare.py。
"""
import sys
from pathlib import Path

COLS = ("序列", "左手接触", "右手接触", "IK中位", "间隙mm", "P2′")


def load(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    head = next((x.split("\t") for x in lines if x.startswith("ep\t序列")), None)
    if head is None:
        raise SystemExit(f"p2_compare: {path} 里没有表头行（ep\\t序列…）")
    missing = [c for c in COLS if c not in head]
    if missing:
        raise SystemExit(f"p2_compare: {path} 的表头缺列 {missing}")
    idx = {c: head.index(c) for c in COLS}
    rows = {}
    for x in lines:
        f = x.split("\t")
        if len(f) == len(head) and f[0].isdigit():
            ep = int(f[0])
            if ep in rows:
                raise SystemExit(f"p2_compare: {path} 里 ep{ep} 出现两次")
            rows[ep] = {c: f[i] for c, i in idx.items()}
    if not rows:
        raise SystemExit(f"p2_compare: {path} 里没有数据行")
    return rows


def num(s: str):
    try:
        return float(s)
    except ValueError:
        return None


def main(argv=None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 2 or not all("=" in a for a in args):
        raise SystemExit("用法：p2_compare.py <A 名>=<A 表> <B 名>=<B 表>")
    (na, pa), (nb, pb) = (a.split("=", 1) for a in args)
    A, B = load(Path(pa)), load(Path(pb))
    print(f"A = {na}：{pa}；B = {nb}：{pb}")
    print(f"ep\t序列\t间隙mm_{na}\t间隙mm_{nb}\t差mm\t左手_{na}\t左手_{nb}\t右手_{na}\t右手_{nb}\tIK_{na}\tIK_{nb}\tP2′_{na}\tP2′_{nb}\t结论")
    both = sorted(set(A) & set(B))
    changed, diffs = [], {}
    for ep in both:
        a, b = A[ep], B[ep]
        ga, gb = num(a["间隙mm"]), num(b["间隙mm"])
        d = None if ga is None or gb is None else gb - ga
        if d is not None:
            diffs[ep] = d
        same = a["P2′"] == b["P2′"]
        if not same:
            changed.append(ep)
        print("\t".join([str(ep), b["序列"], a["间隙mm"], b["间隙mm"], "-" if d is None else f"{d:+.1f}",
                         a["左手接触"], b["左手接触"], a["右手接触"], b["右手接触"], a["IK中位"], b["IK中位"],
                         a["P2′"], b["P2′"], "不变" if same else "变了"]))
    n = len(both)
    print(f"SAME_VERDICT {n}/{n}" if not changed else f"VERDICT_CHANGED {' '.join(f'ep{e}' for e in changed)}（{len(changed)}/{n}）")
    if diffs:
        worst = max(diffs, key=lambda e: abs(diffs[e]))
        print(f"MAX_ABS_GAP_DIFF {abs(diffs[worst]):.1f} mm（ep{worst}）")
        big = [e for e in sorted(diffs) if abs(diffs[e]) > 1.0]
        print(f"GAP_DIFF_GT_1MM {' '.join(f'ep{e}' for e in big) or '无'}")
    print(f"ONLY_{na} {' '.join(f'ep{e}' for e in sorted(set(A) - set(B))) or '无'}")
    print(f"ONLY_{nb} {' '.join(f'ep{e}' for e in sorted(set(B) - set(A))) or '无'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
