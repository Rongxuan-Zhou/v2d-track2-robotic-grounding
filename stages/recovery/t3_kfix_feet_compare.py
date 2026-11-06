"""k-fix 判据 4 的对照表：新参考与旧参考的世界步之后，人体支撑脚离发布地面的高度逐集并排（Mac 上跑，只用标准库；只读两份记录）。
计划 agent-doc/plan-t3-kfix-2026-10-16.md 判据 4：地面校正之后支撑脚高度中位的绝对值 ≤ 3 cm，旧链人体也按地面缩放过，作对照一并报。
两份输入都是 scripts/dev/record 的记录，里面每行一个 JSON（另一位用户 t3_feet_diag.py 的逐集输出）：
  - 新参考：results/t3world/kfix_feet_kt.txt（影子根，floor_fit_kt.json，缩放 0.984）；
  - 旧链：results/t3world/kfix_feet_delivered.txt（另一位用户 convert 目录里现有的世界步结果，floor_fit_f830.json，缩放 0.9856；
    判据 0 已核过这一步与旧参考 tier3_cari4d_f830_floor 同源）。
输出：第一批 7 集逐集并排（世界步之后与之前），各自逐集过线的集数、7 集绝对值的中位；旧链主机位（group 0）全部集的同样统计
（看 3 cm 这条线旧链自己过不过）。
登记时没写“逐集判”还是“合起来判”（疏漏），两种读法都报，不在这里选。
用法：python3 stages/recovery/t3_kfix_feet_compare.py <新参考记录> <旧链记录>；最后一行 KFIX_FEET_COMPARE_DONE。
"""
import json
import statistics
import sys
from pathlib import Path

BATCH = ("ep00", "ep18", "ep20", "ep22", "ep24", "ep26", "ep29")
LIMIT = 0.03


def rows(path: str) -> dict:
    out = {}
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        if ln.startswith("{"):
            r = json.loads(ln)
            out[r["episode"]] = r
    return out


def summary(name: str, rs: list) -> str:
    w = [r["feet_world_m"] for r in rs]
    ok = sum(abs(x) <= LIMIT for x in w)
    return (f"{name}：{len(w)} 集，逐集 |·| ≤ 3 cm 的 {ok} 集；绝对值的中位 {100 * statistics.median(abs(x) for x in w):.2f} cm，"
            f"带符号的中位 {100 * statistics.median(w):+.2f} cm，最低 {100 * min(w):+.2f} cm")


def main() -> int:
    new, old = rows(sys.argv[1]), rows(sys.argv[2])
    pick = lambda d: [d[k] for k in sorted(d) if k[:4] in BATCH]
    bn, bo = pick(new), pick(old)
    if [r["episode"] for r in bn] != [r["episode"] for r in bo] or len(bn) != len(BATCH):
        raise SystemExit(f"第一批的集对不上：新 {[r['episode'] for r in bn]}，旧 {[r['episode'] for r in bo]}")
    print("集\t新参考 世界步后(cm)\t新参考 世界步前(cm)\t旧链 世界步后(cm)\t旧链 世界步前(cm)\t新 − 旧(cm)")
    for a, b in zip(bn, bo):
        print(f"{a['episode']}\t{100 * a['feet_world_m']:+.2f}\t{100 * a['feet_focal_m']:+.2f}\t"
              f"{100 * b['feet_world_m']:+.2f}\t{100 * b['feet_focal_m']:+.2f}\t{100 * (a['feet_world_m'] - b['feet_world_m']):+.2f}")
    print(summary("第一批 新参考", bn))
    print(summary("第一批 旧链", bo))
    main_cam = [r for r in old.values() if r.get("group") == 0]
    print(summary("旧链 主机位全部集", main_cam))
    far = [f"{r['episode']} {100 * r['feet_world_m']:+.2f}" for r in main_cam if abs(r["feet_world_m"]) > LIMIT]
    print("旧链主机位里 |·| > 3 cm 的集：" + ("、".join(far) if far else "无"))
    other = sorted(r["episode"] for r in old.values() if r.get("group") != 0)
    print(f"旧链里不是主机位（group ≠ 0）的集，不计入上面两行：{'、'.join(other) if other else '无'}")
    print("KFIX_FEET_COMPARE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
