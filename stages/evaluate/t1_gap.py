"""Tier 1 与排行榜上两家的差距：每项指标还差多少，修好几条失败的 episode 才够。

只读两份已有记录：select_t1.py 的输出（CHOSEN 行是各 episode 交上去的 run 与按轨迹复算的指标，PACKED 行是整份 CSV 的合计）
与 kaggle CLI 抓的榜单快照（results/submission/leaderboard_*.txt）。先核对 CHOSEN 行的平均等于 PACKED 行，对不上就停。

“修好”的两种假设（都不是测出来的，只用来估量级）：
  像成功的：取现在 MP-SR 成功的 episode 的中位 AUC 与中位 MPPE，SP 与 MP 都算成功；
  只跟住的：取现在没成功但 MPPE 在 15 cm 以内的 episode 的中位 AUC 与中位 MPPE，SP 算成功、MP 不算。
修的顺序按 MPPE 从大到小（先修丢得最远的），只在现在 MP-SR 不成功的 episode 里选。
用法：python3 stages/evaluate/t1_gap.py --select <select 记录> --board <榜单快照> [--candidate]
--candidate：select 记录是选出来、还没交的 CSV，不核对榜单上 maniphys.ai 的四项等于 PACKED（交过的那份才核）。"""

import argparse
import re
import statistics

METRICS = ["add_auc", "spider_sr", "maniptrans_sr", "mppe_cm"]
SUFFIX = {"auc": "add_auc", "sp-sr": "spider_sr", "mp-sr": "maniptrans_sr", "mppe": "mppe_cm"}
TEAMS = ["CHORD Baseline", "Xueyi Wang (FF)", "maniphys.ai"]
N = 30

CHOSEN = re.compile(r"^CHOSEN ep(\d+): (\S+)（现在 \S+）add_auc=(\S+) spider_sr=(\S+) maniptrans_sr=(\S+) mppe_cm=(\S+)")
PACKED = re.compile(r"^PACKED add_auc=(\S+) spider_sr=(\S+) maniptrans_sr=(\S+) mppe_cm=(\S+)")
BOARD_ROW = re.compile(r"^\s*\d+\s+(.+?)\s+\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+\s+([\d.]+)\s*$")


def read_select(path):
    eps, packed = {}, None
    for line in open(path, encoding="utf-8", errors="replace"):
        m = CHOSEN.match(line)
        if m:
            eps[int(m.group(1))] = {"run": m.group(2), **{k: float(v) for k, v in zip(METRICS, m.groups()[2:])}}
        m = PACKED.match(line)
        if m:
            packed = {k: float(v) for k, v in zip(METRICS, m.groups())}
    if len(eps) != N or packed is None:
        raise SystemExit(f"{path}：CHOSEN 行 {len(eps)} 条（要 {N} 条），PACKED 行{'有' if packed else '没有'}")
    for k in METRICS:
        mean = sum(e[k] for e in eps.values()) / N
        if abs(mean - packed[k]) > 1e-5:
            raise SystemExit(f"{k}：CHOSEN 行平均 {mean:.6f} 与 PACKED {packed[k]:.6f} 对不上")
    return eps, packed


def read_board(path):
    board, metric = {}, None
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("== "):
            slug = line[3:].strip()
            metric = SUFFIX[slug.split("tier1-", 1)[1]] if "-tier1-" in slug else None
            continue
        m = BOARD_ROW.match(line)
        if metric and m:
            board.setdefault(m.group(1), {})[metric] = float(m.group(2))
    for t in TEAMS:
        if set(board.get(t, {})) != set(METRICS):
            raise SystemExit(f"{path}：Tier 1 榜单里 {t} 的四项不全：{board.get(t)}")
    return board


def beats(k, ours, target):
    # Kaggle 显示的分数截到 5 位小数（0.471279 显示成 0.47127，20/30 显示成 0.66666），SP、MP 按成功条数比
    if k in ("spider_sr", "maniptrans_sr"):
        return round(ours * N) > round(target * N)
    return ours < target if k == "mppe_cm" else ours > target


def totals(eps):
    return {k: sum(e[k] for e in eps.values()) / N for k in METRICS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select", required=True)
    ap.add_argument("--board", required=True)
    ap.add_argument("--candidate", action="store_true")
    a = ap.parse_args()
    eps, packed = read_select(a.select)
    board = read_board(a.board)
    ours = board["maniphys.ai"]
    print("核对：CHOSEN 行的平均与 PACKED 行一致；榜单上 maniphys.ai 的四项：" + "、".join(f"{k} {ours[k]:g}" for k in METRICS))
    if a.candidate:
        print("候选（还没交）：不核对榜单与 PACKED；下面“我们”的四项是 PACKED：" + "、".join(f"{k} {packed[k]:g}" for k in METRICS))
    else:
        for k in METRICS:
            if abs(ours[k] - packed[k]) > 1e-4:
                raise SystemExit(f"{k}：榜单 {ours[k]} 与 PACKED {packed[k]} 相差超过 1e-4")

    print("\n== 现在每项差多少（30 条 episode；SP、MP 换算成成功条数，AUC、MPPE 换算成 30 条的总和）")
    for t in ["CHORD Baseline", "Xueyi Wang (FF)"]:
        print(f"-- {t}")
        for k in METRICS:
            tv, ov = board[t][k], packed[k]
            if k in ("spider_sr", "maniptrans_sr"):
                need = int(round(tv * N)) + 1 - int(round(ov * N))
                print(f"  {k:14s} 对方 {tv:.5f}（{tv * N:.0f} 条）我们 {ov:.5f}（{ov * N:.0f} 条）"
                      + (f"：要多成功 {need} 条才超过" if need > 0 else "：已经超过"))
            elif k == "add_auc":
                gap = (tv - ov) * N
                print(f"  {k:14s} 对方 {tv:.5f} 我们 {ov:.5f}" + (f"：30 条总和要再加 {gap:.3f}" if gap >= 0 else "：已经超过"))
            else:
                gap = (ov - tv) * N
                print(f"  {k:14s} 对方 {tv:.3f} 我们 {ov:.3f}" + (f"：30 条总和要再减 {gap:.1f} cm" if gap >= 0 else "：已经超过"))

    ok = [e for e in eps.values() if e["maniptrans_sr"] == 1.0]
    near = [e for e in eps.values() if e["maniptrans_sr"] == 0.0 and e["mppe_cm"] <= 15.0]
    order = sorted((i for i, e in eps.items() if e["maniptrans_sr"] == 0.0), key=lambda i: -eps[i]["mppe_cm"])
    print("\n== 现在 MP-SR 不成功的 episode，按 MPPE 从大到小：")
    for i in order:
        e = eps[i]
        print(f"  ep{i:<3d} AUC {e['add_auc']:.3f}  SP {e['spider_sr']:.0f}  MPPE {e['mppe_cm']:7.2f}  {e['run']}")
    fail_sum = sum(eps[i]["mppe_cm"] for i in order if eps[i]["mppe_cm"] > 15.0)
    print(f"  MPPE 超过 15 cm 的 {sum(eps[i]['mppe_cm'] > 15.0 for i in order)} 条合计 {fail_sum:.1f} cm，"
          f"占 30 条总和 {packed['mppe_cm'] * N:.1f} cm 的 {fail_sum / (packed['mppe_cm'] * N):.1%}")

    scen = {
        "像成功的": {"add_auc": statistics.median(e["add_auc"] for e in ok), "spider_sr": 1.0,
                  "maniptrans_sr": 1.0, "mppe_cm": statistics.median(e["mppe_cm"] for e in ok)},
        "只跟住的": {"add_auc": statistics.median(e["add_auc"] for e in near), "spider_sr": 1.0,
                  "maniptrans_sr": 0.0, "mppe_cm": statistics.median(e["mppe_cm"] for e in near)},
    }
    for name, s in scen.items():
        print(f"\n== 假设“{name}”：修好的 episode 取 AUC {s['add_auc']:.3f}、SP {s['spider_sr']:.0f}、"
              f"MP {s['maniptrans_sr']:.0f}、MPPE {s['mppe_cm']:.2f} cm（不低于原值的项保留原值）")
        first = {(t, k): None for t in ("CHORD Baseline", "Xueyi Wang (FF)") for k in METRICS}
        for n in range(len(order) + 1):
            new = {i: dict(e) for i, e in eps.items()}
            for i in order[:n]:
                for k in METRICS:
                    better = s[k] < new[i][k] if k == "mppe_cm" else s[k] > new[i][k]
                    if better:
                        new[i][k] = s[k]
            tot = totals(new)
            for (t, k), v in first.items():
                if v is None and beats(k, tot[k], board[t][k]):
                    first[(t, k)] = n
            if n:
                print(f"  修前 {n:2d} 条（到 ep{order[n - 1]}）：" + "  ".join(
                    f"{k} {tot[k]:.4f}" if k != "mppe_cm" else f"{k} {tot[k]:.2f}" for k in METRICS))
        for t in ("CHORD Baseline", "Xueyi Wang (FF)"):
            parts = [f"{k} {'修不到' if first[(t, k)] is None else f'{first[(t, k)]} 条'}" for k in METRICS]
            print(f"  超过 {t} 至少要修：" + "；".join(parts))


if __name__ == "__main__":
    main()
