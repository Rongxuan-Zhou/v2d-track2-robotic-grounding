"""Tier 3 best_hm：失败起点与逐帧证据表的对照（Mac 上、带 numpy 的环境里跑）。计划 agent-doc/plan-frame-onset-2026-10-14.md 第 3 节第 0 步、第 4 节。

输入：fail_onset.py 写的 onset_<run>.json（只收 t3- 开头、名字带 -hm- 的 run）与另一位用户的逐帧证据表快照 <证据目录>/epNN.csv。
门：onset JSON 的 gate_ok 为真；G4：全部 run 的阈值与 active_after_step 相同、物体朝向阈值 0.6；G5：证据表行数等于 step_count，frame 列是
  0 到 step_count − 1 的连续整数，用到的布尔列（label_left、label_right、viol_*、orient_outlier）取值只有 0、1、True、False。不过的 run 记“门不过”。
旗标（逐帧）：F_flip = orient_outlier（短翻转：与前后 15 帧多数朝向差 > 45°，只抓 15 帧以内的翻转，成段翻转这张表测不了）；F_jump = viol_rot_jump
  （单帧转角跳变）；F_vis = evidence 含 off_image 或 no_mask；F_obj = viol_object_any；F_rot = viol_rot_torque、viol_rot_jump、viol_rot_jump_symmetric、
  viol_rot_drift 任一；F_root = viol_root_acc；F_sw = label_left 或 label_right 与上一帧不同。
窗口 [t − W, t]（W 主值 15，另报 5、30）。A = 有起点的失败世界里窗口内有旗标的比例（按 n_on 加权）；B = 从 b0 到最后一帧每一帧当起点时窗口内有旗标的比例，
  b0 = 骨盆两项与物体两项 active_after_step 的最小值 + 1；E = A / B。
进入汇总：有起点的失败世界 ≥ 50，起点中位 ≥ 100 帧（太早的起点和开头的旗标同在一处，比不出先后），0.02 ≤ B ≤ 0.5（B > 0.5 时 E 到不了 2）。
读法（每种旗标）：进入汇总的 run ≥ 4 条；E ≥ 2 记一次命中；命中 ≥ 2/3 且机会水平 p ≤ 0.05 记“集中”，p 是各 run 以自己的 B 为命中概率时
  “至少这么多命中”的概率（泊松二项，起点挤在一个窗口时 A 只能是 0 或 1，命中的机会约等于 B）；E ≤ 1.25 的 ≥ 2/3 记“不集中”；其余“看不清”。
另报：每条 run 起点的四分位距与最集中窗口的占比（≥ 50% 标“单窗口”）；ep04 已对照视频确认的头尾颠倒帧（313、509、669，见另一位用户的交接 197a02f）
  之后 30 帧内的起点数与同样宽度的机会占比。只报，不据此起训练。读文件前调 tier_guard.forbid_tier1()。
收的集（代码审查第 1 条）：按文件名 onset_<run>.json 先筛（不是 t3 best_hm 的不读），best_hm 的集必须恰好是登记的 18 集、不重复，否则退出码 2；
  --expect-eps 只给测试用。G4 没有严格多数时整组门不过。分位用 np.percentile（与 fail_onset.py 同一定义）。
用法：python stages/evaluate/t3_onset_evidence.py <证据目录> <onset JSON> [<onset JSON> ...] [--expect-eps 1,2,…]
"""

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from libs.v2d_infra import tier_guard  # noqa: E402

FLAGS = ("F_flip", "F_jump", "F_vis", "F_obj", "F_rot", "F_root", "F_sw")
LABEL = {"F_flip": "F_flip（短翻转 ≤15 帧）", "F_jump": "F_jump（单帧跳变）", "F_vis": "F_vis（看不清）", "F_obj": "F_obj（物体违规）",
         "F_rot": "F_rot（转动违规）", "F_root": "F_root（根加速度）", "F_sw": "F_sw（接触切换）"}
BOOL_COLS = ("label_left", "label_right", "viol_object_any", "viol_rot_torque", "viol_rot_jump", "viol_rot_jump_symmetric", "viol_rot_drift",
             "viol_root_acc", "orient_outlier")
BOOL_OK = {"0", "1", "True", "False"}
W_MAIN, W_SENS = 15, (5, 30)
MIN_ONSET, MIN_MEDIAN, B_LO, B_HI, E_HI, E_LO, MIN_RUNS, P_MAX = 50, 100, 0.02, 0.5, 2.0, 1.25, 4, 0.05
KNOWN_FLIPS = {4: (313, 509, 669)}
REGISTERED_HM = (1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17, 19, 21, 25, 27)   # 计划第 1 节：best_hm 训过的 18 集
HM_FILE = re.compile(r"^onset_(t3-ep(\d+)-.*-hm-.*)\.json$")
AFTER = 30


def pbinom_at_least(ps: list, k: int) -> float:
    """独立伯努利之和至少为 k 的概率（泊松二项，动态规划；同 tools/t1_multiworld_estimate.py）。"""
    dist = [1.0]
    for p in ps:
        new = [0.0] * (len(dist) + 1)
        for i, q in enumerate(dist):
            new[i] += q * (1 - p)
            new[i + 1] += q * p
        dist = new
    return sum(dist[k:])


def flags_from_csv(path: Path, T: int):
    rows = list(csv.DictReader(path.open(newline="")))
    frames = [r["frame"] for r in rows]
    if len(rows) != T or frames != [str(i) for i in range(T)]:
        return None, f"证据表 {path.name} 有 {len(rows)} 行，frame 列{'' if frames == [str(i) for i in range(len(rows))] else '不'}连续，step_count 是 {T}"
    bad = {c: sorted({r[c].strip() for r in rows} - BOOL_OK) for c in BOOL_COLS}
    bad = {c: v for c, v in bad.items() if v}
    if bad:
        return None, f"证据表 {path.name} 的布尔列有别的取值：{bad}"

    def col(name):
        return np.array([r[name].strip() in ("1", "True") for r in rows])

    ev = [r["evidence"] for r in rows]
    lab = np.array([[r["label_left"].strip(), r["label_right"].strip()] for r in rows])
    sw = np.zeros(T, dtype=bool)
    sw[1:] = (lab[1:] != lab[:-1]).any(axis=1)
    f = {"F_flip": col("orient_outlier"), "F_jump": col("viol_rot_jump"),
         "F_vis": np.array([("off_image" in e) or ("no_mask" in e) for e in ev]), "F_obj": col("viol_object_any"),
         "F_rot": col("viol_rot_torque") | col("viol_rot_jump") | col("viol_rot_jump_symmetric") | col("viol_rot_drift"),
         "F_root": col("viol_root_acc"), "F_sw": sw}
    return f, ""


def window_has(flag: np.ndarray, w: int) -> np.ndarray:
    c = np.concatenate([[0], np.cumsum(flag.astype(int))])
    t = np.arange(len(flag))
    return (c[t + 1] - c[np.maximum(0, t - w)]) > 0


def enrich(n_on: np.ndarray, flag: np.ndarray, b0: int, w: int):
    has = window_has(flag, w)
    a = float((n_on * has).sum() / n_on.sum()) if n_on.sum() else float("nan")
    b = float(has[b0:].mean()) if b0 < len(has) else float("nan")
    return a, b, (a / b if b > 0 else float("nan"))


def wquant(n_on: np.ndarray, q: float) -> float:
    """每帧计数展开成起点帧号后取 np.percentile（与 fail_onset.py 的分位同一定义）。"""
    x = np.repeat(np.arange(len(n_on)), n_on.astype(int))
    return float(np.percentile(x, q * 100)) if len(x) else -1.0


def reading(es: list, bs: list) -> str:
    n = len(es)
    if n < MIN_RUNS:
        return f"看不清（进入汇总的只有 {n} 条）"
    hi, lo = sum(e >= E_HI for e in es), sum(e <= E_LO for e in es)
    p = pbinom_at_least(bs, hi)
    if hi * 3 >= 2 * n and p <= P_MAX:
        return f"集中（{hi}/{n} 条 E ≥ {E_HI}，机会水平 p = {p:.3f}）"
    if lo * 3 >= 2 * n:
        return f"不集中（{lo}/{n} 条 E ≤ {E_LO}）"
    return f"看不清（E ≥ {E_HI} 的 {hi} 条、机会水平 p = {p:.3f}；E ≤ {E_LO} 的 {lo} 条；共 {n} 条）"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("evdir")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--expect-eps", default=",".join(map(str, REGISTERED_HM)), help="只给测试用")
    a = ap.parse_args(argv)
    tier_guard.forbid_tier1()
    evdir = Path(a.evdir)
    want = sorted(int(x) for x in a.expect_eps.split(","))
    picked, skipped = [], []
    for p in map(Path, a.files):
        m = HM_FILE.match(p.name)
        (picked if m else skipped).append((p, m))
    eps = [int(m.group(2)) for _, m in picked]
    if sorted(eps) != want:
        print(f"t3_onset_evidence: best_hm 的集应当恰好是 {want}、不重复，这次是 {sorted(eps)}", file=sys.stderr)
        return 2
    for p, _ in skipped:
        print(f"{p.name}\t—\t—\t不收：只收 Tier 3 best_hm 的 onset（按文件名，没有读）")
    ds = [json.loads(p.read_text()) for p, _ in picked]
    have = [d for d in ds if d.get("gate_ok")]
    cnt = Counter((tuple(d["thr"]), tuple(d["act"])) for d in have)
    common = None
    if cnt:
        (val, n), = cnt.most_common(1)
        common = val if n * 2 > len(have) else None   # 没有严格多数时整组都算 G4 不过
    per = {w: {f: ([], []) for f in FLAGS} for w in (W_MAIN, *W_SENS)}
    print("run\tep\t有起点\t起点四分位距\t最集中窗口占比\t" + "\t".join(f"{f} B/A/E" for f in FLAGS))
    for (p, fm), d in zip(picked, ds):
        run = d["run"]
        ep = int(fm.group(2))
        if run != fm.group(1):
            print(f"{run}\tep{ep}\t—\t门不过：文件名 {p.name} 与 JSON 里的 run 不符")
            continue
        if not d.get("gate_ok"):
            print(f"{run}\tep{ep}\t—\t门不过（fail_onset）：" + "；".join(d.get("problems", [])))
            continue
        if common is None or (tuple(d["thr"]), tuple(d["act"])) != common or abs(d["thr"][5] - 0.6) > 1e-9:
            print(f"{run}\tep{ep}\t—\t门不过（G4）：阈值 {d['thr']}、active_after_step {d['act']} 与其他 run 不同、没有严格多数或物体朝向阈值不是 0.6")
            continue
        T = int(d["step_count"])
        flags, why = flags_from_csv(evdir / f"ep{ep:02d}.csv", T)
        if flags is None:
            print(f"{run}\tep{ep}\t—\t门不过（G5）：{why}")
            continue
        act = d["act"]
        b0 = min(act[0], act[1], act[4], act[5]) + 1
        n_on = np.asarray(d["n_on"], dtype=float)
        q25, q50, q75 = (wquant(n_on, q) for q in (0.25, 0.5, 0.75))
        w0, w1, nmode = d["mode_window"]
        share = nmode / d["n_onset"] if d["n_onset"] else 0.0
        early = q50 < MIN_MEDIAN
        cells = []
        for f in FLAGS:
            for w in (W_MAIN, *W_SENS):
                a, b, e = enrich(n_on, flags[f], b0, w)
                ok = d["n_onset"] >= MIN_ONSET and not early and B_LO <= b <= B_HI
                if ok:
                    per[w][f][0].append(e)
                    per[w][f][1].append(b)
                if w == W_MAIN:
                    cells.append(f"{b:.2f}/{a:.2f}/{e:.2f}" + ("" if ok else "（不可比）"))
        iqr = f"{q25:g}–{q75:g}（中位 {q50:g}{'，太早' if early else ''}）" if d["n_onset"] else "—"
        print(f"{run}\tep{ep}\t{d['n_onset']}\t{iqr}\t{share:.0%}{'（单窗口）' if share >= 0.5 else ''}\t" + "\t".join(cells))
        if d["n_onset"]:
            inside = "，".join(f"{f} {int(flags[f][w0:w1 + 1].sum())}" for f in FLAGS)
            print(f"  起点最集中的窗口第 {w0}–{w1} 帧（{nmode} 个起点）里带旗标的帧数：{inside}")
        for fr in KNOWN_FLIPS.get(ep, ()):
            k = int(n_on[fr:fr + AFTER + 1].sum())
            base = (AFTER + 1) / max(1, T - b0)
            print(f"  ep{ep} 已确认的头尾颠倒帧 {fr} 之后 {AFTER} 帧内的起点 {k} 个，占有起点的 {k / max(1, d['n_onset']):.1%}；同样宽度的机会占比 {base:.1%}")
    for w in (W_MAIN, *W_SENS):
        tag = "主值" if w == W_MAIN else "敏感性，只报"
        print(f"== 窗口 {w} 帧（{tag}）")
        for f in FLAGS:
            es, bs = per[w][f]
            print(f"  {LABEL[f]}：{reading(es, bs)}；E = " + ", ".join(f"{e:.2f}" for e in es))
    print("说明：F_flip 只抓 15 帧以内的短翻转；成段翻转（整段朝向错）这张表测不了，读“不集中”不等于“翻转不是原因”。")
    print("T3_ONSET_EVIDENCE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
