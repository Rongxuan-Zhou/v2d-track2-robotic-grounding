"""Tier 1 逐条选 checkpoint：给每个候选的 1 世界导出轨迹单独打包打分，按登记规则每条 episode 选一个，再把选中的 30 条打成提交 CSV（容器里执行）。

计划 agent-doc/plan-t1-per-episode-select-2026-10-08.md。打分用官方打包器 tools/pack_submission.py 本身：每个候选建一个只含这一条 episode 的结果目录、
一份把 track_2_tier1 的 scored_episodes 改成这一条的 tracks.json，在进程内调 main()，替换它的 crosscheck 拿到全精度的逐条四项分数
（与计分端同一个 op.episode_metric、同一套发布网格顶点）。
候选目录：<cand>/<run>/episode_XXXXXX.{json,parquet}（1 世界导出）与 eval_349999104_w4096.json（4096 世界评测，只报）。

用法（容器内，kit 依赖在 PYTHONPATH）：
  python select_t1.py --cand <dir> --kit <kit 根> --dataset-root <数据根> --commit-url <URL> --current <清单.tsv>… \
      --out-json <逐候选分数.json> --out-tsv <选择清单.tsv> [--pack <csv>] [--gate AUC,SP,MP,MPPE --gate-label <名>]
--gate：打分门 B 对照的四项合计，默认 10/3 Tier 1 CSV 的 4 位小数值；10/12 起的补候选（计划 agent-doc/plan-next-training-2026-10-12.md
第 0 步第 4 条）传 10/9 打包的合计。选择规则里的 0.3780、25.1706 是登记的常数，不随 --gate 变。
"""

import argparse
import contextlib
import hashlib
import io
import json
import shutil
import sys
import tempfile
from pathlib import Path

METRICS = ("add_auc", "spider_sr", "maniptrans_sr", "mppe_cm")
GATE = {"add_auc": 0.3780, "spider_sr": 0.5333, "maniptrans_sr": 0.2667, "mppe_cm": 25.1706}  # 10/3 Tier 1 CSV 的本地分数（4 位小数）
GATE_TOL = 1e-4
STEP = 349999104


NORM_AUC, NORM_MPPE = 0.3780, 25.1706  # 10/3 Tier 1 合计，用于成功率之后的组合分


def parse_gate(s: str) -> dict:
    """--gate 的值：四个数，逗号分隔，顺序同 METRICS。"""
    v = [float(x) for x in s.split(",")]
    if len(v) != len(METRICS):
        raise argparse.ArgumentTypeError(f"--gate 要 {len(METRICS)} 个数（{','.join(METRICS)}），给了 {len(v)} 个")
    return dict(zip(METRICS, v))


def merge_sha_tables(paths) -> dict:
    """导出链结果表 → {run: (状态, json sha256, parquet sha256)}，只收 Tier 1 的 9 列行。同一条 run 在几张表里出现时（前一段没导出完、
    后一段才导出 ok，例如 t1xa 的 ep21 是 no_done、t1xa2 的 ep21 是 ok），状态 ok 的那一行算数，与表的读入顺序无关；
    两张表里都是 ok 但 sha256 不同就停。10/13 之前是后读的表直接覆盖前面的，表名按字母排在后面的旧表会把 ok 盖成 no_done。"""
    shas = {}
    for t in paths:
        for line in open(t):
            f = line.rstrip("\n").split("\t")
            if len(f) != 9 or f[0] != "1":
                continue
            row, old = (f[4], f[7], f[8]), shas.get(f[2])
            if old is None or (old[0] != "ok" and row[0] == "ok"):
                shas[f[2]] = row
            elif old[0] == row[0] == "ok" and old != row:
                raise SystemExit(f"{f[2]} 在两张导出表里都是 ok，sha256 不同：{old[1:]} 对 {row[1:]}（{t}）")
    return shas


def sort_key(c):
    """登记的选择规则（第 2 版）：MP-SR → SP-SR → AUC / 0.3780 − MPPE / 25.1706（大者优先）。缺指标的候选不进这里（先剔除）。"""
    return (c["maniptrans_sr"], c["spider_sr"], c["add_auc"] / NORM_AUC - c["mppe_cm"] / NORM_MPPE)


def choose(cands, current):
    """同一条 episode 的候选里选一个；规则平手时保留现在交的那条，再平手按 run 名。"""
    best = max(sort_key(c) for c in cands)
    top = [c for c in cands if sort_key(c) == best]
    for c in top:
        if c["run"] == current:
            return c
    return sorted(top, key=lambda c: c["run"])[0]


def load_packer(kit):
    sys.path.insert(0, str(Path(kit) / "tools"))
    import pack_submission as ps
    return ps


def pack(ps, kit, results, dataset_root, commit_url, out_csv, episodes=None):
    """调官方打包器；episodes 给了就只打这几条（改 tracks.json 的 scored_episodes）。返回 crosscheck 的全精度合计。"""
    cfg_path = Path(kit) / "config/tracks.json"
    tmp = None
    if episodes is not None:
        tmp = Path(tempfile.mkdtemp())
        cfg = json.loads(cfg_path.read_text())
        cfg["tracks"]["track_2_tier1"]["scored_episodes"] = sorted(int(e) for e in episodes)
        cfg_path = tmp / "tracks.json"
        cfg_path.write_text(json.dumps(cfg))
    captured = {}
    orig = ps.crosscheck

    def wrap(*a, **k):
        captured.update(orig(*a, **k))
        return captured

    ps.crosscheck = wrap
    sys.argv = ["pack_submission.py", "--track", "track_2_tier1", "--results", str(results), "--dataset-root", str(dataset_root),
                "--code-commit", commit_url, "-o", str(out_csv), "--config", str(cfg_path)]
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rc = ps.main()
    finally:
        ps.crosscheck = orig
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)
    if rc not in (0, None):
        raise RuntimeError(f"打包器退出码 {rc}：{buf.getvalue()[-2000:]}")
    return dict(captured), buf.getvalue()


def episode_of(d):
    eps = sorted(Path(d).glob("episode_*.parquet"))
    if len(eps) != 1:
        raise RuntimeError(f"{d} 里应恰好一个 episode_*.parquet，有 {len(eps)} 个")
    return int(eps[0].stem.split("_")[1]), eps[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--kit", required=True)
    ap.add_argument("--dataset-root", required=True)
    ap.add_argument("--commit-url", required=True)
    ap.add_argument("--current", nargs="+", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-tsv", required=True)
    ap.add_argument("--pack")
    ap.add_argument("--sha-tables", nargs="+", required=True, help="导出链结果表（runs/exports/*.tsv）；候选文件的 sha256 要与表里状态 ok 的那一行相同")
    ap.add_argument("--gate", type=parse_gate, default=dict(GATE), help="打分门 B 的四项合计，逗号分隔，顺序 " + ",".join(METRICS))
    ap.add_argument("--gate-label", default="10/3", help="打分门 B 的来源，只用于打印")
    a = ap.parse_args()
    shas = merge_sha_tables(a.sha_tables)
    ps = load_packer(a.kit)
    current = {}
    for f in a.current:
        mach = Path(f).stem
        for line in open(f):
            t, ep, run, step = line.split()[:4]
            if t == "1":
                current[int(ep)] = run
    if len(current) != 30:
        raise SystemExit(f"现在的清单里 Tier 1 有 {len(current)} 条，应为 30")
    mesh = Path(a.dataset_root) / "track_2/tier_1_multiview_caption/mesh"
    if not mesh.is_dir():
        raise SystemExit(f"没有网格目录 {mesh}：AUC 会被打包器默默跳过")
    import numpy, pandas, pyarrow, trimesh
    print(f"版本：numpy {numpy.__version__} pandas {pandas.__version__} pyarrow {pyarrow.__version__} trimesh {trimesh.__version__}")
    scores, excluded = {}, []
    for d in sorted(Path(a.cand).iterdir()):
        if not d.is_dir():
            continue
        try:
            ep, pq = episode_of(d)
            st, js, ps_ = shas.get(d.name, (None, None, None))
            got = (hashlib.sha256(pq.with_suffix(".json").read_bytes()).hexdigest(), hashlib.sha256(pq.read_bytes()).hexdigest())
            if st != "ok" or got != (js, ps_):
                excluded.append(f"{d.name}：导出表里状态 {st}，sha256 {'相同' if got == (js, ps_) else '不同或没有记录'}")
                continue
            with tempfile.TemporaryDirectory() as tmp:
                res = Path(tmp) / "res"
                res.mkdir()
                for f in d.glob(f"episode_{ep:06d}.*"):
                    shutil.copy(f, res / f.name)
                overall, _ = pack(ps, a.kit, res, a.dataset_root, a.commit_url, Path(tmp) / "x.csv", episodes=[ep])
        except (Exception, SystemExit) as e:  # noqa: BLE001
            excluded.append(f"{d.name}：打包失败 {type(e).__name__} {str(e)[-300:]}")
            continue
        miss = [m for m in METRICS if overall.get(m) is None]
        if miss:
            excluded.append(f"{d.name}：缺指标 {miss}")
            continue
        w4096 = None
        j = d / f"eval_{STEP}_w4096.json"
        if j.exists():
            w4096 = json.loads(j.read_text())["metrics"].get("maniptrans_sr")
        scores[d.name] = {"run": d.name, "episode": ep, **{m: float(overall[m]) for m in METRICS}, "mp_sr_4096": w4096}
        print(f"  ep{ep:<3} {d.name}: " + " ".join(f"{m}={overall[m]:.6f}" for m in METRICS) + f" | 4096 世界 MP-SR {w4096}", flush=True)
    for line in excluded:
        print(f"  剔除：{line}")
    # 打分门 A：整份打包现在交的 30 条 = 30 条逐条分数的平均（≤ 1e-9）；B：整份与 --gate 给的合计（默认 10/3 打印值，≤ 1e-4）
    missing = [ep for ep, r in current.items() if r not in scores]
    if missing:
        raise SystemExit(f"现在交的 run 缺打分：{missing}")
    with tempfile.TemporaryDirectory() as tmp:
        res = Path(tmp) / "res"
        res.mkdir()
        for ep, run in current.items():
            for f in (Path(a.cand) / run).glob(f"episode_{ep:06d}.*"):
                shutil.copy(f, res / f.name)
        whole, _ = pack(ps, a.kit, res, a.dataset_root, a.commit_url, Path(tmp) / "whole.csv")
    gate_ok = True
    for m in METRICS:
        mean = sum(scores[current[ep]][m] for ep in current) / len(current)
        a_ok = abs(whole[m] - mean) <= 1e-9
        b_ok = abs(whole[m] - a.gate[m]) <= GATE_TOL
        gate_ok &= a_ok and b_ok
        print(f"GATE {m}：整份 {whole[m]:.9f}，逐条平均 {mean:.9f}（A {'OK' if a_ok else '不符'}），{a.gate_label} {a.gate[m]:.6f}（B {'OK' if b_ok else '不符'}）")
    Path(a.out_json).write_text(json.dumps(scores, indent=1, ensure_ascii=False))
    if not gate_ok:
        print("SELECT_T1 STOP：打分门不过，不选择")
        raise SystemExit(1)
    by_ep = {}
    for c in scores.values():
        by_ep.setdefault(c["episode"], []).append(c)
    rows, changed, conflicts = [], [], []
    for ep in sorted(by_ep):
        cands = by_ep[ep]
        c = choose(cands, current.get(ep))
        mach = c["run"].rsplit("-", 1)[1]
        rows.append(f"1\t{ep}\t{c['run']}\t{STEP}\t{mach}")
        if c["run"] != current.get(ep):
            changed.append(ep)
        for o in cands:
            if o is c:
                continue
            worse = [m for m in METRICS if (c[m] < o[m]) if m != "mppe_cm"] + (["mppe_cm"] if c["mppe_cm"] > o["mppe_cm"] else [])
            if worse:
                conflicts.append(f"ep{ep} 选 {c['run']}，在 {','.join(worse)} 上不如 {o['run']}")
        print(f"CHOSEN ep{ep}: {c['run']}（现在 {current.get(ep)}）" + " ".join(f"{m}={c[m]}" for m in METRICS) + f" 4096 世界 MP-SR {c['mp_sr_4096']}")
    if sorted(by_ep) != list(range(30)):
        raise SystemExit(f"有分数的 episode 不是 0–29：{sorted(by_ep)}")
    Path(a.out_tsv).write_text("\n".join(rows) + "\n")
    print(f"换了的 episode：{changed}")
    for line in conflicts:
        print(f"  帕累托冲突（只报）：{line}")
    if a.pack:
        if Path(a.pack).exists():
            raise SystemExit(f"{a.pack} 已存在，拒绝覆盖")
        final = Path(a.pack).with_suffix(".results")
        if final.exists():
            raise SystemExit(f"{final} 已存在")
        final.mkdir(parents=True)
        for line in rows:
            _, ep, run, _, _ = line.split("\t")
            for f in (Path(a.cand) / run).glob(f"episode_{int(ep):06d}.*"):
                shutil.copy(f, final / f.name)
        overall, log = pack(ps, a.kit, final, a.dataset_root, a.commit_url, Path(a.pack))
        print(log[-1500:])
        print("PACKED " + " ".join(f"{m}={overall.get(m):.6f}" for m in METRICS))
    print("SELECT_T1_DONE")


if __name__ == "__main__":
    main()
