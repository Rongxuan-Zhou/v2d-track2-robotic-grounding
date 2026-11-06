"""把所有已判定的逐帧满配 run 拼成一张误差表，看 MP-SR / SP-SR 卡在位置还是朝向（Mac 上跑，只用标准库，只读已落盘的结果）。

来源：
  results/day2/batch_verdict_*.txt 的 summary_json（不动基线、ADD-AUC、各成功率、MPPE、判定），只取逐帧回放、有判定的；
  最终 checkpoint 的评测文本 eval_<run>_s349999104.txt（results/day2 或 results/ml40/day2）：平均位置误差、平均朝向误差、失败原因；
  results/day2/sequence_stats_tier1.txt：序列特征（运动占比、总转角、单段最大转角、路程、物体半径、左右手接触帧）；
  results/diverge/support_shapes.txt：支撑面形状（空 / Cylinder / Cube）。
门槛（flash_chord evaluation/metrics.py:21–24，均为每个世界先对时间取平均再判）：MP-SR 位置 < 3 cm 且朝向 < 30°；SP-SR 位置 ≤ 10 cm 且朝向 ≤ 0.5 rad。
表里的平均误差是再对 4096 个世界取平均；“卡在”一栏按这个平均值对 MP 门槛，只是看典型世界卡在哪一项，不是逐世界的统计。
用法：python3 stages/evaluate/batch_error_table.py
"""
import glob
import json
import re
from pathlib import Path

FINAL = 349999104
MP_POS_M, MP_ORI_DEG = 0.03, 30.0


def verdicts() -> dict:
    out = {}
    for f in sorted(glob.glob("results/day2/batch_verdict_*.txt")):
        for line in open(f, encoding="utf-8"):
            if "summary_json " in line:
                d = json.loads(line.split("summary_json ", 1)[1])
                if d.get("playback") == "frame":
                    out[d["run"]] = d
    return out


def eval_fields(run: str) -> dict:
    for root in ("results/day2", "results/ml40/day2"):
        p = Path(root, f"eval_{run}_s{FINAL}.txt")
        if p.exists():
            t = p.read_text(encoding="utf-8")
            def num(key):
                m = re.search(rf"^\s*{re.escape(key)}\s+([-+0-9.eE]+)\s*$", t, re.M)
                return float(m.group(1)) if m else None
            causes = re.search(r"reference_tracking.failure_cause_fraction\s+(\{.*\})", t)
            return {"pos_m": num("object_position_error_m"), "ori_deg": num("object_orientation_error_deg"),
                    "causes": causes.group(1) if causes else "—"}
    return {}


def seq_stats() -> dict:
    out = {}
    lines = [l.rstrip("\n") for l in open("results/day2/sequence_stats_tier1.txt", encoding="utf-8")]
    header = next(l for l in lines if l.startswith("sequence_id\t")).split("\t")
    for l in lines:
        parts = l.split("\t")
        if len(parts) == len(header) and parts[0] != "sequence_id" and not l.startswith("#"):
            out[parts[0]] = dict(zip(header, parts))
    return out


def supports() -> dict:
    out = {}
    for l in open("results/diverge/support_shapes.txt", encoding="utf-8"):
        if l.startswith("#") or "\t" not in l:
            continue
        seq, rest = l.rstrip("\n").split("\t", 1)
        out[seq] = "Cube" if "Cube" in rest else ("Cylinder" if "Cylinder" in rest else "空")
    return out


def ranks(xs: list) -> list:
    """平均秩（处理并列）。"""
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def spearman(a: list, b: list) -> float:
    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb) if va and vb else float("nan")


def main() -> None:
    v = verdicts()
    st, sp = seq_stats(), supports()
    rows = []
    for run, d in v.items():
        e = eval_fields(run)
        s = st.get(d["sequence_id"], {})
        ep = re.search(r"-ep(\d+)-", run).group(1)
        pos, ori = e.get("pos_m"), e.get("ori_deg")
        stuck = []
        if pos is not None and pos >= MP_POS_M:
            stuck.append("位置")
        if ori is not None and ori >= MP_ORI_DEG:
            stuck.append("朝向")
        rows.append((d["maniptrans_sr"], ep, d, e, s, sp.get(d["sequence_id"], "?"), "+".join(stuck) or "—"))
    rows.sort(key=lambda r: (-r[0], int(r[1])))
    print(f"逐帧满配、最终 checkpoint、已判定 {len(rows)} 条（按 MP-SR 从高到低）")
    print("ep | 物体 | 判定 | ADD-AUC（不动） | MP-SR | SP-SR | chord | 平均位置 | 平均朝向 | 卡在 | 运动占比 | 总转角 | 单段最大转角 | 路程 | 半径 | 支撑 | 左/右手接触帧 | 失败原因（chord）")
    for mp, ep, d, e, s, sup, stuck in rows:
        name = re.sub(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_", "", d["sequence_id"])
        pos = f"{e['pos_m'] * 100:.1f} cm" if e.get("pos_m") is not None else "—"
        ori = f"{e['ori_deg']:.1f}°" if e.get("ori_deg") is not None else "—"
        print(f"ep{ep} | {name} | {'通过' if d['verdict'] == 'PASS' else '不通过'} | {d['add_auc']:.3f}（{d['bar']:.3f}） | "
              f"{d['maniptrans_sr']:.3f} | {d['spider_sr_uncentered']:.3f} | {d['chord_sr']:.3f} | {pos} | {ori} | {stuck} | "
              f"{s.get('moving', '—')} | {s.get('rot_total_deg', '—')}° | {s.get('rot_max_deg', '—')}° | {s.get('path_m', '—')} m | "
              f"{s.get('radius_m', '—')} m | {sup} | {s.get('contact_L', '—')}/{s.get('contact_R', '—')} | {e.get('causes', '—')}")
    # 描述性的秩相关（22 条、事后看的，只作线索）
    feats = {"运动占比": lambda s: float(s["moving"].rstrip("%")), "总转角": lambda s: float(s["rot_total_deg"]),
             "单段最大转角": lambda s: float(s["rot_max_deg"]), "路程": lambda s: float(s["path_m"]),
             "物体半径": lambda s: float(s["radius_m"])}
    ok = [(d, s) for _, _, d, _, s, _, _ in rows if s]
    print(f"\nSpearman 秩相关（n={len(ok)}，事后描述，不作检验）：")
    for fname, f in feats.items():
        xs = [f(s) for _, s in ok]
        print(f"  {fname}：对 MP-SR {spearman(xs, [d['maniptrans_sr'] for d, _ in ok]):+.2f}，"
              f"对 chord {spearman(xs, [d['chord_sr'] for d, _ in ok]):+.2f}，"
              f"对 ADD-AUC 相对不动的增量 {spearman(xs, [d['add_auc'] - d['bar'] for d, _ in ok]):+.2f}")


if __name__ == "__main__":
    main()
