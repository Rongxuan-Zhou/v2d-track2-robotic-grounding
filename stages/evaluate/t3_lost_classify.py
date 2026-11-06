"""Tier 3 跟丢的集按原因分两类（Mac 上跑，只用标准库，只读已落盘的记录）：动作本身难（同一集在 Tier 2 也失败，多半是走路时摔）还是重建有问题（Tier 2 做得成、Tier 3 做不成）。
用途：10/26 定剩下 9 天的算力押在 Tier 3 重建还是 Tier 1/2 防摔。只是诊断，不进任何方法与选择，也不用 Tier 1 的数据。
Tier 3 每集的现状：t3m1022a 清单里那条 run，加上 gb5 重训的 run（有评测记录的话），取 4096 世界对自己参考 MPPE 最低的一条；SP-SR ≥ 0.5 且 MPPE ≤ 10 cm 记“跟上”（t3g5 的 R1）。
Tier 2 对照：现在交的 b1014 清单里同一集那条 run，SP-SR ≥ 0.5 记“做得成”。
分类：Tier 3 跟丢、Tier 2 也做不成 →“动作难”；Tier 3 跟丢、Tier 2 做得成 →“重建”。
另报两样旁证（只描述，门槛只用于标注）：Tier 3 那条 run 失败世界里骨盆失败的比例（高 = 摔倒为主）；gb5 参考里首次接触时物体已平移 > 0.10 m，或“无接触而在动”的帧 > 150（`results/t3g5/ref_contact_summary_gb5_mf4090.txt`）。
用法：python3 stages/evaluate/t3_lost_classify.py   最后一行 T3_LOST_CLASSIFY_DONE
"""
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def ep_seq() -> dict:
    out = {}
    for f in ("results/t3static/floor_mf4090.txt", "results/ml40/t3static/floor_ml40.txt"):
        for line in (REPO / f).read_text(encoding="utf-8").splitlines():
            m = re.match(r"^(\d+)\t.*sequence_id=([^/]+)/", line)
            if m:
                out[int(m.group(1))] = m.group(2)
    return out


def ev(run: str, step: str = "349999104"):
    for h in sorted(REPO.glob(f"results/**/eval_{run}_s{step}.txt")):
        for line in h.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.startswith("summary_json "):
                d = json.loads(line[len("summary_json "):])
                d = {**d, **d.get("metrics", {})}
                fc = d.get("reference_tracking", {}).get("failure_cause_fraction") or d.get("failure_cause_fraction") or {}
                sp = d.get("spider_sr", d.get("spider_sr_uncentered"))
                return {"run": run, "mppe": d["mppe_cm"], "sp": sp, "mp": d.get("maniptrans_sr"), "pelvis": fc.get("pelvis")}
    return None


def rows(path: str):
    for line in (REPO / path).read_text(encoding="utf-8").splitlines():
        f = line.split("\t")
        if len(f) >= 4 and not line.startswith("#") and f[1].isdigit():
            yield int(f[1]), f[2], f[3]


def contact() -> dict:
    out = {}
    for line in (REPO / "results/t3g5/ref_contact_summary_gb5_mf4090.txt").read_text(encoding="utf-8").splitlines():
        m = re.search(r"sequence_id=([^/]+)/.*首次接触时已平移 m ([0-9.]+)\t无接触而在动的帧 (\d+)", line)
        if m:
            out[m.group(1)] = (float(m.group(2)), int(m.group(3)))
    return out


def main() -> int:
    seq, con = ep_seq(), contact()
    t3 = {ep: [ev(run, step)] for ep, run, step in rows("stages/evaluate/exports/t3_select_t3m1022a.tsv")}
    for ep in range(30):
        for run in (f"t3-ep{ep}-w2048-cap4m-r3a-gb5-mf4090", f"t3-ep{ep}-w2048-cap10m-r3a-gb5-ml40"):
            e = ev(run)
            if e:
                t3.setdefault(ep, []).append(e)
    t2 = {ep: ev(run, step) for ep, run, step in rows("stages/evaluate/exports/t2_select_b1014.tsv")}
    print("集\tTier 3 最好的 run\tTier 3 SP-SR\tTier 3 MPPE\t骨盆失败比例\tTier 2 SP-SR\tTier 2 MP-SR\tgb5 参考旁证\t分类")
    groups = {"跟上": [], "动作难": [], "重建": [], "缺读数": []}
    for ep in range(30):
        cands = [e for e in t3.get(ep, []) if e]
        best = min(cands, key=lambda e: e["mppe"]) if cands else None
        e2 = t2.get(ep)
        c = con.get(seq.get(ep, ""), None)
        note = ""
        if c:
            flags = []
            if c[0] > 0.10:
                flags.append(f"接触前已移 {c[0]:.2f} m")
            if c[1] > 150:
                flags.append(f"无接触而动 {c[1]} 帧")
            note = "、".join(flags)
        if best is None or e2 is None:
            cls = "缺读数"
        elif best["sp"] >= 0.5 and best["mppe"] <= 10:
            cls = "跟上"
        elif e2["sp"] < 0.5:
            cls = "动作难"
        else:
            cls = "重建"
        groups[cls].append(ep)
        fmt = lambda x, n=3: "—" if x is None else f"{x:.{n}f}"
        print(f"ep{ep}\t{best['run'] if best else '—'}\t{fmt(best and best['sp'])}\t{fmt(best and best['mppe'], 1)}\t{fmt(best and best['pelvis'], 2)}\t"
              f"{fmt(e2 and e2['sp'])}\t{fmt(e2 and e2['mp'])}\t{note}\t{cls}")
    for k, v in groups.items():
        print(f"{k}（{len(v)}）：{v}")
    print("T3_LOST_CLASSIFY_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
