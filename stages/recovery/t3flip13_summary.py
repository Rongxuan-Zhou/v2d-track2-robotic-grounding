"""t3flip13 的 F0（负对照）与 F1（注入阳性对照，只报）汇总（训练机上用 python3 跑，只用标准库）。计划 agent-doc/plan-t3flip13-2026-10-27.md 第 2.1 版第 2 节。

读 t3_fix_tex.py 的两个输出目录：
  F0：每集任一变体对负对照（平移 1 px、换采样点重渲、绕最小惯性轴 5°、三条主轴各 ±5°、±10°）的 accept 数，合计要为 0；另列真实候选的 accept；
  F1：每集注入了哪个翻转、注入的帧段、注入段里判换回同一翻转的段数 / 证据够的段数、判换成别的候选的段数。
用法：python3 stages/recovery/t3flip13_summary.py <F0 目录> <F1 目录>   最后一行 T3FLIP13_SUMMARY_DONE <F0 合计> <F1 找回的集数>/<集数>
"""
import json
import sys
from pathlib import Path


def main() -> None:
    null_dir, inj_dir = Path(sys.argv[1]), Path(sys.argv[2])
    print("== F0 负对照（门槛 0.27，±5°、±10° 三轴 + 平移 1 px + 换采样点）")
    total = 0
    for f in sorted(null_dir.glob("ep*.json")):
        d = json.loads(f.read_text())
        if "decisions" not in d:
            print(f"{d['episode']}：{d.get('skip') or '渲染门槛不过'}")
            continue
        total += d["null_variant_accepts"]
        hits = [(r["start"], r["end"], v, x["name"], round(x.get("gap", 0), 3)) for r in d["decisions"]["null"] for v, x in r["variants"].items() if x["decision"] == "accept"]
        print(f"{d['episode']}：负对照 accept {d['null_variant_accepts']} {hits[:4]}；真实 accept {d['accepts']['real']}；负对照 {len(d['tilt_nulls']) + 3} 个")
    print(f"F0 合计 {total}")
    print("== F1 注入阳性对照（只报）")
    found = n = 0
    for f in sorted(inj_dir.glob("ep*.json")):
        d = json.loads(f.read_text())
        if "decisions" not in d:
            print(f"{d['episode']}：{d.get('skip') or '渲染门槛不过'}")
            continue
        n += 1
        a, b = d["inject_found"].split("/")
        ok = int(a) > 0 and int(a) * 2 >= int(b)
        found += int(ok)
        print(f"{d['episode']}：注入 {d['inject']} 帧 {d['inject_frames']}，找回 {d['inject_found']}，判错 {d['inject_wrong']}")
    print(f"T3FLIP13_SUMMARY_DONE {total} {found}/{n}")


if __name__ == "__main__":
    main()
