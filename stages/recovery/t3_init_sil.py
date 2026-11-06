"""Tier 3 初始化试点核对 B 的补充：剪影检查（另一位用户 t3_object_check.py 的输出）在被替换帧与其余帧上分开报（ml40 主机上跑，只用标准库；
只读本用户 OUT 下两版的 object_check.json 与另一位用户 work_init 里的 fp_init.json）。判读规则见 agent-doc/experiments/2026-10-16-t3-init-pilot-check.md。
object_check.json 的 rows 每行一帧（步长 3 抽样），取 frame 与 iou_visible；被替换帧取 fp_init.json 的 replaced_runs（左闭右开）。
用法：python3 stages/recovery/t3_init_sil.py <OUT> <tier3 根> <集名>...；最后一行 T3_INIT_SIL_DONE。
"""
import json
import statistics
import sys
from pathlib import Path


def rows(p: Path) -> dict:
    d = json.loads(p.read_text())
    out = {}
    for r in d["rows"]:
        f = r.get("frame", r.get("frame_index"))
        v = r.get("iou_visible")
        if f is not None and v is not None:
            out[int(f)] = float(v)
    if not out:
        raise SystemExit(f"{p}：rows 里没有 frame 与 iou_visible，第一行的键 {list(d['rows'][0]) if d['rows'] else '无'}")
    return out


def med(xs: list) -> str:
    return f"{statistics.median(xs):.3f}（{len(xs)} 帧）" if xs else "无"


def main() -> int:
    out, root = Path(sys.argv[1]), Path(sys.argv[2])
    for name in sys.argv[3:]:
        seq = f"episode_{int(name[2:4]):06d}"
        fi = json.loads((root / "runs/work_init" / name / "cari4d" / seq / "foundationpose" / "fp_init.json").read_text())
        rep = set()
        for a, b in fi["replaced_runs"]:
            rep.update(range(int(a), int(b)))
        rf, ri = rows(out / f"work_fp_{name}" / "object_check.json"), rows(out / f"work_init_{name}" / "object_check.json")
        common = sorted(set(rf) & set(ri))
        on = [f for f in common if f in rep]
        off = [f for f in common if f not in rep]
        better = sum(ri[f] > rf[f] + 0.05 for f in on)
        worse = sum(ri[f] < rf[f] - 0.05 for f in on)
        print(f"{name}：两版都查了 {len(common)} 帧，其中被替换帧 {len(on)}。被替换帧 IoU 中位 work_fp {med([rf[f] for f in on])}、"
              f"work_init {med([ri[f] for f in on])}；逐帧变好（+0.05 以上）{better}、变差（−0.05 以上）{worse}。"
              f"其余帧 work_fp {med([rf[f] for f in off])}、work_init {med([ri[f] for f in off])}")
    print("T3_INIT_SIL_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
