"""不经 run，直接从一个 Tier 3 档位目录的 motion 源出帧参考（训练机上、仓库根执行，要 numpy、pyarrow；ml40 经 scripts/dev/in-flash-chord 在容器里跑）。
计划 agent-doc/plan-t3b2-2026-10-28.md 第 1 步第 1 条：给还没有 run 的参考版本（gb5、新参考）出第 0 帧朝向表。
每集：在 runs/motion_src/<档位目录>/ 下按 v2d_meta.json 的 episode 找到序列，交给 make_frame_refs_t3.build()（核对与写法与打包用的帧参考完全相同）。
清单 <out>/manifest_motion.tsv 的列同 make_frame_refs_t3.py，run 写 -、步数写 0；align_frefs_t3.py 照常读。
进程开头 forbid_tier1()：此后读 Tier 1 的任何文件都会抛错。
用法：python3 stages/evaluate/frefs_from_motion_t3.py <档位目录> <episode,...> --out <目录>   最后一行 FRAME_REFS_MOTION_OK n 或 FRAME_REFS_MOTION_FAIL n。
测试：stages/evaluate/test_frefs_from_motion_t3.py。
"""
import argparse
import json
import re
import sys
from pathlib import Path

import make_frame_refs_t3 as mf

TIER_DIR = re.compile(r"tier3(?:_[a-z0-9]+)?")
HEADER = "episode\trun\tstep\ttier_dir\tsequence\tframes\tposes_sha256\tsource_parquet_sha256\tfile\tsha256\n"


def sequences(tdir: str) -> dict:
    """motion_src/<档位目录> 下 episode → 序列（v2d_meta.json 的 episode 与 sequence_id；同一集出现两次就拒绝）。"""
    out = {}
    for meta in sorted((mf.ROOT / "runs/motion_src" / tdir).glob("*/v2d_meta.json")):
        m = json.loads(meta.read_text())
        ep = int(m["episode"])
        if ep in out:
            raise ValueError(f"{tdir} 里 episode {ep} 有两个序列：{out[ep]}、{meta.parent.name}")
        if m.get("sequence_id") != meta.parent.name:
            raise ValueError(f"{meta} 的 sequence_id {m.get('sequence_id')} 与目录名不同")
        out[ep] = meta.parent.name
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tier_dir")
    ap.add_argument("episodes", help="逗号分隔")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    sys.path.insert(0, str(mf.ROOT / "libs/v2d_infra"))
    from tier_guard import forbid_tier1
    forbid_tier1()   # 档位隔离：此后读 Tier 1 的任何文件都会抛错
    if not TIER_DIR.fullmatch(a.tier_dir):
        sys.exit(f"档位目录 {a.tier_dir!r} 不是 tier3 或 tier3_<小写字母数字>")
    eps = [int(x) for x in a.episodes.split(",") if x.strip()]
    if not eps or len(set(eps)) != len(eps):
        sys.exit(f"episode 列表为空或有重复：{a.episodes}")
    seqs = sequences(a.tier_dir)
    if a.out.exists():
        sys.exit(f"{a.out} 已存在，拒绝覆盖")
    a.out.mkdir(parents=True)
    print(f"== Tier 3 帧参考（motion 源，不经 run）：{a.tier_dir} 的 {len(eps)} 集 → {a.out}")
    lines, bad = [], 0
    for ep in eps:
        try:
            if ep not in seqs:
                raise ValueError(f"motion_src/{a.tier_dir} 里没有 episode {ep}")
            seq = seqs[ep]
            T, psha, srcsha, f, sha, ortho, back, nfill = mf.build(ep, a.tier_dir, seq, a.out)
        except Exception as e:   # noqa: BLE001  逐条报
            print(f"  FAIL ep{ep}：{type(e).__name__}: {e}")
            bad += 1
            continue
        lines.append(f"{ep}\t-\t0\t{a.tier_dir}\t{seq}\t{T}\t{psha}\t{srcsha}\t{f}\t{sha}\n")
        print(f"  ok   ep{ep:<2} {a.tier_dir}/{seq}，{T} 帧；不可见 {nfill} 帧按前一帧补；|RᵀR − I| 最大 {ortho:.1e}，转回最大差 {back:.1e}")
    (a.out / "manifest_motion.tsv").write_text(HEADER + "".join(lines), encoding="utf-8")
    print(f"FRAME_REFS_MOTION_OK {len(lines)}" if bad == 0 else f"FRAME_REFS_MOTION_FAIL {bad}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
