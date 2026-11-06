"""把几个 Tier 3 帧参考目录按集合并成一个（Mac 上跑，只用标准库；只读帧参考目录）。计划 agent-doc/plan-t3-kfix-2026-10-16.md 方案 4（复核 MAJOR 8）。

为什么：`pack_tier.sh` 只收一个帧参考目录。k-fix 之后，换成新参考的集用新参考生成的 frame refs（不转偏航，标定内参已含偏航），保留旧参考的集沿用
`rotate_frefs_t3.py` 转过偏航的 frame refs（例如 frame_refs_rot_gridG_yaw），两边要合成一个目录。
做法：
  - 每个源目录读 manifest_*.tsv（表头必须是 frefs_t3_match.COLS），逐行核 npz 存在、文件名是 episode_XXXXXX.npz、sha256 与 manifest 相符；
  - 每个源目录只取命令行指定的集（--take 目录=集,集,…）；同一集只许来自一个源目录，指定了却不在那个目录的 manifest 里就拒绝；
  - 复制 npz 到目标目录，复制后再核一次 sha256；按源 manifest 的文件名分组写回 manifest_<机器>.tsv（列与源相同，行原样）；
  - 写 MERGE.json：每个源目录、它的 manifest 的 sha256（以及若有 ROTATION.json 则记它的 sha256 与 label）、取了哪些集。
目标目录已存在就拒绝。
用法：python3 stages/evaluate/merge_frefs_t3.py --out <目标目录> --take <源目录 1>=0,18,20 --take <源目录 2>=1,2,3 …
--take-plain <源目录>=<集,…>：同 --take，但源目录有 ROTATION.json 就拒绝（新参考那一侧用它：标定内参已含偏航，不能再转一次）。
最后一行 MERGE_FREFS_OK n（n 是集数）。测试：stages/evaluate/test_merge_frefs_t3.py。
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from frefs_t3_match import COLS


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read_manifests(src: Path) -> dict:
    """{episode: (manifest 文件名, 行字典)}；表头、文件名、sha256 不对就拒绝。"""
    out = {}
    mans = sorted(src.glob("manifest_*.tsv"))
    if not mans:
        raise SystemExit(f"{src}：没有 manifest_*.tsv")
    for m in mans:
        lines = m.read_text(encoding="utf-8").splitlines()
        if not lines or lines[0].split("\t") != COLS:
            raise SystemExit(f"{m}：表头不是 {COLS}")
        for ln in lines[1:]:
            if not ln.strip():
                continue
            f = ln.split("\t")
            if len(f) != len(COLS):
                raise SystemExit(f"{m}：有一行不是 {len(COLS)} 列")
            r = dict(zip(COLS, f))
            ep = int(r["episode"])
            npz = src / r["file"]
            if r["file"] != f"episode_{ep:06d}.npz" or not npz.is_file() or sha256(npz) != r["sha256"]:
                raise SystemExit(f"{m.name} ep{ep}：npz 缺失、文件名不对或 sha256 与 manifest 不符")
            if ep in out:
                raise SystemExit(f"{src}：ep{ep} 在 manifest 里出现两次")
            out[ep] = (m.name, r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--take", action="append", default=[], help="<源目录>=<集,集,…>")
    ap.add_argument("--take-plain", action="append", default=[],
                    help="同 --take，但这个源目录不许有 ROTATION.json（新参考的 frame refs：标定内参已含偏航，不能再转一次；复核 MINOR）")
    a = ap.parse_args()
    if not a.take and not a.take_plain:
        raise SystemExit("至少给一个 --take 或 --take-plain")
    if a.out.exists():
        raise SystemExit(f"{a.out} 已存在")
    plan, seen = [], {}
    for item, plain in [(x, False) for x in a.take] + [(x, True) for x in a.take_plain]:
        if "=" not in item:
            raise SystemExit(f"--take 要写成 <源目录>=<集,集,…>，收到 {item!r}")
        d, eps = item.rsplit("=", 1)
        src = Path(d)
        if plain and (src / "ROTATION.json").exists():
            raise SystemExit(f"{src}：--take-plain 的源目录不许有 ROTATION.json（新参考的 frame refs 不能再转偏航）")
        want = sorted({int(e) for e in eps.split(",") if e.strip()})
        rows = read_manifests(src)
        for ep in want:
            if ep not in rows:
                raise SystemExit(f"{src}：manifest 里没有 ep{ep}")
            if ep in seen:
                raise SystemExit(f"ep{ep} 同时来自 {seen[ep]} 与 {src}")
            seen[ep] = src
        plan.append((src, want, rows))
    a.out.mkdir(parents=True)
    by_manifest, record = {}, {"sources": []}
    for src, want, rows in plan:
        rot = src / "ROTATION.json"
        rec = {"dir": str(src), "episodes": want,
               "manifests": {m.name: sha256(m) for m in sorted(src.glob("manifest_*.tsv"))}}
        if rot.is_file():
            rec["rotation_json_sha256"] = sha256(rot)
            rec["rotation_label"] = json.loads(rot.read_text(encoding="utf-8")).get("label")
        record["sources"].append(rec)
        for ep in want:
            mname, r = rows[ep]
            dst = a.out / r["file"]
            shutil.copy2(src / r["file"], dst)
            if sha256(dst) != r["sha256"]:
                raise SystemExit(f"{dst}：复制后 sha256 不对")
            by_manifest.setdefault(mname, []).append(r)
    for mname, rs in by_manifest.items():
        rs.sort(key=lambda r: int(r["episode"]))
        (a.out / mname).write_text("\t".join(COLS) + "\n" + "".join("\t".join(r[c] for c in COLS) + "\n" for r in rs), encoding="utf-8")
    (a.out / "MERGE.json").write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"合并 {len(seen)} 集 → {a.out}：" + "；".join(f"{Path(s['dir']).name} {len(s['episodes'])} 集" for s in record["sources"]))
    print(f"MERGE_FREFS_OK {len(seen)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
