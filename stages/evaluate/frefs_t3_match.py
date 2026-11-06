"""打包 Tier 3 之前，核帧参考目录与结果目录逐集对得上（Mac 上由 pack_tier.sh 调用；只用标准库）。计划 agent-doc/plan-r3o-2026-10-15.md 2.4.1
（按 Fable 5.1 的审查补：只看 npz 在不在，管不住“帧参考来自另一条 run”）。

结果目录里每条 episode_XXXXXX.json（评测器写的 metrics JSON）记着导出用的 checkpoint（/repo/runs/flash_chord/<run>/policy_<步数>.safetensors）与
motion parquet（/repo/runs/motion/<档位目录>/whole_body/soma/sequence_id=<序列>/robot_name=g1/data.parquet）。帧参考目录里的 manifest_*.tsv
（make_frame_refs_t3.py 写的）每集一行：episode run 步数 档位目录 序列 帧数 poses.npy 的 sha256 来源 parquet 的 sha256 文件 sha256。逐集要求：
  - 结果目录里这一集的 parquet 与 JSON 成对（按两种文件的并集逐集走，Fable 复核第 2 条：打包器读的是 parquet）；
  - 所有 manifest 合起来，这一集恰好一行，文件名是 episode_XXXXXX.npz；
  - run、步数、档位目录、序列都与 JSON 里的相同（帧参考来自打包的这条 run 训练用的那份输入）；
  - episode_XXXXXX.npz 在帧参考目录里，sha256 等于 manifest 记的。
帧参考目录里有、结果目录里没有的集只报不拒（打包器只读结果目录里有的集）。
用法：python3 stages/evaluate/frefs_t3_match.py <结果目录> <帧参考目录>；逐集只打不过的，最后一行 FREFS_T3_MATCH_OK n 或 FREFS_T3_MATCH_FAIL n，
退出码 0 / 1。测试：stages/evaluate/test_frefs_t3_match.py。
"""
import hashlib
import json
import re
import sys
from pathlib import Path

CKPT = re.compile(r"/repo/runs/flash_chord/([^/]+)/policy_(\d+)\.safetensors")
MOTION = re.compile(r"/repo/runs/motion/(tier3(?:_[a-z0-9]+)?)/whole_body/soma/sequence_id=([^/]+)/robot_name=g1/data\.parquet")
COLS = ["episode", "run", "step", "tier_dir", "sequence", "frames", "poses_sha256", "source_parquet_sha256", "file", "sha256"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_rows(fref: Path) -> dict:
    rows = {}
    for m in sorted(fref.glob("manifest_*.tsv")):
        lines = m.read_text(encoding="utf-8").splitlines()
        if not lines or lines[0].split("\t") != COLS:
            raise ValueError(f"{m.name} 的表头不是 {COLS}")
        for line in lines[1:]:
            f = line.split("\t")
            if len(f) != len(COLS):
                raise ValueError(f"{m.name} 有一行不是 {len(COLS)} 列：{line!r}")
            r = dict(zip(COLS, f))
            rows.setdefault(int(r["episode"]), []).append((m.name, r))
    return rows


def episodes(res: Path, ext: str) -> dict:
    return {int(p.stem[len("episode_"):]): p for p in res.glob(f"episode_[0-9][0-9][0-9][0-9][0-9][0-9].{ext}")}


def check(res: Path, fref: Path) -> tuple:
    rows = manifest_rows(fref)
    js, pqs = episodes(res, "json"), episodes(res, "parquet")
    eps = sorted(set(js) | set(pqs))   # 按两种文件的并集走：打包器读的是 parquet，没有 JSON 的 parquet 就对不上是哪条 run（Fable 复核第 2 条）
    if not eps:
        return [f"  FAIL 结果目录 {res} 里没有 episode_XXXXXX.json 与 .parquet"], 0, 1
    msgs, ok, bad = [], 0, 0
    for ep in eps:
        j = js.get(ep)
        try:
            if j is None or ep not in pqs:
                raise ValueError("结果目录里这一集只有 " + ("parquet、没有 metrics JSON" if j is None else "metrics JSON、没有 parquet"))
            x = json.loads(j.read_text())
            c, mo = CKPT.fullmatch(str(x.get("checkpoint", ""))), MOTION.fullmatch(str(x.get("parquet", "")))
            if not (c and mo):
                raise ValueError(f"JSON 里的 checkpoint 或 parquet 认不出：{x.get('checkpoint')}、{x.get('parquet')}")
            hit = rows.get(ep, [])
            if len(hit) != 1:
                raise ValueError(f"manifest 里这一集有 {len(hit)} 行（{', '.join(n for n, _ in hit) or '无'}），要恰好 1 行")
            name, r = hit[0]
            want = {"run": c.group(1), "step": c.group(2), "tier_dir": mo.group(1), "sequence": mo.group(2), "file": f"episode_{ep:06d}.npz"}
            diff = [f"{k} {r[k]}（导出 {v}）" for k, v in want.items() if r[k] != v]
            if diff:
                raise ValueError(f"{name} 与导出不符：{'；'.join(diff)}")
            npz = fref / r["file"]
            if not npz.is_file():
                raise ValueError(f"帧参考目录里没有 {r['file']}")
            got = sha256(npz)
            if got != r["sha256"]:
                raise ValueError(f"{r['file']} 的 sha256 {got[:16]}… 与 {name} 记的 {r['sha256'][:16]}… 不同")
            ok += 1
        except (OSError, ValueError) as e:
            msgs.append(f"  FAIL ep{ep}：{type(e).__name__}: {e}")
            bad += 1
    extra = sorted(set(rows) - set(eps))
    if extra:
        msgs.append(f"  --   帧参考里有、结果目录里没有的集（打包器不读）：{extra}")
    return msgs, ok, bad


def main() -> int:
    if len(sys.argv) != 3:
        sys.exit("用法：frefs_t3_match.py <结果目录> <帧参考目录>")
    try:
        msgs, ok, bad = check(Path(sys.argv[1]), Path(sys.argv[2]))
    except (OSError, ValueError) as e:
        print(f"  FAIL 读不了帧参考的 manifest：{type(e).__name__}: {e}")
        print("FREFS_T3_MATCH_FAIL 1")
        return 1
    for m in msgs:
        print(m)
    print(f"FREFS_T3_MATCH_OK {ok}" if bad == 0 else f"FREFS_T3_MATCH_FAIL {bad}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
