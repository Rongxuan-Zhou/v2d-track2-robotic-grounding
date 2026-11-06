"""按选择清单从几个已拉回的导出目录拼一个 Tier 3 结果目录（Mac 上、仓库根执行，只用标准库）。计划 agent-doc/plan-t3pack-2026-10-29.md 第 1 节第 3 条。

为什么：换入几集的新清单，其余集沿用已交清单的导出（runs/submission/<名>/t3），新集由 pull_submission.sh select 拉到另一个目录；
pack_tier.sh 只收一个结果目录，要把两边按集拼起来。
做法：
  - 选择清单每行 tab 分隔：档位 episode run 步数 机器（# 开头为注释），档位必须是 3，episode 不重复；
  - 每一行在各源目录里看 episode_XXXXXX.json：checkpoint 的文件名必须是 policy_<步数>.safetensors、上一级目录名必须是 run，
    environment_steps 等于步数；满足的源必须恰好一个，对应的 parquet 必须存在；
  - 复制 JSON 与 parquet 到目标目录，复制后核 sha256 与源相同；在目标目录旁边写 <目标目录>.ASSEMBLE.json（清单的 sha256、每集取自哪个源、
    两份文件的 sha256），不放进目标目录，免得打包器看到多余的文件。
目标目录已存在就拒绝。不读任何档位的数据集（进程开头 forbid_tier1()）。
用法：python3 stages/evaluate/assemble_t3_results.py <选择清单> <目标目录> <源目录> [<源目录> …]   最后一行 ASSEMBLE_T3_OK n 或报错退出
测试：stages/evaluate/test_assemble_t3_results.py。
"""
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_list(path: Path) -> list:
    rows, seen = [], set()
    for ln in path.read_text().splitlines():
        if not ln.strip() or ln.startswith("#"):
            continue
        f = ln.split("\t")
        if len(f) < 5:
            raise SystemExit(f"清单行不足 5 列：{ln!r}")
        tier, ep, run, step = f[0], int(f[1]), f[2], int(f[3])
        if tier != "3":
            raise SystemExit(f"档位不是 3：{ln!r}")
        if ep in seen:
            raise SystemExit(f"episode {ep} 重复")
        seen.add(ep)
        rows.append((ep, run, step))
    return rows


def matches(js: Path, run: str, step: int) -> bool:
    d = json.loads(js.read_text())
    ck = Path(d["checkpoint"])
    return ck.name == f"policy_{step}.safetensors" and ck.parent.name == run and int(d["environment_steps"]) == step


def main(argv) -> int:
    sys.path.insert(0, str(ROOT / "libs/v2d_infra"))
    from tier_guard import forbid_tier1
    forbid_tier1()   # 档位隔离：此后读 Tier 1 的任何文件都会抛错
    if len(argv) < 4:
        raise SystemExit(__doc__.strip().splitlines()[-2])
    sel, out, srcs = Path(argv[1]), Path(argv[2]), [Path(x) for x in argv[3:]]
    rec_path = out.parent / f"{out.name}.ASSEMBLE.json"
    if out.exists() or rec_path.exists():
        raise SystemExit(f"目标目录或记录已存在：{out}")
    rows = read_list(sel)
    plan = []
    for ep, run, step in rows:
        name = f"episode_{ep:06d}"
        hits = [s for s in srcs if (s / f"{name}.json").is_file() and matches(s / f"{name}.json", run, step)]
        if len(hits) != 1:
            raise SystemExit(f"ep{ep} {run}@{step}：符合的源目录有 {len(hits)} 个（要恰好 1 个）：{[str(h) for h in hits]}")
        if not (hits[0] / f"{name}.parquet").is_file():
            raise SystemExit(f"ep{ep}：{hits[0]} 里缺 {name}.parquet")
        plan.append((ep, run, step, hits[0], name))
    out.mkdir(parents=True)
    rec = {"select": str(sel), "select_sha256": sha256(sel), "episodes": {}}
    for ep, run, step, src, name in plan:
        shas = {}
        for ext in ("json", "parquet"):
            a, b = src / f"{name}.{ext}", out / f"{name}.{ext}"
            shutil.copy2(a, b)
            sa, sb = sha256(a), sha256(b)
            if sa != sb:
                raise SystemExit(f"ep{ep} {ext} 复制后 sha256 不同")
            shas[ext] = sb
        rec["episodes"][str(ep)] = {"run": run, "step": step, "source": str(src), "sha256": shas}
        print(f"{ep}\t{run}\t{step}\t{src}")
    rec_path.write_text(json.dumps(rec, ensure_ascii=False, indent=1))
    print(f"ASSEMBLE_T3_OK {len(plan)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
