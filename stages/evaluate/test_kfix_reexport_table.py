"""kfix_reexport_table.py 的测试（暂无契约 ID：运行规范待成文；计划 agent-doc/plan-t3-kfix-2026-10-16.md 判据 5）。只用标准库，在 Mac 上跑。
临时目录里造 runs/flash_chord/<run>/ 的 10/11 旧评测、plain 与 idchk 的 4096 世界 JSON（idchk、kfix 带包装层旁证 <JSON>.override.json）、
新参考的 4096 世界 JSON 与 1 世界导出：
  1. 包装层恒等、评测器一致（idchk 比 plain 低 0.013，同 9/28 实测的重评噪声，容差内）：--check 打 KFIX_CHECK 恒等 一致；ep0、ep29 换并写表
     （sha256 对），ep18 不换，ep24 导出不全不写，ep26 参考不对作废；
  2. 包装层不恒等（idchk 与 plain 的 MP-SR 差 0.05）：--check 打 KFIX_CHECK 不恒等 一致，结果表只有表头；
  3. 评测器不一致（plain 与 10/11 差 0.05）：--check 打 KFIX_CHECK 恒等 不一致；没有 plain 的 kfix 行不判，ep0 有 plain 时旧值取 plain
     （plain 的 MP-SR 0.999，门槛 0.949，新值 0.95 过线 → 只有 ep0 写表）；
  4. 没有 idchk 的 JSON：结果表只有表头；
  5. plain 的 JSON 记的参考不是清单给的那份：--check 打 KFIX_CHECK 不恒等 缺，结果表只有表头；
  6. 10/11 的 JSON 记的参考不是清单给的那份：--check 打 KFIX_CHECK 恒等 不一致（不可比，按不一致处理）；
  7. ep0 的包装层旁证 step_count_ok 为假（包装层判作废、JSON 留着）：ep0 不判，结果表只有 ep29；
  8. idchk 没有包装层旁证：--check 打 KFIX_CHECK 不恒等 缺，结果表只有表头。
用法：python3 stages/evaluate/test_kfix_reexport_table.py；最后一行 ALL_OK。
"""
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

TOOL = Path(__file__).resolve().parent / "kfix_reexport_table.py"
STEP = "349999104"
OLD = {0: (0.965, 0.992), 18: (0.862, 0.938), 24: (0.933, 0.995), 26: (0.999, 0.999), 29: (0.0, 0.713)}
NEW = {0: (0.95, 0.99), 18: (0.80, 0.93), 24: (0.93, 0.99), 26: (0.99, 0.99), 29: (0.0, 0.673)}
OLDPQ = "/repo/runs/motion/tier3_v2/seq26/robot_name=g1/data.parquet"


def wj(p: Path, mp: float, sp: float, parquet=None, side=None) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    d = {"metrics": {"maniptrans_sr": mp, "spider_sr_uncentered": sp, "add_auc": 0.5, "mppe_cm": 3.0, "world_count": 4096}}
    if parquet:
        d["parquet"] = parquet
    p.write_text(json.dumps(d))
    if side is not None:
        Path(str(p) + ".override.json").write_text(json.dumps(side))


def good_side(pq: str) -> dict:
    return {"calls": 1, "step_count_ok": True, "parquet": pq}


def j4096(root: Path, run: str, suf: str, e: int) -> Path:
    return root / "runs/flash_chord" / run / "export_8c778f9" / f"s{STEP}_w4096{suf}" / f"episode_{e:06d}.json"


def build(root: Path, idchk_mp: float = 0.986, plain_mp: float = 0.999, with_idchk: bool = True, plain_ep0=None,
          plain_pq=None, old26_pq=None, idchk_side: bool = True, ep0_side=None) -> Path:
    rows = ["run\tstep\tparquet\tsuffix\tmode"]
    run26 = "t3-ep26-w2048-cap4m-r3a-v2-mf4090"
    rows.append(f"{run26}\t{STEP}\t-\t_plain\tplain")
    rows.append(f"{run26}\t{STEP}\t{OLDPQ}\t_idchk\tidchk")
    wj(j4096(root, run26, "_plain", 26), plain_mp, 0.999, parquet=plain_pq or OLDPQ)
    if with_idchk:
        wj(j4096(root, run26, "_idchk", 26), idchk_mp, 0.999, parquet=OLDPQ, side=good_side(OLDPQ) if idchk_side else None)
    for e, (mp, sp) in OLD.items():
        run = f"t3-ep{e}-w2048-cap4m-r3a-v2-mf4090"
        base = root / "runs/flash_chord" / run
        wj(base / f"eval_{STEP}_w4096.json", mp, sp, parquet=(old26_pq or OLDPQ) if e == 26 else f"/repo/old{e}.parquet")
        pq = f"/repo/runs/motion/tier3_kfix/seq{e}/robot_name=g1/data.parquet"
        nm, ns = NEW[e]
        jpq = "/repo/runs/motion/tier3_v2/wrong/data.parquet" if e == 26 else pq
        wj(j4096(root, run, "_kfixgb", e), nm, ns, parquet=jpq, side=(ep0_side if e == 0 and ep0_side else good_side(jpq)))
        exp = base / "export_8c778f9" / f"s{STEP}_w1_kfixgb"
        exp.mkdir(parents=True, exist_ok=True)
        (exp / f"episode_{e:06d}.json").write_text(f"json{e}")
        if e != 24:
            (exp / f"episode_{e:06d}.parquet").write_bytes(f"pq{e}".encode())
        rows.append(f"{run}\t{STEP}\t{pq}\t_kfixgb\tkfix")
    if plain_ep0 is not None:
        wj(j4096(root, "t3-ep0-w2048-cap4m-r3a-v2-mf4090", "_plain", 0), *plain_ep0)
    lst = root / "list.tsv"
    lst.write_text("\n".join(rows) + "\n")
    return lst


def run(root: Path, lst: Path, check: bool = False) -> tuple:
    args = [sys.executable, str(TOOL), str(lst)] + (["--check"] if check else ["--write-table", "runs/exports/kfix.tsv"])
    r = subprocess.run(args, cwd=root, capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stdout + r.stderr
    t = root / "runs/exports/kfix.tsv"
    return r, (t.read_text().splitlines() if t.exists() else [])


def last(r) -> str:
    return r.stdout.strip().splitlines()[-1]


def eps(table: list) -> list:
    return sorted(int(ln.split("\t")[1]) for ln in table[1:])


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        lst = build(root)
        r, _ = run(root, lst, check=True)
        assert last(r) == "KFIX_CHECK 恒等 一致", r.stdout
        r, table = run(root, lst)
        assert eps(table) == [0, 29], (eps(table), r.stdout)
        row0 = next(ln.split("\t") for ln in table[1:] if ln.split("\t")[1] == "0")
        assert row0[7] == hashlib.sha256(b"json0").hexdigest() and row0[8] == hashlib.sha256(b"pq0").hexdigest()
        assert "参考不是清单给的那份" in r.stdout and "1 世界导出不全" in r.stdout and last(r) == "KFIX_TABLE_DONE 2"
    print("  1. 包装层恒等（差 0.013 在容差内）、评测器一致：ep0、ep29 换并写表，ep18 不换，ep24 导出不全不写，ep26 参考不对作废")
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        lst = build(root, idchk_mp=0.949)
        r, _ = run(root, lst, check=True)
        assert last(r) == "KFIX_CHECK 不恒等 一致", r.stdout
        r, table = run(root, lst)
        assert len(table) == 1 and "结果表不写任何行" in r.stdout, r.stdout
    print("  2. 包装层不恒等（差 0.05）：--check 报不恒等，结果表只有表头")
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        lst = build(root, idchk_mp=0.949, plain_mp=0.949, plain_ep0=(0.999, 0.999))
        r, _ = run(root, lst, check=True)
        assert last(r) == "KFIX_CHECK 恒等 不一致", r.stdout
        r, table = run(root, lst)
        assert "没有 plain 评测，不判" in r.stdout and "旧值取 同一评测器的 plain" in r.stdout, r.stdout
        assert "MP-SR 0.999→0.950" in r.stdout and "MP-SR ≥ 0.949" in r.stdout, r.stdout
        assert eps(table) == [0], table
    print("  3. 评测器不一致（差 0.05）：没有 plain 的 kfix 行不判；ep0 旧值取 plain（0.999，门槛 0.949），新 0.95 过线 → 只有 ep0 写表")
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        lst = build(root, with_idchk=False)
        r, table = run(root, lst)
        assert len(table) == 1, (table, r.stdout)
    print("  4. 没有 idchk 的 JSON：结果表只有表头")
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        lst = build(root, plain_pq="/repo/runs/motion/tier3_v1/seq26/robot_name=g1/data.parquet")
        r, _ = run(root, lst, check=True)
        assert last(r) == "KFIX_CHECK 不恒等 缺" and "要都是清单给的" in r.stdout, r.stdout
        r, table = run(root, lst)
        assert len(table) == 1, (table, r.stdout)
    print("  5. plain 评的参考不是清单给的那份：不恒等，结果表只有表头")
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        lst = build(root, old26_pq="/repo/runs/motion/tier3_v1/seq26/robot_name=g1/data.parquet")
        r, _ = run(root, lst, check=True)
        assert last(r) == "KFIX_CHECK 恒等 不一致" and "不可比" in r.stdout, r.stdout
    print("  6. 10/11 的评测记的参考不同：按评测器不一致处理")
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        pq0 = "/repo/runs/motion/tier3_kfix/seq0/robot_name=g1/data.parquet"
        lst = build(root, ep0_side={"calls": 1, "step_count_ok": False, "parquet": pq0})
        r, table = run(root, lst)
        assert eps(table) == [29] and "包装层旁证不合（step_count_ok），不判" in r.stdout, (eps(table), r.stdout)
    print("  7. ep0 的包装层旁证判作废：ep0 不判，结果表只有 ep29")
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        lst = build(root, idchk_side=False)
        r, _ = run(root, lst, check=True)
        assert last(r) == "KFIX_CHECK 不恒等 缺" and "没有包装层旁证" in r.stdout, r.stdout
        r, table = run(root, lst)
        assert len(table) == 1, (table, r.stdout)
    print("  8. idchk 没有包装层旁证：不恒等，结果表只有表头")
    print("ALL_OK")
