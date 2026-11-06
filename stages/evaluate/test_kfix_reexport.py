"""kfix_reexport.sh 的行为测试（暂无契约 ID：运行规范待成文；计划 agent-doc/plan-t3-kfix-2026-10-16.md “评测器版本与两道核对”）。
只用标准库，Mac 与训练机都能跑。临时目录里搭最小仓库：真的 kfix_reexport.sh 与 kfix_reexport_table.py；假的 stages/evaluate/fc_export.sh
（记下每次调用；预演按 FAKE_PRE_FAIL 失败；正式调用按 FAKE_FAIL 失败，否则按 spec.json 写评测 JSON、包装层旁证与 1 世界导出）。
  1. 正常：预演全过，plain、idchk、各 kfix 行（4096 世界、1 世界）依次跑；评测器一致，不补 plain；结果表有 ep0、ep24、ep26、ep29（ep18 不换）；最后一行 KFIX_REEXPORT_DONE 0；
  2. 预演不过（ep18 的 1 世界导出）：KFIX_REEXPORT_STOPPED preflight，没有正式调用；
  3. idchk 评测失败：KFIX_REEXPORT_STOPPED idchk，kfix 行一行没跑；
  4. 评测器不一致（plain 与 10/11 差 0.05）：每条 kfix 行先补 plain；ep18 补的 plain 失败就跳过 ep18（它的 4096 世界评测不跑）；
  5. 汇总脚本出错（ep29 的新评测 JSON 缺 mppe_cm）：KFIX_REEXPORT_STOPPED table，结果表不留。
用法：python3 stages/evaluate/test_kfix_reexport.py；最后一行 ALL_OK。
"""
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEP = "349999104"
EPS = (0, 18, 24, 26, 29)
OLD = {0: (0.965, 0.992), 18: (0.862, 0.938), 24: (0.933, 0.995), 26: (0.999, 0.999), 29: (0.0, 0.713)}
NEW = {0: (0.95, 0.99), 18: (0.80, 0.93), 24: (0.93, 0.99), 26: (0.99, 0.99), 29: (0.0, 0.673)}

FAKE = r'''#!/usr/bin/env bash
run=$1; step=$2; suf=${V2D_EXPORT_SUFFIX:-}; w=${V2D_EXPORT_WORLDS:-1}
echo "${V2D_EXPORT_DRYRUN:-0} $run $suf $w" >> "$FAKE_CALLS"
if [ "${V2D_EXPORT_DRYRUN:-0}" = 1 ]; then
  [ "$run:$suf:$w" = "${FAKE_PRE_FAIL:-}" ] && exit 2
  echo "fc_export: DRYRUN"; exit 0
fi
[ "$run:$suf:$w" = "${FAKE_FAIL:-}" ] && exit 3
python3 - "$run" "$step" "$suf" "$w" <<'PY'
import json, os, re, sys
from pathlib import Path
run, step, suf, w = sys.argv[1:]
spec = json.loads(Path("spec.json").read_text())
ep = int(re.match(r"t3-ep(\d+)-", run).group(1))
d = Path("runs/flash_chord") / run / "export_8c778f9" / f"s{step}_w{w}{suf}"
d.mkdir(parents=True, exist_ok=True)
j = d / f"episode_{ep:06d}.json"
pq = os.environ.get("V2D_EXPORT_PARQUET") or spec["oldpq"][str(ep)]
mp, sp = spec["metrics"][f"{ep}{suf}"]
m = {"maniptrans_sr": mp, "spider_sr_uncentered": sp, "add_auc": 0.5, "mppe_cm": 3.0, "world_count": int(w)}
if spec.get("drop_mppe") == f"{ep}{suf}":
    del m["mppe_cm"]
j.write_text(json.dumps({"metrics": m, "parquet": pq}))
if os.environ.get("V2D_EXPORT_PARQUET"):
    Path(str(j) + ".override.json").write_text(json.dumps({"calls": 1, "step_count_ok": True, "parquet": pq}))
if w == "1":
    (d / f"episode_{ep:06d}.parquet").write_bytes(b"PAR1")
PY
'''


def setup(root: Path, plain26: float = 0.999, drop_mppe: str = "") -> None:
    (root / "stages/evaluate/exports").mkdir(parents=True)
    for f in ("kfix_reexport.sh", "kfix_reexport_table.py"):
        shutil.copy(HERE / f, root / "stages/evaluate" / f)
    (root / "stages/evaluate/fc_export.sh").write_text(FAKE)
    (root / "stages/evaluate/fc_export.sh").chmod(0o755)
    rows = ["run\tstep\tparquet\tsuffix\tmode"]
    run26 = "t3-ep26-w2048-cap4m-r3a-v2-mf4090"
    oldpq = {str(e): f"/repo/runs/motion/tier3_v2/seq{e}/robot_name=g1/data.parquet" for e in EPS}
    rows += [f"{run26}\t{STEP}\t-\t_plain\tplain", f"{run26}\t{STEP}\t{oldpq['26']}\t_idchk\tidchk"]
    metrics = {"26_plain": (plain26, 0.999), "26_idchk": (plain26, 0.999)}
    for e in EPS:
        run = f"t3-ep{e}-w2048-cap4m-r3a-v2-mf4090"
        base = root / "runs/flash_chord" / run
        base.mkdir(parents=True)
        mp, sp = OLD[e]
        (base / f"eval_{STEP}_w4096.json").write_text(json.dumps(
            {"metrics": {"maniptrans_sr": mp, "spider_sr_uncentered": sp, "add_auc": 0.5, "mppe_cm": 3.0}, "parquet": oldpq[str(e)]}))
        pq = f"/repo/runs/motion/tier3_kfix/seq{e}/robot_name=g1/data.parquet"
        (root / pq[len("/repo/"):]).parent.mkdir(parents=True)
        (root / pq[len("/repo/"):]).write_bytes(b"ref")
        rows.append(f"{run}\t{STEP}\t{pq}\t_kfixgb\tkfix")
        metrics[f"{e}_kfixgb"] = NEW[e]
        metrics.setdefault(f"{e}_plain", OLD[e])
    (root / oldpq["26"][len("/repo/"):]).parent.mkdir(parents=True)
    (root / oldpq["26"][len("/repo/"):]).write_bytes(b"old26")
    (root / "stages/evaluate/exports/kfix_t3_mf4090.tsv").write_text("\n".join(rows) + "\n")
    (root / "spec.json").write_text(json.dumps({"oldpq": oldpq, "metrics": metrics, "drop_mppe": drop_mppe}))


def run(root: Path, **env) -> tuple:
    calls = root / "calls.txt"
    e = dict(os.environ, FAKE_CALLS=str(calls), V2D_FC_SRC="/fake")
    e.update(env)
    r = subprocess.run(["bash", str(root / "stages/evaluate/kfix_reexport.sh")], check=False, cwd=root, env=e, capture_output=True, text=True,
                       errors="replace", timeout=120)
    real = [ln.split()[1:] for ln in (calls.read_text().splitlines() if calls.exists() else []) if ln.startswith("0 ")]
    return r.returncode, r.stdout, real   # 只看 stdout：汇总脚本的 Traceback 走 stderr，接在后面会顶掉最后一行


def last(text: str) -> str:
    return text.strip().splitlines()[-1]


def table(root: Path) -> list:
    t = root / "runs/exports/kfix_t3_mf4090.tsv"
    return sorted(int(ln.split("\t")[1]) for ln in t.read_text().splitlines()[1:]) if t.exists() else None


R = "t3-ep{}-w2048-cap4m-r3a-v2-mf4090"

if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root)
        rc, text, real = run(root)
        assert rc == 0 and last(text) == "KFIX_REEXPORT_DONE 0" and "预演全过" in text and "sha256" in text, text
        want = [[R.format(26), "_plain", "4096"], [R.format(26), "_idchk", "4096"]]
        for e in EPS:
            want += [[R.format(e), "_kfixgb", "4096"], [R.format(e), "_kfixgb", "1"]]
        assert real == want, real
        assert table(root) == [0, 24, 26, 29], (table(root), text)
    print("  1. 正常：两道核对过，kfix 行依次评测与导出，结果表 ep0、24、26、29，KFIX_REEXPORT_DONE 0")
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root)
        rc, text, real = run(root, FAKE_PRE_FAIL=f"{R.format(18)}:_kfixgb:1")
        assert rc == 1 and last(text) == "KFIX_REEXPORT_STOPPED preflight" and not real and "1 世界导出" in text, (rc, text, real)
    print("  2. 预演不过：KFIX_REEXPORT_STOPPED preflight，没有正式调用")
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root)
        rc, text, real = run(root, FAKE_FAIL=f"{R.format(26)}:_idchk:4096")
        assert rc == 1 and last(text) == "KFIX_REEXPORT_STOPPED idchk" and len(real) == 2, (rc, text, real)
    print("  3. idchk 评测失败：KFIX_REEXPORT_STOPPED idchk，kfix 行没跑")
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root, plain26=0.949)
        rc, text, real = run(root, FAKE_FAIL=f"{R.format(18)}:_plain:4096")
        assert rc == 0 and "评测器不一致" in text and last(text) == "KFIX_REEXPORT_DONE 1", text
        r18 = [c for c in real if c[0] == R.format(18)]
        assert r18 == [[R.format(18), "_plain", "4096"]], r18
        r0 = [c for c in real if c[0] == R.format(0)]
        assert r0 == [[R.format(0), "_plain", "4096"], [R.format(0), "_kfixgb", "4096"], [R.format(0), "_kfixgb", "1"]], r0
    print("  4. 评测器不一致：每条 kfix 行先补 plain；ep18 补的 plain 失败就跳过")
    with tempfile.TemporaryDirectory() as t:
        root = Path(t); setup(root, drop_mppe="29_kfixgb")
        rc, text, real = run(root)
        assert rc == 1 and last(text) == "KFIX_REEXPORT_STOPPED table" and table(root) is None, (rc, text)
    print("  5. 汇总脚本出错：KFIX_REEXPORT_STOPPED table，结果表不留")
    print("ALL_OK")
