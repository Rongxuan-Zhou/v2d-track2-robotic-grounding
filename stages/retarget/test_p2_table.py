"""p2_table.py 的测试（守计划 agent-doc/plan-t3-v2-retarget18-2026-10-10.md 第 4 节的 P2′ 判据；暂无契约 ID）。只用标准库。
  1. 间隙恰好 ±30.0 mm 且有接触：过；ALIGN 状态但 |间隙| ≤ 30 mm：过
  2. |间隙| 30.1 mm、两手都没有接触：不过，并写明原因；FAIL:retarget：失败；记录头要求了、却没有结果行的集：失败
  3. --no-contact：两手都没有接触的那集只按间隙判，单独打 CONTACT_NA 行
  4. 脚底高度范围按日志段的集号对上（日志段标题“最后 15 行”与“脚底高度与最后 15 行”都认）；
     --a-phase 给出 A 段面高、间隙与差，A 段在支撑面上而重定向后是地面的集注明并打 SURFACE_TO_FLOOR 行
  5. 真记录 results/t3p2/retarget_mf4090.txt（若在）：ep8 不过（-33.9 mm）、ep24 过，与 10/10 的人工判读相同
  6. 没有 EPS_DONE、没有记录头、有记录头没要求的集：报错
用法：python3 stages/retarget/test_p2_table.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def rec(eps: list, rows: list, sole: dict, done: bool = True, head: str = "最后 15 行") -> str:
    out = [f"# command  : stages/retarget/eps.sh 3 {' '.join(map(str, eps))} ",
           "来历 abc，2026-10-11 04:04:12 CST；目录 runs/motion_src/tier3_v2、runs/motion/tier3_v2",
           "one.sh 3 0 退出码 0（04:05:32）", "## 结果行（档位 ep 序列 帧数 fps 左手接触 右手接触 IK中位 间隙mm 支撑面 状态）"]
    out += ["\t".join(map(str, r)) for r in rows]
    for ep, (lo, hi) in sole.items():
        out += [f"## runs/logs/batch/t3_v2_ep{ep:02d}.log（{head}）", "[INFO] Same-frame robot-only anti-penetration lift: [+0.0000, +0.0000] m",
                f"[INFO] Saved robot sole Z range: [{lo:+.4f}, {hi:+.4f}] m", "RESULT gap_mm=0 surface=floor ok=1"]
    return "\n".join(out + (["EPS_DONE"] if done else [])) + "\n"


ROWS = [
    [3, 0, "s0", 600, 30.0, 10, 0, 0.5, -30.0, "floor", "ALIGN"],
    [3, 3, "s3", 600, 30.0, 0, 12, 0.5, 30.0, "support", "ALIGN"],
    [3, 5, "s5", 600, 30.0, 4, 4, 0.5, -12.5, "floor", "ALIGN"],
    [3, 7, "s7", 600, 30.0, 4, 4, 0.5, -30.1, "support", "ALIGN"],
    [3, 9, "s9", 600, 30.0, 0, 0, 0.5, 2.0, "floor", "OK+NOCONTACT"],
    [3, 12, "s12", "", "", "", "", "", "", "", "FAIL:retarget"],
    [3, 16, "s16", 600, 30.0, 3, 0, 0.5, 780.4, "floor", "ALIGN"],
    [3, 29, "s29", 467, 30.0, 0, 0, 0.7, -4.1, "floor", "OK+NOCONTACT"],
]
EPS = [0, 3, 5, 7, 9, 12, 14, 16, 29]          # ep14 没有结果行
A_PHASE = "\n".join(["ep\t物体\t帧\t位移mm\tH_m\th_mm\t面高_m\t间隙mm\t间隙30mm\tZ_m\tX_m\tD_m\tL_m\tk\tparquet_sha256",
                     "5\tbox\t5\t14.2\t1.383\t+4.2\t0.000\t+4.2\t-6.8\t2.53\t-0.34\t2.71\t0.905\t0.995\t5c508937b139",
                     "16\tspray\t5\t22.5\t1.385\t+858.8\t0.878\t-19.1\t-14.1\t2.77\t-0.14\t2.80\t0.397\t1.051\tc1978473f854",
                     "T3_FLOAT_FACTS_DONE"]) + "\n"


def run(text: str, *extra, a_text: str = None, path: Path = None):
    with tempfile.TemporaryDirectory() as t:
        r = Path(t) / "r.txt"
        r.write_bytes(b"# command  : bash -c \xe6\x94\\225\n" + text.encode("utf-8"))   # 同真记录：头一行有坏的 UTF-8
        args = [sys.executable, str(HERE / "p2_table.py"), str(path or r), *extra]
        if a_text is not None:
            (Path(t) / "a.txt").write_text(a_text, encoding="utf-8")
            args += ["--a-phase", str(Path(t) / "a.txt")]
        return subprocess.run(args, capture_output=True, text=True)


def lines_by_ep(out: str) -> dict:
    return {int(x.split("\t")[0]): x.split("\t") for x in out.splitlines() if x.split("\t")[0].isdigit()}


def test_verdicts():
    p = run(rec(EPS, ROWS, {0: (0.0004, 0.176), 5: (0.001, 0.11)}), a_text=A_PHASE)
    assert p.returncode == 0, p.stderr
    got = lines_by_ep(p.stdout)
    assert [got[e][13] for e in EPS] == ["过", "过", "过", "不过", "不过", "失败", "失败", "不过", "不过"], p.stdout
    assert "穿地" in got[7][14] and "两手都没有接触" in got[9][14] and got[12][14] == "FAIL:retarget", p.stdout
    assert got[14][14] == "没有结果行" and "悬空" in got[16][14], p.stdout
    assert got[0][9] == "+0.0004～+0.1760" and got[3][9] == "-", p.stdout
    assert got[5][10:13] == ["0.000", "+4.2", "-16.7"] and got[0][10] == "-", p.stdout
    assert "A 段静置在面高 0.878 m 的支撑面上，重定向后最近的是地面" in got[16][14] and "A 段" not in got[5][14], p.stdout
    tail = p.stdout.splitlines()[-4:]
    assert tail == ["P2_PASS 3/9：ep0 ep3 ep5", "P2_FAIL 4/9：ep7 ep9 ep16 ep29", "RETARGET_FAIL 2/9：ep12 ep14",
                    "SURFACE_TO_FLOOR 1：ep16（A 段在支撑面上、重定向后最近的是地面；只报）"], tail


def test_no_contact():
    p = run(rec(EPS, ROWS, {}, head="脚底高度与最后 15 行"), "--no-contact", "29")
    assert p.returncode == 0, p.stderr
    got = lines_by_ep(p.stdout)
    assert got[29][13] == "过" and got[29][14] == "接触一条不适用，只按间隙判", p.stdout
    assert got[9][13] == "不过", p.stdout          # 没登记的那集照旧
    assert "P2_PASS 4/9：ep0 ep3 ep5 ep29" in p.stdout and p.stdout.splitlines()[-1] == "CONTACT_NA ep29=过", p.stdout


def test_new_log_head():
    p = run(rec([0], ROWS[:1], {0: (0.0004, 0.176)}, head="脚底高度与最后 15 行"))
    assert p.returncode == 0, p.stderr
    assert lines_by_ep(p.stdout)[0][9] == "+0.0004～+0.1760", p.stdout


def test_real_round2_record():
    real = REPO / "results" / "t3p2" / "retarget_mf4090.txt"
    if not real.is_file():
        print("  （跳过：没有 results/t3p2/retarget_mf4090.txt）")
        return
    p = run("", path=real)
    assert p.returncode == 0, p.stderr
    got = lines_by_ep(p.stdout)
    assert got[8][13] == "不过" and got[8][6] == "-33.9" and got[24][13] == "过", p.stdout
    assert got[8][9] == "+0.0005～+0.1106" and got[24][9] == "+0.0004～+0.1760", p.stdout


def test_bad_records_refuse():
    p = run(rec(EPS, ROWS, {}, done=False))
    assert p.returncode != 0 and "EPS_DONE" in p.stderr, p.stdout + p.stderr
    p = run(rec(EPS, ROWS, {}).replace("# command  : stages/retarget/eps.sh", "# command  : stages/retarget/other.sh"))
    assert p.returncode != 0 and "记录头有 0 行" in p.stderr, p.stdout + p.stderr
    p = run(rec([0, 3], ROWS, {}))
    assert p.returncode != 0 and "记录头没要求的集" in p.stderr, p.stdout + p.stderr


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("ALL_OK")
