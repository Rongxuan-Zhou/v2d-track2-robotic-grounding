"""t3_pick_version.py 的测试（守计划 agent-doc/plan-t3-pilot-v2-2026-10-10.md 第 4 节登记的选版规则；运行规范未成文，暂无契约 ID）。只用标准库。
假记录照 t3_float_facts.py --all-rest --offset-gate 30 的输出格式写（每集一行 15 列、OFFSET_GATE 行、结束标记）。核：
  1. 两版都不过 → PICK_NONE；只一版过 → 选它
  2. 两版都过、|o| 之差大于 2·√(SE²+SE²) → 取 |o| 小的；差在这以内 → 取 ep8、ep24 间隙绝对值较大者更小的那版
  3. 选出的版 ep8 或 ep24 的间隙绝对值超过 50 mm → PICK_BLOCKED
  4. 记录缺结束标记、缺 OFFSET_GATE 行或缺 ep8 → 报错退出
用法：python3 stages/recovery/test_t3_pick_version.py
"""
import contextlib
import io
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t3_pick_version as P  # noqa: E402

HEAD = "ep\t物体\t帧\t位移mm\tH_m\th_mm\t面高_m\t间隙mm\t间隙30mm\tZ_m\tX_m\tD_m\tL_m\tk\tparquet_sha256"


def record(verdict, o, se, sd, gap8, gap24, done=True, gate=True, ep8=True):
    rows = [HEAD]
    for ep, gap in ((0, 5.0), (8, gap8), (24, gap24)):
        if ep == 8 and not ep8:
            continue
        rows.append("\t".join([str(ep), "obj", "5", "9.0", "1.388", f"{gap:+.1f}", "0.000", f"{gap:+.1f}", f"{gap:+.1f}", "2.6", "+0.1",
                               "2.8", "0.9", "1.0", f"{ep:012d}"]))
    rows.append("不按位移筛静置，全部集都进拟合；只按 10/10 登记的 --offset-gate 判读")
    if gate:
        rows.append(f"OFFSET_GATE_{verdict}（地面上 14 集，常数模型 o = {o:+.1f} ± {se:.1f} mm，每集标准差 {sd:.1f} mm，门槛 |o| ≤ 30 mm）")
    if done:
        rows.append("T3_FLOAT_FACTS_DONE")
    return "\n".join(rows) + "\n"


def run(**recs):
    with tempfile.TemporaryDirectory() as t:
        args = []
        for name, text in recs.items():
            p = Path(t) / f"{name}.txt"
            p.write_text(text, encoding="utf-8")
            args.append(f"{name}={p}")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            P.main(args)
        return buf.getvalue().rstrip().splitlines()[-1]


def test_none_and_single():
    assert run(floor=record("FAIL", 80.0, 10.0, 40.0, 5, 5), f830=record("FAIL", 60.0, 10.0, 40.0, 5, 5)) == "PICK_NONE"
    assert run(floor=record("FAIL", 80.0, 10.0, 40.0, 5, 5), f830=record("PASS", 10.0, 10.0, 40.0, 5, -8)) == "PICK f830"


def test_both_pass():
    # |o| 之差 25 > 2·√(5²+5²) ≈ 14.1：取 |o| 小的 floor，尽管它的 ep8 间隙更大
    assert run(floor=record("PASS", -2.0, 5.0, 30.0, 40, 5), f830=record("PASS", 27.0, 5.0, 30.0, 3, 3)) == "PICK floor"
    # |o| 之差 5 ≤ 2·√(10²+10²) ≈ 28.3：看 ep8、ep24 间隙绝对值的较大者，floor 是 40，f830 是 12，取 f830
    assert run(floor=record("PASS", 5.0, 10.0, 40.0, 40, 5), f830=record("PASS", -10.0, 10.0, 40.0, 3, -12)) == "PICK f830"


def test_blocked():
    assert run(floor=record("PASS", 5.0, 10.0, 40.0, 61, 5), f830=record("FAIL", 90.0, 10.0, 40.0, 3, 3)) == "PICK_BLOCKED floor"


def test_bad_records_refuse():
    for kw in ({"done": False}, {"gate": False}, {"ep8": False}):
        try:
            run(floor=record("PASS", 5.0, 10.0, 40.0, 5, 5, **kw))
            raise AssertionError(f"应当拒绝：{kw}")
        except SystemExit as e:
            assert "t3_pick_version" in str(e), (kw, e)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("ALL_OK")
