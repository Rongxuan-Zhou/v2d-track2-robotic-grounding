"""p2_compare.py 的测试（守计划 agent-doc/plan-t3-simreg-p2-2026-10-11.md 第 2 节第 4 步：两版 P2′ 表逐集并列，只报；暂无契约 ID）。只用标准库。
  1. 两版都有的集逐集并列，间隙之差按 B−A 算；结论都一样时打 SAME_VERDICT n/n，最大差与超过 1 mm 的集照报
  2. 一集的结论变了：打 VERDICT_CHANGED；“失败” 行（数值列是 “-”）不算进间隙之差
  3. 只在一版里有的集打 ONLY_<名>
  4. 没有表头、同一集出现两次、参数不是 <名>=<路径>：报错
  5. 真记录（若在）：results/t3p2/p2_18.txt 与 results/t3simreg/p2_17.txt 能读，ep21 只在 f830 里
用法：python3 stages/retarget/test_p2_compare.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
HEAD = "ep\t序列\t帧\t左手接触\t右手接触\tIK中位\t间隙mm\t支撑面\t状态\t脚底Z_m\tA段面高_m\tA段间隙mm\t差mm\tP2′\t原因"


def table(rows: list) -> str:
    out = ["# command  : python3 stages/retarget/p2_table.py x", "记录 x；P2′：…", HEAD]
    for ep, gap, left, right, verdict in rows:
        if verdict == "失败":
            out.append("\t".join([str(ep)] + ["-"] * 12 + ["失败", "没有结果行"]))
        else:
            out.append("\t".join([str(ep), f"seq{ep}", "500", str(left), str(right), "0.6", f"{gap:.1f}", "floor", "ALIGN",
                                  "+0.0～+0.1", "0.000", "0.0", "0.0", verdict, "-"]))
    return "\n".join(out + ["P2_PASS …"]) + "\n"


def run(*args: str, rc: int = 0) -> str:
    r = subprocess.run([sys.executable, str(HERE / "p2_compare.py"), *args], capture_output=True, text=True, timeout=60)
    assert r.returncode == rc, (args, r.returncode, r.stdout, r.stderr)
    return r.stdout + r.stderr


def main() -> int:
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        (d / "a.txt").write_text(table([(0, -23.4, 280, 455, "过"), (12, -63.5, 320, 423, "不过"), (21, -26.1, 449, 462, "过")]))
        (d / "b.txt").write_text(table([(0, -23.4, 284, 458, "过"), (12, -64.4, 377, 414, "不过"), (5, 0, 0, 0, "失败")]))
        out = run(f"f830={d / 'a.txt'}", f"simreg={d / 'b.txt'}")
        assert "SAME_VERDICT 2/2" in out and "MAX_ABS_GAP_DIFF 0.9 mm（ep12）" in out and "GAP_DIFF_GT_1MM 无" in out, out
        assert "12\tseq12\t-63.5\t-64.4\t-0.9\t320\t377\t423\t414" in out, out
        assert "ONLY_f830 ep21" in out and "ONLY_simreg ep5" in out, out
        (d / "c.txt").write_text(table([(0, -35.0, 284, 458, "不过"), (12, 0, 0, 0, "失败")]))
        out = run(f"f830={d / 'a.txt'}", f"x={d / 'c.txt'}")
        assert "VERDICT_CHANGED ep0 ep12（2/2）" in out and "MAX_ABS_GAP_DIFF 11.6 mm（ep0）" in out and "GAP_DIFF_GT_1MM ep0" in out, out
        assert "12\tseq12\t-63.5\t-\t-\t" not in out and "\t-\t-\t320\t-\t" in out, out
        (d / "bad.txt").write_text("no header\n")
        assert "没有表头" in run(f"a={d / 'a.txt'}", f"b={d / 'bad.txt'}", rc=1)
        (d / "dup.txt").write_text(table([(0, 1.0, 1, 1, "过"), (0, 2.0, 1, 1, "过")]))
        assert "出现两次" in run(f"a={d / 'a.txt'}", f"b={d / 'dup.txt'}", rc=1)
        assert "用法" in run(str(d / "a.txt"), str(d / "b.txt"), rc=1)
    old, new = REPO / "results/t3p2/p2_18.txt", REPO / "results/t3simreg/p2_17.txt"
    if old.exists() and new.exists():
        out = run(f"f830={old}", f"simreg={new}")
        assert "ONLY_f830 ep21" in out and "ONLY_simreg 无" in out, out
    print("ALL_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
