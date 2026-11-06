"""p2_vs_theirs.py 的测试（守计划 agent-doc/plan-t3-hm-p2-2026-10-12.md 第 5 节“与提供方记录对照”：只报；暂无契约 ID）。只用标准库。
  1. 各栏都对得上：每集“同”，打 THEIRS_SAME n/n；只在对方那边的集列进 ONLY_THEIRS，不算不同
  2. 间隙差 0.2 mm、接触帧差 1、最近的面不同、状态与对齐不一致、我们的“失败”行：各自算不同，打 THEIRS_DIFF 并列出栏名；间隙差 0.1 mm 不算
  3. 对方记录缺段、同一集出现两次、我们的表没有表头：报错
  4. 真记录（若在）：把 results/t3hm/their_gate_bhm_ml40.txt 里对方用官方支撑面的 p2_bhm 表当“我们的表”，与同一份记录对照，
     不同的应当恰好是间隙或最近的面随支撑面变的 ep3、6、7、20、25（接触、IK、脚底 Z 与支撑面无关，应当全同）
用法：python3 stages/retarget/test_p2_vs_theirs.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
HEAD = "ep\t序列\t帧\t左手接触\t右手接触\tIK中位\t间隙mm\t支撑面\t状态\t脚底Z_m\tA段面高_m\tA段间隙mm\t差mm\tP2′\t原因"


def row(ep, gap="-0.0", left=200, right=300, surface="floor", status="OK", ik="0.700", fail=False) -> str:
    if fail:
        return "\t".join([str(ep)] + ["-"] * 12 + ["失败", "没有结果行"])
    return "\t".join([str(ep), f"seq{ep}", "500", str(left), str(right), ik, gap, surface, status, "+0.0010～+0.1000",
                      "-", "-", "-", "过", "-"])


def ours(rows: list) -> str:
    return "\n".join(["# command  : python3 stages/retarget/p2_table.py x", "记录 x；P2′：…", HEAD] + rows + ["P2_PASS …"]) + "\n"


def theirs(p2_rows: list, sup: list, extra: str = "") -> str:
    out = ["# command  : bash -c …", "==== p2_bhm.txt（2026-10-13 04:27:38 CST，sha256 0）", "记录 logs/rec_bhm.txt；P2′：…", HEAD]
    out += p2_rows + ["P2_PASS …", "==== oursup_bhm.txt（2026-10-13 04:43:51 CST，sha256 0）"]
    out += [f"{ep} RESULT gap_mm={g} surface={s} ok={ok}" for ep, g, s, ok in sup]
    return "\n".join(out) + "\n" + extra


def run(*args: str, rc: int = 0) -> str:
    r = subprocess.run([sys.executable, str(HERE / "p2_vs_theirs.py"), *args], capture_output=True, text=True, timeout=60)
    assert r.returncode == rc, (args, r.returncode, r.stdout, r.stderr)
    return r.stdout + r.stderr


def main() -> int:
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        base_p2 = [row(0), row(3, gap="755.0", status="ALIGN"), row(28)]
        base_sup = [(0, "-0.0", "floor", 1), (3, "-0.0", "support", 1), (28, "-0.0", "floor", 1)]
        (d / "t.txt").write_text(theirs(base_p2, base_sup))
        (d / "o.txt").write_text(ours([row(0), row(3, surface="support")]))
        out = run(str(d / "o.txt"), str(d / "t.txt"))
        assert "THEIRS_SAME 2/2" in out and "ONLY_OURS 无" in out and "ONLY_THEIRS ep28" in out, out
        (d / "o2.txt").write_text(ours([row(0, gap="0.1"), row(3, surface="support", left=201)]))
        out = run(str(d / "o2.txt"), str(d / "t.txt"))
        assert "THEIRS_DIFF ep3（左手接触）（1/2）" in out and "0\tseq0\t500\t500" in out, out
        (d / "o3.txt").write_text(ours([row(0, gap="0.2"), row(3, gap="-0.0", surface="floor", status="ALIGN"), row(28, fail=True)]))
        out = run(str(d / "o3.txt"), str(d / "t.txt"))
        assert "THEIRS_DIFF ep0（间隙mm）；ep3（支撑面、状态）；ep28（帧、左手接触、右手接触、IK中位、脚底Z_m、间隙mm、支撑面、状态）（3/3）" in out, out
        (d / "nosup.txt").write_text(theirs(base_p2, base_sup).split("==== oursup_bhm.txt")[0])
        assert "没有 oursup_bhm.txt 这一段" in run(str(d / "o.txt"), str(d / "nosup.txt"), rc=1)
        (d / "dup.txt").write_text(theirs(base_p2, base_sup + [(0, "-0.0", "floor", 1)]))
        assert "出现两次" in run(str(d / "o.txt"), str(d / "dup.txt"), rc=1)
        (d / "bad.txt").write_text("no header\n")
        assert "没有表头" in run(str(d / "bad.txt"), str(d / "t.txt"), rc=1)
        rec = REPO / "results/t3hm/their_gate_bhm_ml40.txt"
        if rec.exists():
            text = rec.read_text(encoding="utf-8", errors="replace")
            p2 = text.split("==== p2_bhm.txt")[1].split("\n====")[0]
            (d / "p2_bhm_as_ours.txt").write_text(p2.split("\n", 1)[1])
            out = run(str(d / "p2_bhm_as_ours.txt"), str(rec))
            diff = next(x for x in out.splitlines() if x.startswith("THEIRS_"))
            eps = sorted(int(x.split("（")[0]) for x in diff.split(" ", 1)[1].rsplit("（", 1)[0].replace("ep", "").split("；"))
            assert eps == [3, 6, 7, 20, 25], (diff, out)
            assert "接触" not in diff and "IK" not in diff and "脚底" not in diff and "帧" not in diff, diff
            assert "ONLY_OURS 无" in out and "ONLY_THEIRS 无" in out, out
    print("ALL_OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
