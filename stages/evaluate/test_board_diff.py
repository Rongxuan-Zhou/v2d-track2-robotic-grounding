"""board_diff.py 的测试：守 agent-doc/board-watch-2026-10-19.md 第 2 节的比较规则。只用标准库。
造两份 12 个榜的快照：完全相同 → BOARD_NO_CHANGE、没有 ALERT；FF 在 tier2-sp-sr 超过我们 → 那个榜记变化、打 ALERT；
新队排在我们后面 → 记变化、不打名次 ALERT；只有 11 个榜 → 退出码非 0。
用法：python3 stages/evaluate/test_board_diff.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
METRICS = ("auc", "sp-sr", "mp-sr", "mppe")
BASE = {"maniphys.ai": 0.7, "Xueyi Wang (FF)": 0.6, "CHORD Baseline": 0.5, "Other Team": 0.3}


def snapshot(path: Path, change=None, extra=None, drop_last=False):
    out = ["# command  : test", "# ---------------------------------------------------------------"]
    keys = [(t, m) for t in (1, 2, 3) for m in METRICS]
    if drop_last:
        keys = keys[:-1]
    for t, m in keys:
        out.append(f"== v2d-challenge-track2-tier{t}-{m}")
        out.append("  teamId  teamName          submissionDate              score    ")
        out.append("--------  ----------------  --------------------------  -------  ")
        scores = dict(BASE)
        if change and change[0] == (t, m):
            scores.update(change[1])
        if extra and extra[0] == (t, m):
            scores.update(extra[1])
        if m == "mppe":
            scores = {k: 1 / v for k, v in scores.items()}
        for i, (team, s) in enumerate(scores.items()):
            out.append(f"{17000000 + i}  {team:<16}  2026-10-1{i} 01:02:03.000000  {s:.5f}  ")
    path.write_text("\n".join(out) + "\n", encoding="utf-8")


def run(old: Path, new: Path):
    p = subprocess.run([sys.executable, str(HERE / "board_diff.py"), str(old), str(new)], capture_output=True, text=True)
    return p.returncode, p.stdout


def main():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        snapshot(d / "a.txt")
        snapshot(d / "b.txt")
        rc, out = run(d / "a.txt", d / "b.txt")
        assert rc == 0 and "BOARD_NO_CHANGE" in out and "ALERT" not in out, out

        snapshot(d / "c.txt", change=((2, "sp-sr"), {"Xueyi Wang (FF)": 0.8}))
        rc, out = run(d / "a.txt", d / "c.txt")
        assert rc == 0 and "BOARD_CHANGED 1 个榜" in out and "ALERT tier2-sp-sr 名次 1 → 2" in out, out

        snapshot(d / "e.txt", extra=((3, "auc"), {"New Team": 0.4}))
        rc, out = run(d / "a.txt", d / "e.txt")
        assert rc == 0 and "BOARD_CHANGED 1 个榜" in out and "新进 New Team" in out and "ALERT tier" not in out, out

        snapshot(d / "f.txt", drop_last=True)
        rc, out = run(d / "a.txt", d / "f.txt")
        assert rc != 0, out
    print("ALL_OK")


if __name__ == "__main__":
    main()
