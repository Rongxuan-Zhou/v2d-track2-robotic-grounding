"""check_t2_select.py 的测试：守 agent-doc/plan-t2-select-b-2026-10-12.md 第 2 版第 3 节的“选中者核对”（审查 M1）。只用标准库。
造 3 条 episode 的选择清单、评测记录、导出 JSON 与 CSV：全部一致 → OK；导出 JSON 的 checkpoint 不同、world_count 不是 1、
CSV 里一条 episode 有两个 checkpoint、CSV 多一条 episode → 各自 BAD，退出码 1。
用法：python3 stages/evaluate/test_check_t2_select.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
STEP = "349999104"
HEAD = "row_id,pos_x,pos_y,pos_z,quat_x,quat_y,quat_z,quat_w,eval_script_sha256,eval_code_sha256,checkpoint_sha256,packer_script_sha256,uploader_script_sha256,code_commit_url"


def setup(root: Path, json_bad=None, world_bad=None, csv_two=None, csv_extra=False):
    ev, res = root / "evals", root / "res"
    ev.mkdir()
    res.mkdir()
    runs = {0: "t2-ep0-a-ml40", 1: "t2-ep1-b-mf4090", 2: "t2-ep2-c-ml40"}
    (root / "sel.tsv").write_text("\n".join(f"2\t{e}\t{r}\t{STEP}\t{r.rsplit('-', 1)[1]}" for e, r in runs.items()) + "\n")
    rows = [HEAD]
    for e, r in runs.items():
        sha = f"{e}" * 64
        j = {"checkpoint": f"/repo/runs/flash_chord/{r}/policy_{STEP}.safetensors", "checkpoint_sha256": sha, "environment_steps": int(STEP)}
        (ev / f"eval_{r}_s{STEP}.txt").write_text("summary_json " + json.dumps(j) + "\n")
        (res / f"episode_{e:06d}.json").write_text(json.dumps({"checkpoint_sha256": "f" * 64 if e == json_bad else sha,
                                                               "world_count": 2 if e == world_bad else 1}))
        for fr in range(3):
            s = "e" * 64 if (e == csv_two and fr == 2) else sha
            rows.append(f"track_2/e{e:06d}/w000/f{fr:06d}/o0,0,0,0,0,0,0,1,x,y,{s},p,u,c")
    if csv_extra:
        rows.append(f"track_2/e000009/w000/f000000/o0,0,0,0,0,0,0,1,x,y,{'9' * 64},p,u,c")
    (root / "t.csv").write_text("\n".join(rows) + "\n")


def run(root: Path, with_csv=True):
    args = [sys.executable, str(HERE / "check_t2_select.py"), "--selection", str(root / "sel.tsv"), "--evals", str(root / "evals"),
            "--results", str(root / "res")] + (["--csv", str(root / "t.csv")] if with_csv else [])
    return subprocess.run(args, capture_output=True, text=True)


def test_all_consistent():
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        setup(root)
        r = run(root)
        assert r.returncode == 0 and r.stdout.strip().endswith("CHECK_T2_SELECT_OK"), r.stdout + r.stderr
        assert run(root, with_csv=False).returncode == 0


def test_mismatches_flagged():
    for kw in ({"json_bad": 1}, {"world_bad": 2}, {"csv_two": 0}, {"csv_extra": True}):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            setup(root, **kw)
            r = run(root)
            assert r.returncode == 1 and "CHECK_T2_SELECT_BAD" in r.stdout, (kw, r.stdout + r.stderr)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok {name}")
    print("ALL_OK")
