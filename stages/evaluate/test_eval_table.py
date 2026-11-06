"""eval_table.py 的测试（暂无契约 ID：运行规范待成文）。只用标准库。

临时目录里放两份评测文本（各一行 summary_json），核：
  1. 不加 --mean：表头加两行，没有平均行；每行的小数位同原来（MPPE 两位、朝向一位、其余三位）；
  2. 加 --mean：最后一行“平均（2 条，等权）”，按原始值求平均：0.1235 与 0.1236 的原始平均 0.12355 打成 0.124；
     若先按表里打印的 0.123、0.124 再平均，会得 0.123。失败原因缺的键按 0 算；
  3. 文件里没有 summary_json 行：退出，报文件名。
用法：python3 stages/evaluate/test_eval_table.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def summary(run: str, mp: float, mppe: float, fail: dict) -> str:
    j = {"checkpoint": f"/repo/runs/flash_chord/{run}/policy_349999104.safetensors", "environment_steps": 349999104, "world_count": 4096,
         "step_count": 700, "maniptrans_sr": mp, "spider_sr_uncentered": 0.5, "add_auc": 0.25, "mppe_cm": mppe,
         "object_position_error_m": 0.0123, "object_orientation_error_deg": 4.56, "chord_sr": 0.0,
         "reference_tracking": {"failure_cause_fraction": fail}, "checkpoint_sha256": "ab" * 32}
    return "fc_eval: 完成\nsummary_json " + json.dumps(j) + "\n"


def table(*args) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(REPO / "stages/evaluate/eval_table.py"), *map(str, args)], capture_output=True, text=True, timeout=60)


def test_rows_and_mean() -> None:
    with tempfile.TemporaryDirectory() as t:
        a, b = Path(t) / "a.txt", Path(t) / "b.txt"
        a.write_text(summary("t3-ep1-x", 0.1235, 10.0, {"pelvis": 0.2, "object": 0.6, "palm": 1.0}), encoding="utf-8")
        b.write_text(summary("t3-ep2-x", 0.1236, 21.0, {"object": 1.0}), encoding="utf-8")
        r = table(a, b)
        lines = r.stdout.splitlines()
        assert r.returncode == 0 and len(lines) == 3 and not any(l.startswith("平均") for l in lines), r.stdout
        assert lines[1].split("\t") == ["t3-ep1-x", "349999104", "4096", "700", "0.123", "0.500", "0.250", "10.00", "1.23", "4.6", "0.000",
                                        "0.200", "0.600", "1.000", "abababababab"], lines[1]
        r = table("--mean", a, b)
        last = r.stdout.splitlines()[-1].split("\t")
        assert r.returncode == 0 and last[0] == "平均（2 条，等权）" and last[1:4] == ["", "", ""], last
        assert last[4:14] == ["0.124", "0.500", "0.250", "15.50", "1.23", "4.6", "0.000", "0.100", "0.800", "0.500"], last
        assert last[14] == "", last


def test_missing_summary() -> None:
    with tempfile.TemporaryDirectory() as t:
        a = Path(t) / "a.txt"
        a.write_text("no summary here\n", encoding="utf-8")
        r = table("--mean", a)
        assert r.returncode != 0 and "a.txt 里没有 summary_json 行" in r.stderr, r.stderr


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("ALL_OK")
