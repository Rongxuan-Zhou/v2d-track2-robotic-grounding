"""trial_attribution.py 的测试：按造的 --by-stage 输出核对第 6–8 段的平均与归因条件。

守 agent-doc/plan-step3-recipe-trial-2026-10-04.md 第 4 节“只报、作归因”的条件（终止:物体 ≥ 基线 2 倍且进度低 0.1 以上）。
"""

import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HEAD = "  段 | 步数范围 | voc | 进度 | 终止:物体"


def block(run, term, prog):
    rows = [f"  {s} | x | 0 | {prog} | {term}" for s in range(9)]
    return "\n".join([f"## {run}", f"== run-{run}.wandb：history 9 行", HEAD, *rows])


def run_case(tmp_path, var_term, var_prog):
    eps = [{"ep": e, "base42": f"b{e}", "base43": f"c{e}", "var42": f"v{e}", "var43": f"w{e}"} for e in (1, 2)]
    text = []
    for e in eps:
        text += [block(e["base42"], 0.1, 0.9), block(e["base43"], 0.1, 0.9),
                 block(e["var42"], var_term, var_prog), block(e["var43"], var_term, var_prog)]
    f = tmp_path / "bs.txt"
    f.write_text("\n".join(text), encoding="utf8")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"episodes": eps}), encoding="utf8")
    out = subprocess.run([sys.executable, str(HERE / "trial_attribution.py"), str(f), str(spec)],
                         capture_output=True, text=True, check=True).stdout
    return out


def test_attribution_hit(tmp_path):
    out = run_case(tmp_path, 0.25, 0.7)
    assert "ATTRIBUTION 朝向终止过严或参考做不到" in out
    assert "2.50" in out


def test_attribution_miss(tmp_path):
    out = run_case(tmp_path, 0.25, 0.85)
    assert "不满足" in out
