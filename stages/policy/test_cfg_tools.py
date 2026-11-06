"""assert_cfg.py 与 cfg_diff.py 按下标展开列表（计划 agent-doc/plan-t2ft-2026-10-26.md 复核 2）。要 PyYAML，在 FlashCHORD 容器里跑。
用法：python /repo/stages/policy/test_cfg_tools.py   最后一行 ALL_OK
"""
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BASE = """curriculum:
  thresholds: [10, 20]
  stages:
  - voc_scale: 1.0
  - voc_scale: 0.0
    immediate_first_frame_probability: 0.5
training:
  curriculum:
    thresholds: [10, 20]
    stages:
    - voc_scale: 1.0
    - voc_scale: 0.0
      immediate_first_frame_probability: 0.5
  optimization:
    learning_rate_schedule_updates: null
empty: []
"""


def run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(HERE / script), *args], capture_output=True, text=True)


def main() -> int:
    with tempfile.TemporaryDirectory() as d:
        base, ft = Path(d) / "base.yaml", Path(d) / "ft.yaml"
        base.write_text(BASE)
        ft.write_text(BASE.replace("immediate_first_frame_probability: 0.5", "immediate_first_frame_probability: 1.0")
                      .replace("learning_rate_schedule_updates: null", "learning_rate_schedule_updates: 683592"))
        # 1. 正向断言能读到列表元素里的键；空列表照旧是一个键
        r = run("assert_cfg.py", str(ft), "curriculum.stages.1.immediate_first_frame_probability=1.0",
                "training.curriculum.stages.1.immediate_first_frame_probability=1.0", "curriculum.thresholds.1=20",
                "training.optimization.learning_rate_schedule_updates=683592", "empty=[]", "!curriculum.stages.2.voc_scale")
        assert r.returncode == 0, r.stdout + r.stderr
        r = run("assert_cfg.py", str(base), "curriculum.stages.1.immediate_first_frame_probability=1.0")
        assert r.returncode == 1 and "0.5" in r.stdout, r.stdout
        # 2. 差异按下标列出：放行 curriculum 与 training.curriculum 时通过，只放行 curriculum 时 training 那一份算意外
        allowed = "curriculum.stages.1.immediate_first_frame_probability,training.curriculum.stages.1.immediate_first_frame_probability,training.optimization.learning_rate_schedule_updates"
        r = run("cfg_diff.py", str(base), str(ft), allowed)
        assert r.returncode == 0 and "意外 0 个" in r.stdout, r.stdout
        r = run("cfg_diff.py", str(base), str(ft), "curriculum.stages.1.immediate_first_frame_probability,training.optimization.learning_rate_schedule_updates")
        assert r.returncode == 1 and "意外  training.curriculum.stages.1.immediate_first_frame_probability" in r.stdout, r.stdout
        # 3. 同一份配置自比没有差异
        r = run("cfg_diff.py", str(base), str(base), "")
        assert r.returncode == 0, r.stdout
    print("ALL_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
