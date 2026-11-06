"""把一份 4096 世界评测 JSON 记成与 fc_eval 记录同样的 summary_json 行（宿主机 python 即可，只用标准库，只读）。
计划 agent-doc/plan-t3static-2026-10-21.md 执行记录：t3m1022a 里 ep9、10、11、28 用的是第 4096000 步的占位 run，评测只在 runs/flash_chord/<run>/eval_<步数>_w4096.json 里，
results/ 下没有记录；stages/evaluate/t3static_select.py 读 results/**/eval_<run>_s<步数>.txt 的 summary_json 行。
用法（仓库根）：python3 stages/evaluate/eval_json_record.py <run> <步数>   最后一行 EVAL_JSON_RECORD_DONE
"""
import json
import sys
from pathlib import Path

run, step = sys.argv[1], sys.argv[2]
p = Path("runs/flash_chord") / run / f"eval_{step}_w4096.json"
d = json.loads(p.read_text())
print(f"== {p}")
print(f"  checkpoint {d.get('checkpoint')}  sha256 {str(d.get('checkpoint_sha256', ''))[:16]}…")
print(f"  world_count {d.get('world_count')}  step_count {d.get('step_count')}  non_finite_worlds {d.get('non_finite_worlds')}")
print("  summary_json " + json.dumps(d, ensure_ascii=False))
print("EVAL_JSON_RECORD_DONE")
