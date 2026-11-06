"""比较两份 `--cfg job --resolve` 输出的 YAML：列出所有不同的键，并标出不在允许清单里的差异。

在 FlashCHORD 容器里跑（容器里有 PyYAML）。用法：python cfg_diff.py <官方配方.yaml> <本次运行.yaml> <允许的键前缀,逗号分隔>
退出码：0 = 只有允许的差异；1 = 有意料之外的差异。
"""
import sys
import yaml


def flat(d, prefix=""):
    # 列表按下标展开（curriculum.stages.8.immediate_first_frame_probability 这类键，计划 agent-doc/plan-t2ft-2026-10-26.md 复核 2）
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            out.update(flat(v, f"{prefix}{k}."))
    elif isinstance(d, list) and d:
        for i, v in enumerate(d):
            out.update(flat(v, f"{prefix}{i}."))
    else:
        out[prefix[:-1]] = d
    return out


base = flat(yaml.safe_load(open(sys.argv[1])))
run = flat(yaml.safe_load(open(sys.argv[2])))
allowed = [a for a in sys.argv[3].split(",") if a]
unexpected = 0
for k in sorted(set(base) | set(run)):
    b, r = base.get(k, "<无>"), run.get(k, "<无>")
    if b == r:
        continue
    ok = any(k == a or k.startswith(a + ".") for a in allowed)
    unexpected += not ok
    print(f"  {'允许' if ok else '意外'}  {k}: 配方={b!r} → 本次={r!r}")
print(f"  共 {len(run)} 个键；与官方配方不同 {sum(1 for k in set(base) | set(run) if base.get(k, '<无>') != run.get(k, '<无>'))} 个，其中意外 {unexpected} 个")
sys.exit(1 if unexpected else 0)
