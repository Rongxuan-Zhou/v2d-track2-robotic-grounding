"""按完整键路径核对 `--cfg job --resolve` 展开的配置（fc_config_check.sh 写的 runs/cfgcheck/<run>.yaml）。在 FlashCHORD 容器里跑（要 PyYAML）。

配置核对（cfg_diff.py）只拦“意外的差异”，漏写一个覆盖时差异为零照样通过；这里做正向核对：训练链开跑前确认每条 run 的配置确实是期望的样子。
断言写法（可以给多个）：
  key=value    键存在且等于 value（value 按 YAML 标量解析：true / false / 数字 / 字符串）
  key~=text    键存在、是字符串且包含 text（例如 task.parquet~=sequence_id=<序列>/robot_name=g1/data.parquet）
  !key         键不存在
任何一条不满足就逐条列出、退出码 1。
用法：python /repo/stages/policy/assert_cfg.py <yaml> <断言>...
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


def main() -> int:
    path, checks = sys.argv[1], sys.argv[2:]
    cfg = flat(yaml.safe_load(open(path)))
    bad = []
    for c in checks:
        if c.startswith("!"):
            if c[1:] in cfg:
                bad.append(f"{c[1:]} 不该存在，实际 {cfg[c[1:]]!r}")
        elif "~=" in c:
            k, text = c.split("~=", 1)
            if not (isinstance(cfg.get(k), str) and text in cfg[k]):
                bad.append(f"{k} 应包含 {text!r}，实际 {cfg.get(k, '<无>')!r}")
        elif "=" in c:
            k, v = c.split("=", 1)
            want = yaml.safe_load(v)
            if k not in cfg or cfg[k] != want or type(cfg[k]) is not type(want):
                bad.append(f"{k} 应为 {want!r}，实际 {cfg.get(k, '<无>')!r}")
        else:
            bad.append(f"看不懂的断言：{c}")
    print(f"{path}：{len(checks)} 条断言，{'全过' if not bad else f'{len(bad)} 条不符'}")
    for b in bad:
        print(f"  不符  {b}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
