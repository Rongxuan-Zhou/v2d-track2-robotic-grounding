"""fc_run.sh 的 r3o 守卫测试（守 docs/contracts/policy.md 的 POLICY-I010；计划 agent-doc/plan-r3o-2026-10-15.md 2.1，代码审查第 7 条）。只用标准库。

把 fc_run.sh 拷进临时目录（它先 cd 到仓库根，守卫在参数检查之后、建目录之前），造两份假副本：SOURCE_V2D.txt 带 object-orientation 的
（打了物体朝向补丁）与不带的。run 名带 -r3o-、设了 V2D_OBJ_ORI_WEIGHT、副本打了补丁，三样同时具备或同时没有才过守卫；过了守卫的
会建 runs/flash_chord/<run>（之后因为没有 parquet、或 Mac 上没有 flock 而停），没过的报 “r3o 的 run 名” 那一句、退出码 2、不建目录。
用法：python3 stages/policy/test_fc_run_r3o_guard.py
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MSG = "r3o 的 run 名"


def run(root: Path, name: str, src, weight) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("V2D_")}
    if src:
        env["V2D_FC_SRC"] = str(root / src)
    if weight is not None:
        env["V2D_OBJ_ORI_WEIGHT"] = weight
        env["V2D_OBJ_ORI_STD"] = "0.4"
    return subprocess.run(["bash", str(root / "stages/policy/fc_run.sh"), name, "/repo/none/data.parquet", "8", "64"],
                          cwd=root, env=env, capture_output=True, text=True, timeout=60)


def test_guard() -> None:
    with tempfile.TemporaryDirectory() as t:
        root = Path(t)
        (root / "stages/policy").mkdir(parents=True)
        shutil.copy(REPO / "stages/policy/fc_run.sh", root / "stages/policy/fc_run.sh")
        for d, label in (("fc_oo", "flash_chord v0.4.0 + a.patch + b.patch + third_party/patches/flash_chord-v0.4.0-object-orientation.patch"),
                         ("fc_plain", "flash_chord v0.4.0 + a.patch + third_party/patches/flash_chord-v0.4.0-nonfinite-guard.patch")):
            (root / d).mkdir()
            (root / d / "SOURCE_V2D.txt").write_text(label + "\n")
        cases = [
            ("t1-ep20-w2048-cap4m-r3o-mf4090", "fc_oo", "1.0", True),      # 三样都有
            ("t1-ep20-w2048-cap4m-r3o-mf4090", "fc_plain", "1.0", False),  # 普通副本上训 r3o
            ("t1-ep20-w2048-cap4m-r3o-mf4090", "fc_oo", None, False),      # 没设权重
            ("t1-ep20-w2048-cap4m-r3o-mf4090", None, "1.0", False),       # 没挂副本
            ("t1-ep20-w2048-cap4m-r3a-mf4090", "fc_oo", "1.0", False),     # 名字不是 r3o
            ("t1-ep20-w2048-cap4m-r3a-mf4090", "fc_oo", None, False),      # 普通 run 挂了 oo 副本
            ("t1-ep20-w2048-cap4m-r3a-mf4090", "fc_plain", None, True),    # 普通 run、普通副本
            ("t1-ep20-w2048-cap4m-r3a-mf4090", None, None, True),          # 不挂副本（镜像里的源码）
        ]
        for name, src, weight, ok in cases:
            shutil.rmtree(root / "runs", ignore_errors=True)
            p = run(root, name, src, weight)
            made = (root / "runs/flash_chord" / name).is_dir()
            if ok:
                assert MSG not in p.stderr and made, (name, src, weight, p.stderr)
            else:
                assert p.returncode == 2 and MSG in p.stderr and not made, (name, src, weight, p.returncode, p.stderr)


if __name__ == "__main__":
    test_guard()
    print("ok  test_guard")
    print("ALL_OK")
