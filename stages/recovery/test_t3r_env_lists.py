"""stages/recovery/t3r_env_lists.py 的测试：守计划 agent-doc/plan-t3takeover-1b-2026-10-21.md 判据 E0（复核改动第 5 条：按 RECORD 比，不只比名字与版本）。
用法：python3 stages/recovery/test_t3r_env_lists.py（也可用 pytest 跑）。最后一行 T3R_ENV_LISTS_TESTS_OK。
"""
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from t3r_env_lists import scan

TOOL = HERE / "t3r_env_lists.py"


def _dist(venv: Path, name: str, version: str, files: dict[str, str], extra: dict[str, str] | None = None) -> None:
    sp = venv / "lib/python3.11/site-packages"
    d = sp / f"{name}-{version}.dist-info"
    d.mkdir(parents=True, exist_ok=True)
    (d / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n\nbody\n")
    rows = [f"{p},sha256={h},10" for p, h in files.items()]
    rows += [f"{d.name}/METADATA,sha256=meta{version},10", f"{d.name}/RECORD,,"]
    for k, v in (extra or {}).items():
        (d / k).write_text(v)
        rows.append(f"{d.name}/{k},sha256=x{v},1")
    (d / "RECORD").write_text("\n".join(rows) + "\n")


def test_digest_ignores_install_side_files():
    """E0：摘要不看 bin 下的脚本、pyc、INSTALLER、REQUESTED、direct_url.json——两个环境装法不同、内容相同时摘要相同。"""
    with tempfile.TemporaryDirectory() as d:
        a, b = Path(d) / "a", Path(d) / "b"
        _dist(a, "Foo_Bar", "1.0", {"foo/__init__.py": "h1", "../../../bin/foo": "s1", "foo/__pycache__/x.cpython-311.pyc": "p1"},
              {"INSTALLER": "pip\n", "REQUESTED": ""})
        _dist(b, "foo-bar", "1.0", {"foo/__init__.py": "h1", "../../../bin/foo": "s2", "foo/__pycache__/x.cpython-311.pyc": "p2"},
              {"INSTALLER": "uv\n", "direct_url.json": "{}"})
        sa, sb = scan(a), scan(b)
        assert set(sa) == set(sb) == {"foo-bar"}
        assert sa["foo-bar"]["digest"] == sb["foo-bar"]["digest"], (sa, sb)


def test_compare_flags_content_version_and_missing():
    """E0：同版本而文件内容不同、版本不同、少一个发行版，--compare 都报出来并给退出码 1；完全相同时退出码 0（对照）。"""
    with tempfile.TemporaryDirectory() as d:
        ref, same, diff = Path(d) / "ref", Path(d) / "same", Path(d) / "diff"
        for v in (ref, same):
            _dist(v, "numpy", "2.3.5", {"numpy/core.so": "h1"})
            _dist(v, "scipy", "1.15.3", {"scipy/x.py": "h2"})
        _dist(diff, "numpy", "2.3.5", {"numpy/core.so": "OTHER"})
        _dist(diff, "torch", "2.7.0+cpu", {"torch/x.py": "h3"})
        p = subprocess.run([sys.executable, str(TOOL), str(same), "--compare", str(ref)], capture_output=True, text=True, check=False)
        assert p.returncode == 0 and "不同 0 个" in p.stdout, p.stdout
        p = subprocess.run([sys.executable, str(TOOL), str(diff), "--compare", str(ref)], capture_output=True, text=True, check=False)
        assert p.returncode == 1, p.stdout
        assert "内容不同\tnumpy" in p.stdout and "缺\tscipy\t我们没有" in p.stdout and "缺\ttorch\t参照没有" in p.stdout, p.stdout


def test_compare_list_matches_compare():
    """E0：拿存下的清单比（--compare-list，记录文件的 # 头与末行跳过），结论与直接读参照 venv 比（--compare）相同。"""
    with tempfile.TemporaryDirectory() as d:
        ref, ours = Path(d) / "ref", Path(d) / "ours"
        _dist(ref, "numpy", "2.3.5", {"numpy/core.so": "h1"})
        _dist(ours, "numpy", "2.3.5", {"numpy/core.so": "h1"})
        listing = subprocess.run([sys.executable, str(TOOL), str(ref)], capture_output=True, text=True, check=True).stdout
        saved = Path(d) / "list.txt"
        saved.write_text("# command  : x\n# ---\n" + listing)
        p = subprocess.run([sys.executable, str(TOOL), str(ours), "--compare-list", str(saved)], capture_output=True, text=True, check=False)
        assert p.returncode == 0 and "不同 0 个" in p.stdout, p.stdout
        _dist(ours, "numpy", "2.3.5", {"numpy/core.so": "CHANGED"})
        p = subprocess.run([sys.executable, str(TOOL), str(ours), "--compare-list", str(saved)], capture_output=True, text=True, check=False)
        assert p.returncode == 1 and "内容不同\tnumpy" in p.stdout, p.stdout


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  {name} 过")
    print("T3R_ENV_LISTS_TESTS_OK")
