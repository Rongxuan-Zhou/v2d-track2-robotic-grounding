"""档位隔离的机制守卫：调用 forbid_tier1() 之后，本进程任何读取 Tier 1 目录的动作都会抛错。

CLAUDE.md 要求档位隔离"要有机制守，不能只靠自觉"。规则原文：Tier 2/3 的方法要用
"anything other than the clean trajectory captions provided in Tier 1"。
提交用的 Tier 2/3 管线在入口处调用 forbid_tier1()。

覆盖两条读取路径：
  1. Python 层（open / os.listdir / os.scandir / glob.glob，覆盖 np.load、json、trimesh、glob）：audit hook；
  2. pyarrow 的 C++ 文件系统（pq.read_table / ParquetFile），audit hook 拦不到：包装这两个入口。
audit hook 一旦加上就无法移除（Python 的设计），所以只在提交管线的进程里调用。
"""
import os
import sys

TIER1_DIRNAME = "tier_1_multiview_caption"


class Tier1AccessError(RuntimeError):
    """故意不继承 OSError：glob 等代码会 except OSError 把错误吞掉、静默返回空结果（Python 3.10 实测）。"""


def _check(path):
    try:
        p = os.fsdecode(path)
    except TypeError:
        return
    if TIER1_DIRNAME in p:
        raise Tier1AccessError(f"档位隔离：Tier 2/3 管线试图读取 Tier 1 文件 {p}")


def forbid_tier1():
    def hook(event, args):
        if event in ("open", "os.listdir", "os.scandir", "glob.glob", "glob.glob/2") and args:
            _check(args[0])
    sys.addaudithook(hook)
    try:
        import pyarrow.parquet as pq
    except ImportError:   # 没装 pyarrow 的解释器（例如 ml40 宿主）读不了 parquet，只需 audit hook
        return
    for name in ("read_table", "ParquetFile"):
        orig = getattr(pq, name)
        if getattr(orig, "_tier_guarded", False):
            continue
        def guarded(source, *a, _orig=orig, **k):
            if isinstance(source, (str, bytes, os.PathLike)):
                _check(source)
            return _orig(source, *a, **k)
        guarded._tier_guarded = True
        setattr(pq, name, guarded)
