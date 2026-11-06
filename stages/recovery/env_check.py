import sys, os, glob, importlib.metadata as md
print("=== python ===", sys.executable)
print("=== sys.path ===")
for p in sys.path[:8]:
    if p: print("   ", p)
print()
for name in ("torch", "soma"):
    try:
        m = __import__(name)
        print(f"=== {name} ===")
        print("   file:", m.__file__)
        print("   ver :", getattr(m, "__version__", "n/a"))
    except Exception as e:
        print(f"=== {name} === IMPORT FAILED: {e}")
print()
try:
    import torch
    print("torch cuda available:", torch.cuda.is_available())
except Exception as e:
    print("torch probe failed:", e)
print()
print("=== 安装的 dist-info（查重复/影子）===")
for pat in ("torch-*", "py_soma_x-*"):
    for root in sys.path:
        if not root: continue
        for p in glob.glob(os.path.join(root, pat)):
            print("   ", p)
print()
for d in ("torch", "py-soma-x"):
    try: print(f"   metadata {d}: {md.version(d)}")
    except Exception as e: print(f"   metadata {d}: {e}")
