"""列出镜像里 Python 环境的全部依赖（名字==版本，排序），fc_run.sh 取其 sha256 作为依赖 freeze 指纹（ADR-00003）。"""
import importlib.metadata as m

print("\n".join(sorted({f"{d.metadata['Name']}=={d.version}" for d in m.distributions()})))
