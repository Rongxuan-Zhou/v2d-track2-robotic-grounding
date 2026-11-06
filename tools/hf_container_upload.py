# 把提交容器与技术报告传到 HF 公开 dataset 仓库（训练机上跑，经 tools/hf_remote.sh 带凭据；凭据只从环境变量来）。
# 先建仓库（public，已存在且可见性不同就停），再传 README.md、技术报告 PDF、容器 tar.zst 与它的 sha256 文件，
# 传完列出仓库文件与大小，核对容器文件的大小与本地相同。用 huggingface_hub 的 HfApi（2.x，有 hf_xet 时走 xet）。
# 用法：python tools/hf_container_upload.py <用户/仓库> <目录>，目录里放 README.md、*.pdf、*.tar.zst、*.tar.zst.sha256。
# 最后一行 HF_CONTAINER_UPLOAD_DONE 或 HF_CONTAINER_UPLOAD_FAILED <原因>。
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

repo, d = sys.argv[1], Path(sys.argv[2])
if not os.environ.get("HF_TOKEN"):
    sys.exit("HF_CONTAINER_UPLOAD_FAILED 没有 HF_TOKEN")
api = HfApi()
api.create_repo(repo, repo_type="dataset", private=False, exist_ok=True)
info = api.repo_info(repo, repo_type="dataset")
if info.private:
    sys.exit("HF_CONTAINER_UPLOAD_FAILED 仓库是私有的，不是公开")
files = sorted(p for p in d.iterdir() if p.is_file())
need = [p for p in files if p.name == "README.md" or p.suffix == ".pdf" or p.name.endswith(".tar.zst") or p.name.endswith(".sha256")]
assert any(p.name == "README.md" for p in need), "缺 README.md"
for p in sorted(need, key=lambda p: p.stat().st_size):
    print(f"上传 {p.name}（{p.stat().st_size} B）", flush=True)
    api.upload_file(path_or_fileobj=str(p), path_in_repo=p.name, repo_id=repo, repo_type="dataset",
                    commit_message=f"add {p.name}")
remote = {f.path: f.size for f in api.list_repo_tree(repo, repo_type="dataset") if hasattr(f, "size")}
bad = [p.name for p in need if remote.get(p.name) != p.stat().st_size]
for k, v in sorted(remote.items()):
    print(f"{k}\t{v}")
if bad:
    sys.exit(f"HF_CONTAINER_UPLOAD_FAILED 大小不符：{bad}")
print(f"https://huggingface.co/datasets/{repo}")
print("HF_CONTAINER_UPLOAD_DONE")
