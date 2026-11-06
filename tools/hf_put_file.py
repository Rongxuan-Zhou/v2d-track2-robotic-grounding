# 往 HF dataset 仓库传一个文件（覆盖同名文件），传完核对远端大小（训练机上跑，经 tools/hf_remote.sh 带凭据）。
# 用法：python tools/hf_put_file.py <用户/仓库> <本地文件> <仓库内路径>。最后一行 HF_PUT_DONE 或 HF_PUT_FAILED <原因>。
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

repo, local, dest = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
if not os.environ.get("HF_TOKEN"):
    sys.exit("HF_PUT_FAILED 没有 HF_TOKEN")
api = HfApi()
api.upload_file(path_or_fileobj=str(local), path_in_repo=dest, repo_id=repo, repo_type="dataset", commit_message=f"update {dest}")
size = {f.path: f.size for f in api.list_repo_tree(repo, repo_type="dataset") if hasattr(f, "size")}.get(dest)
if size != local.stat().st_size:
    sys.exit(f"HF_PUT_FAILED 远端大小 {size}，本地 {local.stat().st_size}")
print(f"{dest}\t{size}")
print("HF_PUT_DONE")
